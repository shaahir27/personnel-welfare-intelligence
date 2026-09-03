"""Append-only storage for voluntary self-assessment answers.

One job: persist what a person wrote on the check-in screen, and read back
their own submissions.

Why this is not in the pipeline
-------------------------------
Everything else the API serves is precomputed by ``scripts/run_pipeline.py``
and read-only at request time. Check-in answers are the one thing a user
creates, so they need a write path, and that write path must not touch
``data/processed/`` -- the pipeline rewrites that directory wholesale on every
run and would silently delete them.

Why a JSONL file rather than the database
-----------------------------------------
``backend/db/`` is an empty package in this build; the only SQLite in use is
the pseudonym vault, which nothing in the analytics path may open. A one-line-
per-submission file is the smallest thing that is honest about what it is: it
appends, it never rewrites history, and a submission is durable the moment the
line is flushed. When a real database layer lands, this module is the only
thing that changes -- the route above it already speaks in terms of "record
this submission" rather than in terms of files.

What this deliberately does NOT do
----------------------------------
Answers are never fed back into scoring. The nine behavioral signals are
computed from HR records alone, and a person's self-assessment does not move
their risk score in either direction. That is what makes it safe to say on the
screen that answering is optional and that skipping it does not flag anybody:
if answers changed the score, "optional" would not be true.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from backend.config import settings

QUESTIONS_PATH = Path(__file__).resolve().parent / "wellness_questions.json"

KIND_SCALE = "scale"
KIND_FREE_TEXT = "free_text"

# Bound on a single free-text answer. ASSUMPTION: a check-in note is a few
# sentences; anything longer is more likely a paste accident than an answer, and
# an unbounded field written straight to disk is not something to leave open.
MAX_FREE_TEXT_CHARS = 2000

# Bound on how many answers one submission may carry. The question bank serves
# two general plus at most three tailored questions, so anything beyond a small
# multiple of that is malformed. ASSUMPTION.
MAX_ANSWERS_PER_SUBMISSION = 20


class InvalidSubmission(ValueError):
    """Raised when a submitted check-in does not have a usable shape."""


@lru_cache(maxsize=1)
def question_kinds(path: Path = QUESTIONS_PATH) -> Mapping[str, str]:
    """Return every question id in the bank with the kind of answer it takes.

    Args:
        path: The question bank file.

    Returns:
        Mapping of question id to ``"scale"`` or ``"free_text"``. Loaded once:
        the bank is static between deployments.

    Why the store reads the bank:
        An answer is only meaningful against the question it answers. Storing
        an id the bank does not contain, or a slider value for a question that
        asked for words, is storing something nobody can later read back as
        an answer to anything.
    """
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    kinds: Dict[str, str] = {}
    for question in bank.get("general", []):
        kinds[str(question["id"])] = KIND_FREE_TEXT if question.get("free_text") else KIND_SCALE
    for questions in bank.get("by_signal", {}).values():
        for question in questions:
            kinds[str(question["id"])] = KIND_FREE_TEXT if question.get("free_text") else KIND_SCALE
    return kinds


def _coerce_answer(raw: Any, kinds: Mapping[str, str]) -> Dict[str, Any]:
    """Validate and normalise one answer.

    Args:
        raw: One entry from the submitted ``answers`` list.
        kinds: Question id -> answer kind, from :func:`question_kinds`.

    Returns:
        The normalised answer: ``question_id``, and exactly one of ``value``
        (0-4 scale) or ``text`` (free text), matching the question's kind.

    Raises:
        InvalidSubmission: If the entry is not a dict, names a question the
            bank does not contain, or carries an answer of the wrong kind or
            none at all.

    Note:
        Validation rejects rather than repairs, matching
        ``backend/ingestion/validators.py``. A silently corrected answer is a
        record of something the person did not say.
    """
    if not isinstance(raw, dict):
        raise InvalidSubmission("each answer must be an object")

    question_id = str(raw.get("question_id", "")).strip()
    if not question_id:
        raise InvalidSubmission("each answer must carry a question_id")
    kind = kinds.get(question_id)
    if kind is None:
        raise InvalidSubmission(f"'{question_id}' is not a question in the bank")

    answer: Dict[str, Any] = {"question_id": question_id}

    if raw.get("text") is not None:
        text = str(raw["text"]).strip()
        if len(text) > MAX_FREE_TEXT_CHARS:
            raise InvalidSubmission(
                f"free-text answer exceeds {MAX_FREE_TEXT_CHARS} characters"
            )
        if text:
            answer["text"] = text

    if raw.get("value") is not None:
        try:
            value = int(raw["value"])
        except (TypeError, ValueError) as exc:
            raise InvalidSubmission(
                f"answer to {question_id} is not a whole number"
            ) from exc
        if not 0 <= value <= 4:
            raise InvalidSubmission(
                f"answer to {question_id} must be on the 0-4 scale"
            )
        answer["value"] = value

    if "value" not in answer and "text" not in answer:
        raise InvalidSubmission(f"answer to {question_id} is empty")
    if kind == KIND_SCALE and "value" not in answer:
        raise InvalidSubmission(f"{question_id} takes a 0-4 answer, not free text")
    if kind == KIND_FREE_TEXT and "text" not in answer:
        raise InvalidSubmission(f"{question_id} takes free text, not a 0-4 answer")
    # Keep exactly the field the question asked for.
    field = "value" if kind == KIND_SCALE else "text"
    return {"question_id": question_id, field: answer[field]}


def record_submission(
    pseudonym_id: str,
    answers: Sequence[Any],
    path: Path | None = None,
    kinds: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Append one check-in submission.

    Args:
        pseudonym_id: Whose submission this is.
        answers: The raw submitted answers.
        path: Destination file. Defaults to
            ``settings.CHECKIN_RESPONSES_PATH``.
        kinds: Question id -> kind. Defaults to the shipped question bank.

    Returns:
        The stored record, including its ``submitted_at`` timestamp.

    Raises:
        InvalidSubmission: If the submission is empty, oversized, or any
            individual answer is malformed.
    """
    if not isinstance(answers, (list, tuple)):
        raise InvalidSubmission("answers must be a list")
    if not answers:
        raise InvalidSubmission("submission contained no answers")
    if len(answers) > MAX_ANSWERS_PER_SUBMISSION:
        raise InvalidSubmission(
            f"submission carried more than {MAX_ANSWERS_PER_SUBMISSION} answers"
        )

    known = kinds if kinds is not None else question_kinds()
    coerced = [_coerce_answer(entry, known) for entry in answers]
    seen = set()
    for entry in coerced:
        if entry["question_id"] in seen:
            raise InvalidSubmission(f"{entry['question_id']} was answered twice in one submission")
        seen.add(entry["question_id"])

    record = {
        "pseudonym_id": str(pseudonym_id),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "answers": coerced,
    }

    destination = Path(path or settings.CHECKIN_RESPONSES_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def submissions_for(pseudonym_id: str, path: Path | None = None) -> List[Dict[str, Any]]:
    """Return one person's own past submissions, newest first.

    Args:
        pseudonym_id: Whose submissions to read.
        path: Source file. Defaults to ``settings.CHECKIN_RESPONSES_PATH``.

    Returns:
        That person's submissions. Empty when they have never submitted, or
        when the file does not exist yet.

    Note:
        A malformed line is skipped rather than raising. This file is appended
        to by a running server; a truncated final line after an abrupt stop
        should cost that one submission, not the whole history.
    """
    source = Path(path or settings.CHECKIN_RESPONSES_PATH)
    if not source.exists():
        return []

    found: List[Dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("pseudonym_id") == pseudonym_id:
            found.append(record)

    found.sort(key=lambda r: str(r.get("submitted_at", "")), reverse=True)
    return found
