"""Resolve a pseudonym back to a person, through the audited path, from a terminal.

One job: exercise the one supported route back from ``pseudonym_id`` to
``personnel_id``, so that the audit trail behind it contains real rows.

Why this is a command and not an API route
-------------------------------------------
The whole system is built so that no name, service number or raw
``personnel_id`` reaches the analytics layer, the models, the stored scores or
any API response. Adding an HTTP route that returns a real identity would put
identity and welfare data on the same wire, one authorisation bug away from each
other -- which is the exact failure the pseudonymisation exists to prevent, and
this codebase has already found two authorisation bugs of that shape.

Re-identification is also not a screen action. It is what happens when a welfare
officer has decided to actually go and speak to someone, which is a deliberate,
attributable, occasional act. A command that must be run with a stated purpose,
by someone with shell access to the vault host, matches that shape. A button
does not.

What the audit trail was before this
------------------------------------
The machinery in ``backend/preprocessing/pseudonymize.py`` was complete --
``resolve()`` checked the role, demanded a purpose, and wrote to
``reidentification_audit`` whether it granted or refused. But nothing in the
running system ever called it, so the table had zero rows, and the project was
describing an audited re-identification path it had never once exercised. A
control nobody has run is a claim, not a control.

Usage
-----
    python scripts/reidentify.py --pseudonym PSNa1b2c3d4e5f60718 \\
        --officer WO-DEMO-01 \\
        --purpose "Welfare visit scheduled following High-band escalation"

    python scripts/reidentify.py --audit          # show the trail, newest first

Every invocation is recorded, including the ones that are refused -- a run of
refusals against one pseudonym is precisely what an audit trail exists to
reveal, so it must not be the case that only successes leave a trace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402
from backend.preprocessing import pseudonymize  # noqa: E402


def _print_audit(vault: pseudonymize.PseudonymVault, limit: int) -> None:
    """Print the re-identification trail.

    Args:
        vault: The opened vault.
        limit: How many rows to show, newest first.
    """
    rows = vault.audit_trail(limit=limit)
    if not rows:
        print("No re-identification has ever been attempted against this vault.")
        return
    print(f"{len(rows)} most recent re-identification attempt(s), newest first:\n")
    for row in rows:
        outcome = "GRANTED" if row["granted"] else "REFUSED"
        print(f"  [{outcome}] {row['requested_at']}")
        print(f"    pseudonym : {row['pseudonym_id']}")
        print(f"    requester : {row['requester_id']} ({row['requester_role']})")
        print(f"    purpose   : {row['purpose']}")
        print()


def main(argv: list[str] | None = None) -> int:
    """Resolve one pseudonym, or print the audit trail.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code. 0 on success, 2 when re-identification was refused
        -- refusal is a normal outcome of an authorisation check, not a crash,
        and it is audited either way.
    """
    parser = argparse.ArgumentParser(
        description="Audited re-identification of one pseudonym.",
        epilog=(
            "Only a welfare officer may re-identify, and only with a stated "
            "purpose. Both the grant and the refusal are written to the "
            "reidentification_audit table inside the identity vault."
        ),
    )
    parser.add_argument("--pseudonym", help="The pseudonym_id to resolve.")
    parser.add_argument(
        "--officer",
        default="WO-DEMO-01",
        help="Identifier of the requesting officer, recorded in the audit row.",
    )
    parser.add_argument(
        "--role",
        default=settings.ROLE_WELFARE_OFFICER,
        help=(
            "Role claimed by the requester. Anything outside "
            f"{list(pseudonymize.REIDENTIFICATION_ROLES)} is refused and the "
            "refusal is audited -- pass one deliberately to see that happen."
        ),
    )
    parser.add_argument(
        "--purpose",
        default="",
        help="Why. Recorded verbatim. A blank purpose is refused.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print the re-identification audit trail and exit.",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Rows to show with --audit."
    )
    args = parser.parse_args(argv)

    vault = pseudonymize.PseudonymVault()

    if args.audit:
        if args.pseudonym:
            summary = vault.audit_summary(args.pseudonym)
            print(f"Re-identification attempts against {args.pseudonym}:")
            for key, value in summary.items():
                print(f"  {key:<18} {value}")
            print()
        _print_audit(vault, args.limit)
        return 0

    if not args.pseudonym:
        parser.error("--pseudonym is required unless --audit is given")

    print(f"Vault holds {vault.count()} registered people.")
    try:
        personnel_id = vault.resolve(
            pseudonym_id=args.pseudonym,
            requester_id=args.officer,
            requester_role=args.role,
            purpose=args.purpose,
        )
    except pseudonymize.ReidentificationDenied as exc:
        print(f"REFUSED: {exc}")
        print("The refusal has been written to the audit trail.")
        return 2

    print(f"GRANTED: {args.pseudonym} -> {personnel_id}")
    print(
        "This resolution has been written to the audit trail. The identity is "
        "printed here and nowhere else -- it is not returned by any API route, "
        "not written to any analytics store, and not cached."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
