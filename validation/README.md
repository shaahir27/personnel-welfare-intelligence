# `validation/`

External validation. Deliberately **outside `backend/`** — none of this is part
of the served system, and nothing in `backend/` imports it.

| File | Job |
| --- | --- |
| `build_rating_profiles.py` | Render N cases as plain descriptions with no score attached, for people to rate. |
| `run_human_rating_validation.py` | Measure rater agreement, then model-vs-consensus correlation. |

## The problem this addresses

Every metric this project reports is measured against the synthetic label
produced by `latent_welfare_risk()` in the data generator. R² measures how well
the model reproduces a formula the project wrote. It cannot measure predictive
validity, because nothing in the corpus is a welfare outcome. That is stated
throughout the documentation and is the honest position — but "we admit it" is a
weaker place to stand than a number that did not come from our own generator.

### Why there is no dataset route

There is no public corpus of CAPF duty hours, leave and deployments with a
welfare outcome attached. There cannot be — that data is not public and should
not be.

**Do not reach for the IBM HR Analytics Employee Attrition dataset.** It comes
up first in every search, it has `OverTime`, `DistanceFromHome`,
`BusinessTravel`, `YearsInCurrentRole` and an `Attrition` label, and it looks
perfect. It is **itself synthetic** — IBM data scientists generated it.
Validating our generated data against theirs proves nothing, and a reviewer who
knows the dataset will say so. Claiming external validation while providing none
is considerably worse than claiming nothing.

## What is available instead: a judgement from outside the generator

Give people the circumstances of a case, with no score shown, and ask how
concerned they would be. The ratings that come back are not derived from
`latent_welfare_risk()` — they come from people reading a situation and forming
a view. If the model's ranking correlates with their consensus ranking, that is
evidence the score tracks something a person recognises as welfare concern,
which is the claim the system actually needs to make.

## How to run it

```bash
# 1. Build the sheet (needs a completed pipeline run)
python validation/build_rating_profiles.py --count 50

# 2. Collect ratings. rating_profiles.md is the text to paste into a form.
#    DELETE the model_score and pseudonym_id columns before sharing the CSV.

# 3. Analyse whatever comes back
python validation/run_human_rating_validation.py --ratings collected.csv
```

`collected.csv` needs three columns: `profile_id`, `rater_id`, `rating` (0–10).
One row per rater per profile. A skipped profile is simply an absent row — it is
dropped pairwise rather than having a value invented for it.

### Collection notes

- **5–10 raters, independently.** Do not let them discuss the profiles first;
  the inter-rater number is only meaningful if the judgements are independent.
- **Include people with a service connection where possible** — NCC
  instructors, anyone with a family or service link, faculty. Ordinary students
  are still usable, but say so in the write-up.
- **Do not show anyone the model's score.** The `model_score` column exists so
  the analysis can join back; a rater who has seen it is no longer an
  independent judgement, and there is no way to un-see it.

## Reading the two numbers

**Inter-rater agreement comes first, and it bounds the second number.** If the
humans do not agree with each other, that is itself a finding, and it caps what
any model could achieve — you cannot correlate with a consensus that does not
exist. Quoting a model-human correlation without it is quoting a number whose
ceiling is unknown.

**Model-human correlation** is Spearman, not Pearson: the claim is about
*ranking* — whether the system puts the same people near the top a person
would — and a 0–10 concern rating is ordinal.

### State the limits when writing it up

All three, unprompted, because a reader will think of them anyway:

- the raters are not welfare officers,
- they read a summary rather than meeting a person,
- correlation with human intuition is not correlation with outcomes.

A ρ around 0.7 against a consensus whose own inter-rater agreement is around
0.7 is a good result honestly reported. A ρ of 0.9 quoted without the
inter-rater figure is a number nobody should believe.

## Implementation notes

Spearman is computed directly rather than through scipy, so these scripts run in
an environment with no package-registry access — the same constraint the rest of
the build works under. It is Pearson on ranks, with ties sharing the mean of the
ranks they span, which matters here because a 0–10 rating has a great many ties.

Profiles are sampled **stratified across the score range**, not at random. A
random 50 from a population that bunches in the Moderate band gives raters fifty
near-identical descriptions, and a correlation measured on a sample with no
spread is not measuring much. That is a choice about the instrument and it is
stated here rather than left to be discovered in the code.

## The other half of external validation

The voice side is separate and already exists: `voice-lab/` collects real
recordings from real people with self-assigned labels and runs the project's
exact acoustic pipeline over them, unretrained. That validates component E. This
directory is the HR-side counterpart, and neither substitutes for the other.
