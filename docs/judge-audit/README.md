# Judge audit: which judge follows the rubric?

Argon (gemini-4-argon) and Haiku (claude-haiku-5-5) judged the **same** Argon answers with the **same** judge
prompts. They agree on 91–93% of items. On the rest, Argon's judge says CORRECT and Haiku's says WRONG in
299 cases; the reverse happens 5 times. That one-sided disagreement moves accuracy by 7–9 points, so we need to
know which judge is right.

`audit_sheet.csv` has 121 items in random order: 100 disagreements, the 5 reverse cases, and 16 items where
both judges agreed (controls). Which judge said what is hidden; the key is kept apart, in
`runs-argon/crossjudge/audit_key.csv`, so don't open it until you're done.

## How to label

For each row, fill `your_label` with **CORRECT** or **WRONG**. Judge by **the rubric for that row's benchmark
and question type** (below), not by your own standard: the question is which judge applies the rubric as
written. `notes` is optional. Write in a word or two why for any close call (e.g. "hedged 3 or 4", "right
date, wrong year").

Use only the gold answer and the rubric. Don't look up the conversation; the judges didn't see it either.

## Rubrics (verbatim from the judge prompts)

**LoCoMo** (Mem0 `ACCURACY_PROMPT`, every category):
> The generated answer might be much longer, but you should be generous with your grading - as long as it
> touches on the same topic as the gold answer, it should be counted as CORRECT.
> For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer
> might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should
> be generous with your grading - as long as it refers to the same date or time period as the gold answer, it
> should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT
> if it's the same date.

**LongMemEval** (upstream `evaluate_qa.py`, by `question_type`):
- **single-session-user, single-session-assistant, multi-session:** CORRECT if the response contains the
  correct answer, is equivalent to it, or contains all the intermediate steps to get it. If the response only
  contains a subset of the information required by the answer, WRONG.
- **temporal-reasoning:** as above, and do not penalize off-by-one errors in a number of days, weeks, months,
  and so on (e.g. 19 days when the answer is 18 is still CORRECT).
- **knowledge-update:** CORRECT if the response contains the correct answer. If it gives some earlier
  information along with an updated answer, it's CORRECT as long as the updated answer is the required one.
- **single-session-preference:** the gold column is a rubric for the desired personalized response. CORRECT if
  the response recalls and uses the user's personal information correctly; it doesn't need every rubric point.
- **abstention:** the gold column explains why the question is unanswerable. CORRECT if the response
  identifies the question as unanswerable (it may say the information is incomplete, or that other
  information is given but not the asked one).

## When done

Save the sheet in place and tell Claude. Scoring compares each judge with your labels on the sampled
disagreements, then corrects each judge's accuracy for the error rate it showed.

Rebuild the same sample: `uv run python scripts/build_judge_audit.py` (seed 0).
