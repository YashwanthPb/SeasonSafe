# SeasonSafe: verified first-run findings

Run completed 11 November 2026 on the official UCI dataset. This is an exploratory course-project
analysis, not a production-ready targeting model or a team-authored final paper.

## Main finding
A fixed purchase-score cutoff did not maintain a 10% selection budget across months. Enforcing
a monthly top-10% rule controls capacity, but it did not produce a clear precision advantage.
Its capacity guarantee is mathematical; the empirical findings concern how much the frozen
cutoff drifted and what happened to precision and probability reliability.

## Data and locked choices
- 12,330 sessions; 1,908 purchases (15.47%).
- Train: 7,056 rows / 817 purchases from available February-September months.
- October validation: 549 / 115. November-December test: 4,725 / 976.
- Primary: histogram gradient boosting using arrival-context proxy features only.
- Chosen by October average precision 0.2502, narrowly above logistic regression 0.2488.
- Frozen 10% cutoff: 0.2849378071. Model selection did not use held-out outcomes.
- 125 identical rows retained in the primary analysis; removal tested separately.

## Primary 10% capacity experiment

| Period | Policy | Budget | Selected | Actual buyers selected | Precision |
|---|---|---:|---:|---:|---:|
| October (validation) | Fixed cutoff / top 10% | 54 | 54 | 11 | 20.37% |
| November | Fixed October cutoff | 299 | 249 | 85 | 34.14% |
| November | Monthly top 10% | 299 | 299 | 98 | 32.78% |
| December | Fixed October cutoff | 172 | 182 | 43 | 23.63% |
| December | Monthly top 10% | 172 | 172 | 40 | 23.26% |

November's fixed cutoff left 50 slots unused (16.72% of its 299-slot budget). The batch policy
included 13 additional observed buyers among 50 additional sessions selected. Those are historical
buyers identified, NOT additional purchases caused by an intervention.

December's fixed cutoff exceeded capacity by 10 sessions (5.81% of its 172-slot budget). The batch
policy respected capacity and selected three fewer observed buyers while reviewing ten fewer sessions.
Raw buyer counts cannot be interpreted as superiority at equal resource use.

The approximate 95% paired-bootstrap interval for top-budget minus fixed-cutoff precision was
[-4.11, +0.97] percentage points in November and [-1.99, +2.33] in December. Both include zero.
These conditional intervals are not equivalence tests or proof the policies are equally good.

## Probability reliability
For the 172 sessions selected by the December top-budget policy, the model predicted 58.65 buyers;
40 actually purchased. Overestimation was 18.65 buyers, or about 46.6% of the observed count.
In November, it predicted 100.47 buyers among 299 selected; 98 purchased. Calibration differed by
month. No model recalibration was performed after seeing these results.

## Prediction strength and the honesty point
The primary model's test ROC-AUC was 0.5968 in November and 0.6382 in December: limited ranking power.
At top 10%, purchase rates were 32.78% and 23.26%, against monthly base rates of 25.35% and 12.51%.
October top-decile precision (20.37%) was slightly BELOW that month's 20.95% base rate. The validation
winner's average-precision margin was only 0.00142; model-choice uncertainty is a real limitation.
Do not imply that selecting the best overall ranking metric also maximizes top-decile precision.

For the SAME boosting algorithm, pooled November-December ROC-AUC was:
- Arrival-context proxies: 0.5999.
- Completed-session behavior without PageValues: 0.6769.
- Completed-session behavior including PageValues: 0.8239.
The much stronger retrospective result cannot be advertised as early-session performance. It also
does not prove PageValues is leaky: its point-in-time construction is not established by the CSV.

## Recommended conclusion
Report a trade-off: capacity control is straightforward, while ranking and probability reliability
remain limited and unstable across months. A real retailer should validate available-at-decision
features and probability calibration before a controlled pilot. Incremental conversion or discount
benefit requires intervention data or a randomized experiment, which this dataset does not contain.

## Sensitivity changes the strength of the conclusion
After removing the 125 exact duplicate rows and repeating the pre-specified pipeline, the frozen
cutoff selected 206 of 298 slots in November and 129 of 170 in December. December changed from
over-capacity to under-capacity. Therefore, the precise December overload finding is NOT robust
to this data-handling choice. Capacity mismatch persists, but its direction and magnitude vary.
Report both analyses prominently. This instability is evidence against overconfident deployment,
not a reason to hide the sensitivity run. See SENSITIVITY.md and its complete outputs.

## Boundaries
One retailer, one year, monthly resolution, no clickstream timestamps or treatment labels. The
monthly ranking is an offline benchmark that sees all scores within a month, not an online queue.
The 10% budget is a scenario assumption. It is not a measured operational constraint.
Segments, additional budgets and deduplication are diagnostics, not grounds for replacing the
pre-specified primary result with a nicer-looking one.

Evidence: results/policy_comparison.csv; test_predictions.csv; test_models.csv;
validation_models.csv; locked_decisions.json; policy_precision_difference_intervals.csv.
