# Results interpretation checklist
These files contain computed results from the supplied input. Check locked_decisions.json:
quick_test_only=true means synthetic/software validation, NEVER empirical evidence.
Primary model was chosen using October only; do not reselect after seeing test results.
Compare November and December separately. Read precision together with number selected.
Batch top-budget meets capacity by construction: this is NOT an ML discovery.
Capacity is a fraction of observed monthly traffic, not a known fixed staffing level.
Fixed cutoffs can exceed the target in October too when probabilities tie; inspect counts.
The batch ranking is an offline benchmark. No within-month timestamps exist for a live queue.
Any session_no_pagevalues/session_with_pagevalues result is retrospective, not an early prediction.
PageValues has unclear historical availability; a performance drop does not itself prove leakage.
The dataset cannot identify incremental sales caused by offers, assistance or discounts.
Bootstrap intervals are conditional, approximate and not corrected for multiple comparisons.
Retained identical rows may affect uncertainty; add a documented deduplicated sensitivity run.
Do not infer small-segment conclusions. Do not call these Amazon or Indian-store results.
