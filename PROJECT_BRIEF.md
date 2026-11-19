# SeasonSafe
## Can an e-commerce purchase-scoring policy survive a change in shopping patterns?

Status: research design, runnable AI-generated scaffold, and verified first run, 11 November 2026.
Official-data results are in FINDINGS.md; protocol was written before the held-out run.

### Business problem
A retailer can review a limited share of customer opportunities. A score cutoff set in one month
may select too many or too few opportunities in a later month. Even if ranking remains useful,
predicted purchase probabilities may become unreliable. This project audits those failures.
The dataset does not establish which shoppers benefit from an intervention. We evaluate historical
buyer concentration and probability reliability, not causal uplift, realized profit, or saved sales.

### Primary question and contribution
Does a purchase-score cutoff chosen to select approximately 10% of October sessions preserve
selection capacity and buyer concentration in November and December? How does it compare with
an offline policy selecting the highest-scoring 10% within each evaluation month?

The proposed contribution combines a frozen threshold, capacity utilization, probability reliability,
and feature-availability constraints in one operational audit. This is a proposed course-project
contribution, not a claim of a new ML algorithm or a worldwide first. Earlier work already studies
purchase prediction and arrival-context features. Public projects also examine PageValues and
time-based testing; those ideas alone are not our originality claim.

### Three supporting questions
1. How much performance changes when using context, completed-session behavior, and PageValues?
2. Does predicted buyer volume match actual buyer volume within the selected group?
3. Do new and returning visitors have different selection rates or precision?

### Data and decision timing
Use UCI Online Shoppers Purchasing Intention, dataset 468, CC BY 4.0, attributed to Sakar and Kastro.
UCI describes 12,330 sessions and 1,908 positive purchase outcomes, each session from a different user
within a one-year period. Verify these counts on download. Revenue is a boolean outcome, NOT money.

Primary model: arrival-context proxy features only (browser, OS, region, traffic source code, visitor
type, calendar month, weekend and SpecialDay). Their availability at arrival is an assumption to
validate in a future production system. Coded categories are nominal; no meaning is invented for codes.
SpecialDay is supplied; it is not evidence of Indian festivals or a particular shopping campaign.

Retrospective comparisons: completed-session page counts, durations, BounceRates and ExitRates.
The public CSV has no per-click histories or timestamps for reconstructing the first 30/60 seconds.
These comparisons cannot demonstrate real-time deployment. PageValues is transaction-related;
its exact point-in-time construction is unclear. Exclusion is conservative, not proof of leakage.

### Locked experiment
- Train: available months through September. No random split used as primary evidence.
- Validation: October. Choose among logistic regression, random forest and histogram gradient boosting
  using average precision, then Brier score as a tie-break. Choose ONLY within arrival-context family.
- Test: November and December, separately and pooled for diagnostic model metrics.
- Primary capacity: 10%; 5% and 20% are pre-specified sensitivity analyses.
- Fixed policy: freeze October's cutoff before seeing test outcomes. Ties can exceed capacity.
- Batch policy: rank within each month and select floor(capacity * sessions). Random seeded tie-break.
- No test-driven tuning, model reselection, or choosing the month that looks best.
- Exact same input records stay in the same time partition; identical rows are retained because the
  dataset supplies no session identifiers proving they are duplicate records. Document a sensitivity
  analysis with exact duplicates removed. Do not silently change the primary analysis.
- Month contains limited temporal detail. This is a one-year chronological stress test, not proof
  of performance in another year, country, merchant, or device population.

### Metrics and interpretation
Primary business outputs: selection rate, over-budget count, unused capacity, observed buyer
concentration (precision), buyers captured (recall), and actual minus predicted buyers selected.
Predictive diagnostics: average precision (not trapezoidal PR-AUC), ROC-AUC, Brier and log loss.
Use the constant training-prevalence model as a probability baseline; random targeting's expected
precision equals the evaluation month's prevalence. Inspect both months rather than aggregate only.

A top-budget rule satisfies capacity by construction; present this as a design property. The empirical
question is the change in buyer concentration and reliability. A fixed threshold selecting more rows
may capture more buyers simply because it uses more capacity. Never compare recall alone.
Approximate paired row-bootstrap intervals quantify precision differences conditional on fitted
models; they do not account for model selection, temporal correlation, or multiple comparisons.
One-year seasonal observations do not justify formal long-run temporal guarantees.

### What makes the work yours
The team should document one short discussion with an actual e-commerce operator, account manager,
or small seller about capacity and acceptable errors. Do not invent interviews. The 10% example is
a scenario assumption until someone supplies a real constraint. Record how feedback changes the
business interpretation before running final evaluation. Conduct the literature comparison, defend
feature timing, investigate a surprising result, and write the interpretation independently.

### Expected figures (generated after a real-data run)
1. Purchase rate by available month.
2. October vs November vs December selection rate at 10% capacity.
3. Reliability diagrams by month.
Add a compact table comparing context and retrospective feature sets, plus segment findings with counts.

### Academic requirements verified in the latest template
IEEE conference format; Section II says six pages including references, but the bibliography section
says four pages. Confirm the limit with the instructor. Second review: 20 November 2026; final:
4 December 2026; source states 11:59 PM AoE. Work toward the calendar date, not a last-hour conversion.
Originality is 6/30; relevance, significance, achievement, writing, reproducibility and technical quality
are 4 each. GitHub code/data links and reproducibility notes are required. The template says AI text/code
must be below 20% and declared. This package is substantially AI-generated and is NOT a compliant final
submission by itself; editing it does not automatically establish compliance. Confirm allowed assistance.

### Work plan
11-13 Nov: approve framing, read sources, data audit, obtain actual stakeholder input if possible.
14-16 Nov: team implementation/verification, training and October validation, lock decisions.
17-19 Nov: run held-out evaluation, diagnose errors, prepare second-review evidence.
20 Nov: second review. Record instructor feedback and changes separately from original results.
21-29 Nov: documented sensitivity checks, interpretation, reproducibility rerun by another teammate.
30 Nov-3 Dec: team-written IEEE report, references, AI declaration, final checks.
4 Dec: final submission.

### Viva questions you must be able to answer
Why not accuracy? What is the difference between average precision and precision at 10%?
Why exclude PageValues from the main policy without declaring it conclusively leaky?
Can a likely buyer be assumed to benefit from a discount? No: propensity is not treatment effect.
Why is whole-month top-10% selection an offline benchmark rather than a live queue?
Does temporal ordering eliminate all leakage? No: feature timing and preprocessing still matter.
If a simple model wins, is the project unsuccessful? No: report the decision evidence honestly.
Can this be called Amazon data or a validated Amazon tool? No.
