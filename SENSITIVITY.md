# Exact-duplicate removal sensitivity

Separate pre-specified sensitivity; primary findings retain all records. Removing exact rows
does not prove those observations were erroneous duplicates. No primary model was replaced.

| Month | Policy | Budget | Selected | Precision |
|---|---|---:|---:|---:|
| Nov | fixed_october_cutoff | 298 | 206 | 33.01% |
| Nov | batch_top_budget | 298 | 298 | 31.88% |
| Dec | fixed_october_cutoff | 170 | 129 | 25.58% |
| Dec | batch_top_budget | 170 | 170 | 22.94% |

See sensitivity_deduplicated/locked_decisions.json for independent validation choice and counts.
No duplicate-robustness claim should extend beyond these displayed checks.