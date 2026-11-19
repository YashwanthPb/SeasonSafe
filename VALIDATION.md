# Software validation
Synthetic smoke test passed on 11 November 2026: all 9 model/feature pipelines trained;
27 diagnostic model-period rows produced; chronological partitions stayed separate;
unseen test category and missing training value handled; top-budget counts and score ties
checked; metric arithmetic checked; CSV/JSON exports and 3 plots generated.
Notebook format and Python syntax checked. This does not test the UCI download or establish
actual UCI performance. Real-data end-to-end run remains pending because DNS/download failed.
Synthetic fixtures/results are excluded from this deliverable. No real-data metrics are claimed.

## Official-data validation completed subsequently
Official-data primary experiment and pre-specified duplicate sensitivity both completed.
Verified 12,330 rows, 1,908 purchases, 4,725 unique held-out row IDs, and month boundaries.
Recomputed frozen-cutoff selection counts, buyer counts, precision, budget, and predicted
buyer totals independently from exported probabilities for November and December.
Inspected charts and checked notebook syntax/format. Synthetic results remain excluded.

## Reproduction and automated checks (19 November 2026)
Environment: Python 3.13.18, numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, matplotlib 3.11.1.
`experiment.py` re-run on results/source_dataset.csv reproduced every primary CSV (max absolute
difference 3.6e-12); `--dedup` reproduced sensitivity_deduplicated/ (max 3.4e-12).
`python -m pytest -q`: 12 passed. The tests cover top-k selection with ties, capacity arithmetic,
chronological partitions, schema/boolean/month validation, the unseen-Month encoding property, a quick
synthetic end-to-end run, the --dedup CLI, a quick extended run, notebook-script code identity, pinned
notebook versions, and recomputation of the Nov/Dec headline counts from results/test_predictions.csv.
