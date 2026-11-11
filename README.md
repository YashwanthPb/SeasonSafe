# SeasonSafe experiment kit

Start with PROJECT_BRIEF.md. Open SeasonSafe_Colab.ipynb in https://colab.research.google.com/.
Run the notebook cells in order. It is self-contained: experiment.py is optional for local execution.
Automatic download uses UCI. If it fails, download the ZIP from
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
and upload the extracted online_shoppers_intention.csv to Colab, then set CSV_PATH.
Do not upload private company data into this public-data workflow.

The final cell downloads computed tables, charts, predictions and a decision manifest as a ZIP.
Read results/INTERPRET_RESULTS.md before interpreting outputs. Open validation_models.csv for
model selection evidence and policy_comparison.csv for the main business question.
Local usage: python experiment.py --csv results/source_dataset.csv --output reproduced_results
Dependencies: requirements.txt. The manifest records actual runtime versions and dataset SHA-256.

Status: official data downloaded through the browser after a direct-download failure. The primary
run and duplicate-removal sensitivity completed. Read FINDINGS.md and SENSITIVITY.md. Original and
cleaned data are bundled under results/ with attribution. The notebook is self-contained and can
rerun the analysis; it includes a clearly labeled reference snapshot, not fabricated execution outputs.

Files: PROJECT_BRIEF.md; SeasonSafe_Colab.ipynb; experiment.py; FEATURES.csv;
SOURCES.md; AI_USE_LOG.md; TEAM_WORKSHEET.md; requirements.txt; VALIDATION.md.
Also included: FINDINGS.md, SENSITIVITY.md, requirements_verified.txt, results/,
and sensitivity_deduplicated/. The exact December capacity finding changes under deduplication;
read this sensitivity before presenting conclusions.
When rechecking saved probabilities against a cutoff, load test_predictions.csv using
pandas.read_csv(path, float_precision='round_trip') to preserve values at the exact threshold.
The self-contained notebook repeats the primary run. To repeat the sensitivity locally, call
run_experiment(raw.drop_duplicates(), 'sensitivity_deduplicated', data_hash=...) separately;
calculate the hash from the deduplicated CSV bytes and label the run as transformed data.
The project report is available in [PDF](SeasonSafe_Project_Report.pdf) and [Word](SeasonSafe_Project_Report.docx).


