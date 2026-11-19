# SeasonSafe

**Can an e-commerce purchase-scoring policy survive a change in shopping patterns?**
This project audits whether a purchase-score cutoff that selects 10% of October sessions still holds a
10% workload in November and December. It uses the UCI Online Shoppers Purchasing Intention dataset
(12,330 sessions, CC BY 4.0). Course project for Advanced Machine Learning for Business Transformation,
Goa Institute of Management. Team 46 (Section C): Garima Rohilla, Harleen Kaur, Yashwanth M.

Report: [PDF](SeasonSafe_Project_Report.pdf) · [Word](SeasonSafe_Project_Report.docx) ·
[IEEE LaTeX source](report/seasonsafe.tex)

## Read in this order
1. `PROJECT_BRIEF.md`: problem, locked protocol, metrics, viva questions.
2. `FINDINGS.md`: locked primary results (pre-specified).
3. `SENSITIVITY.md`: pre-specified duplicate-removal sensitivity.
4. `EXTENDED_FINDINGS.md`: post-hoc analyses (ablation, rule baselines, cutoff uncertainty,
   recalibration, drift, importance, rolling policy, tuning). These **qualify** the primary result.
5. `results/INTERPRET_RESULTS.md` before quoting any number.

## Repository layout
| Path | Contents |
|---|---|
| `experiment.py` | Locked primary pipeline; also runs the duplicate-removal sensitivity (`--dedup`) |
| `extended_analysis.py` | Post-hoc analyses → `results_extended/` |
| `SeasonSafe_Colab.ipynb` | Self-contained Colab version of the primary run |
| `tests/` | Software checks (synthetic data) and a recomputation of the published headline numbers |
| `results/` | Primary outputs, including `source_dataset.csv` (as loaded) and `cleaned_dataset.csv` |
| `sensitivity_deduplicated/` | Duplicate-removal sensitivity outputs |
| `results_extended/` | Post-hoc outputs; every CSV has an `analysis_status` column |
| `report/` | IEEE conference LaTeX source, compiled PDF, and `make_figures.py` (builds the six multi-panel figures) |
| `FEATURES.csv`, `SOURCES.md`, `VALIDATION.md`, `TEAM_WORKSHEET.md`, `AI_USE_LOG.md` | Supporting records |
| `SeasonSafe_Complete.zip` | Original 11 Nov delivery bundle (superseded by the unpacked files above) |

## Reproduce
Python 3.13 or 3.14. Exact versions used for the published run: `requirements_verified.txt`.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements_verified.txt pytest
python experiment.py --csv results/source_dataset.csv --output reproduced_results
python experiment.py --csv results/source_dataset.csv --dedup --output reproduced_dedup
python extended_analysis.py --csv results/source_dataset.csv --output reproduced_extended
python report/make_figures.py   # rebuilds report/figures/ from results/ and results_extended/
python -m pytest -q
```

Expected: the selected model is `hist_boost` in both runs; the reproduced CSVs match `results/` and
`sensitivity_deduplicated/` to floating-point precision (verified 19 Nov 2026, Python 3.13). The primary
run takes about 10 s and the extended run about 1 min on a laptop-class CPU.

Data: if `--csv` is omitted, `experiment.py` downloads the official ZIP from UCI. If that fails, download it from
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset and pass the
extracted `online_shoppers_intention.csv`. `results/source_dataset.csv` is the official data as loaded
and re-saved by pandas, so its file hash differs from the manifest's `data_sha256`, which hashes the
original downloaded bytes.

Colab: open `SeasonSafe_Colab.ipynb` and run the cells in order. It installs the pinned versions and
repeats the primary run only. When rechecking saved probabilities against a cutoff, load
`test_predictions.csv` with `pandas.read_csv(path, float_precision='round_trip')`.

## Status and honesty notes
- Primary and sensitivity runs: complete, independently recomputed (`VALIDATION.md`, `tests/`).
- Extended analyses: complete, explicitly post-hoc. They show that the size and direction of the capacity
  error depend on an unseen-Month encoding artefact and on cutoff estimation noise, and that the model
  is close to a "new visitors first" rule. Present both layers.
- AI assistance is substantial and declared in `AI_USE_LOG.md`. The course caps AI-generated content
  below 20%; this repository is not certified as compliant.
- Do not upload private company data into this public-data workflow.
