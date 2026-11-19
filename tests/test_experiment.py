"""Software checks for SeasonSafe. Synthetic data tests software only; they are NOT empirical evidence.

Run: python -m pytest -q
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment as ex  # noqa: E402

MONTH_ROWS = {'Feb': 60, 'Mar': 200, 'May': 300, 'June': 60, 'Jul': 80, 'Aug': 80, 'Sep': 80,
              'Oct': 120, 'Nov': 250, 'Dec': 160}


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for month, n in MONTH_ROWS.items():
        vt = rng.choice(['New_Visitor', 'Returning_Visitor', 'Other'], n, p=[.25, .7, .05])
        logit = -2 + 1.2 * (vt == 'New_Visitor') + rng.normal(0, 1, n)
        y = rng.random(n) < 1 / (1 + np.exp(-logit))
        for i in range(n):
            rows.append({'Administrative': int(rng.integers(0, 5)), 'Administrative_Duration': float(rng.gamma(2, 20)),
                         'Informational': int(rng.integers(0, 3)), 'Informational_Duration': float(rng.gamma(1, 10)),
                         'ProductRelated': int(rng.integers(1, 40)), 'ProductRelated_Duration': float(rng.gamma(2, 300)),
                         'BounceRates': float(rng.uniform(0, .2)), 'ExitRates': float(rng.uniform(0, .2)),
                         'PageValues': float(rng.gamma(1, 5) * y[i]),
                         'SpecialDay': float(rng.choice([0, .4, .8])) if month in ('Feb', 'May') else 0.0,
                         'Month': month, 'OperatingSystems': int(rng.integers(1, 4)), 'Browser': int(rng.integers(1, 4)),
                         'Region': int(rng.integers(1, 5)), 'TrafficType': int(rng.integers(1, 6)),
                         'VisitorType': vt[i], 'Weekend': bool(rng.random() < .25), 'Revenue': bool(y[i])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ unit checks
def test_top_mask_selects_exact_floor_budget_with_ties():
    scores = np.array([.9, .5, .5, .5, .5, .1, .1, .1, .1, .1, .1])
    m = ex.top_mask(scores, .2)
    assert m.sum() == 2 and m[0]
    assert m[1:5].sum() == 1  # the tie at .5 is broken, not over-selected
    assert np.array_equal(m, ex.top_mask(scores, .2))  # seeded: deterministic


def test_top_mask_minimum_one():
    assert ex.top_mask(np.array([.3, .2]), .01).sum() == 1


def test_policy_metrics_capacity_arithmetic():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.linspace(.9, .0, 10)
    over = ex.policy_metrics(y, p, p >= .5, .2)  # selects 5 against budget 2
    assert (over['budget'], over['selected'], over['over_budget'], over['unused_capacity']) == (2, 5, 3, 0)
    assert over['buyers_selected'] == 2 and over['precision'] == pytest.approx(.4)
    under = ex.policy_metrics(y, p, p >= .95, .2)
    assert under['selected'] == 0 and under['unused_capacity'] == 2 and np.isnan(under['precision'])


def test_prepare_partitions_are_chronological_and_disjoint():
    df, groups, dup = ex.prepare(synthetic())
    assert groups['train']._month.max() <= 9
    assert set(groups['validation']._month) == {10}
    assert set(groups['test']._month) == {11, 12}
    ids = [set(g._row_id) for g in groups.values()]
    assert sum(map(len, ids)) == len(df) and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_prepare_rejects_bad_schema_and_booleans():
    raw = synthetic()
    with pytest.raises(ValueError, match='schema'):
        ex.prepare(raw.drop(columns='PageValues'))
    bad = raw.copy(); bad['Revenue'] = bad['Revenue'].astype(object); bad.loc[0, 'Revenue'] = 'maybe'
    with pytest.raises(ValueError, match='boolean'):
        ex.prepare(bad)
    bad = raw.copy(); bad.loc[0, 'Month'] = 'Smarch'
    with pytest.raises(ValueError, match='month'):
        ex.prepare(bad)


def test_month_feature_is_unseen_for_every_evaluation_row():
    """Documents a design property reported in EXTENDED_FINDINGS.md: the one-hot Month columns
    are all zero for every October-December row, so Month carries no information there."""
    _, g, _ = ex.prepare(synthetic())
    pipe = ex.make_model('logistic', ex.CONTEXT, quick=True).fit(g['train'][ex.CONTEXT], g['train'].Revenue)
    enc = pipe.named_steps['preprocess'].named_transformers_['categorical']
    names = enc.get_feature_names_out([c for c in ex.CONTEXT if c in ex.CAT])
    month_cols = [i for i, n in enumerate(names) if n.startswith('Month_')]
    for part in (g['validation'], g['test']):
        Xt = enc.transform(part[[c for c in ex.CONTEXT if c in ex.CAT]])
        assert np.all(Xt[:, month_cols] == 0)


# ------------------------------------------------------------------ end-to-end (synthetic)
def test_quick_run_writes_all_outputs(tmp_path):
    ex.run_experiment(synthetic(), tmp_path, 'synthetic', quick=True)
    for f in ['validation_models.csv', 'test_models.csv', 'test_predictions.csv', 'policy_comparison.csv',
              'locked_decisions.json', 'policy_precision_difference_intervals.csv', '01_monthly_purchase_rates.png',
              '02_capacity_drift.png', '03_reliability.png', 'INTERPRET_RESULTS.md']:
        assert (tmp_path / f).exists(), f
    manifest = json.loads((tmp_path / 'locked_decisions.json').read_text())
    assert manifest['quick_test_only'] is True
    pol = pd.read_csv(tmp_path / 'policy_comparison.csv')
    batch = pol[pol.policy == 'batch_top_budget']
    assert (batch.selected == batch.budget).all()  # capacity by construction


def test_cli_dedup_labels_transformed_data(tmp_path):
    raw = synthetic()
    raw = pd.concat([raw, raw.iloc[:7]], ignore_index=True)
    csv = tmp_path / 'in.csv'; raw.to_csv(csv, index=False)
    out = tmp_path / 'dedup'
    # Quick mode is not exposed on the CLI; run the full models on the small synthetic table.
    r = subprocess.run([sys.executable, str(ROOT / 'experiment.py'), '--csv', str(csv), '--dedup', '--output', str(out)],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr
    assert 'removed 7 exact duplicate rows' in r.stdout
    manifest = json.loads((out / 'locked_decisions.json').read_text())
    assert manifest['data_sha256'].startswith('deduplicated:') and manifest['duplicate_rows_retained'] == 0


def test_extended_quick_run(tmp_path):
    import extended_analysis as ea
    ea.run(synthetic(), tmp_path, policy_csv=tmp_path / 'missing.csv', quick=True)
    for f in ['ablation_month_specialday.csv', 'rule_baselines.csv', 'cutoff_uncertainty.csv', 'recalibration.csv',
              'drift_psi.csv', 'permutation_importance.csv', 'rolling_cutoff_policy.csv', 'capacity_curve.csv',
              '04_capacity_curve.png', '06_rule_baselines.png']:
        assert (tmp_path / f).exists(), f
    assert (pd.read_csv(tmp_path / 'rule_baselines.csv').analysis_status.str.startswith('post-hoc')).all()


# ------------------------------------------------------------------ notebook / published-result consistency
def test_notebook_code_matches_script():
    nb = json.loads((ROOT / 'SeasonSafe_Colab.ipynb').read_text(encoding='utf-8'))
    cell = next(''.join(c['source']) for c in nb['cells']
                if c['cell_type'] == 'code' and 'def run_experiment' in ''.join(c['source']))
    script = (ROOT / 'experiment.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
    assert cell.strip() == script.strip(), 'Notebook copy of experiment.py has drifted; re-sync cell 4.'


def test_notebook_installs_pinned_versions():
    nb = json.loads((ROOT / 'SeasonSafe_Colab.ipynb').read_text(encoding='utf-8'))
    install = next(''.join(c['source']) for c in nb['cells'] if '%pip' in ''.join(c['source']))
    for line in (ROOT / 'requirements_verified.txt').read_text().split():
        assert line in install, line


PUBLISHED = Path(ROOT / 'results')


@pytest.mark.skipif(not (PUBLISHED / 'test_predictions.csv').exists(), reason='published results not present')
def test_published_headline_numbers_recompute_from_saved_predictions():
    pred = pd.read_csv(PUBLISHED / 'test_predictions.csv', float_precision='round_trip')
    cut = json.loads((PUBLISHED / 'locked_decisions.json').read_text())['thresholds']['0.1']
    prim = pred[(pred.family == 'arrival_context') & (pred.model == 'hist_boost')]
    expected = {'Nov': (299, 249, 85, 98), 'Dec': (172, 182, 43, 40)}
    for month, (budget, fixed_sel, fixed_buy, batch_buy) in expected.items():
        part = prim[prim.Month == month]
        y, p = part.y.to_numpy(), part.p.to_numpy()
        assert int(np.floor(len(p) * .1)) == budget
        assert int((p >= cut).sum()) == fixed_sel and int(y[p >= cut].sum()) == fixed_buy
        assert int(y[ex.top_mask(p, .1)].sum()) == batch_buy
