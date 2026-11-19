"""SeasonSafe extended analyses (post-hoc, added 19 November 2026).

Every analysis here was designed AFTER the locked primary run's held-out results were seen.
None of it replaces the primary result in results/. Outputs go to results_extended/ and each
table carries an `analysis_status` column so post-hoc evidence is never mistaken for locked evidence.

Usage: python extended_analysis.py --csv results/source_dataset.csv --output results_extended
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import experiment as ex

POST_HOC = 'post-hoc (designed after held-out results were seen)'
BUDGET = .10
MONTH_ORDER = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ---------------------------------------------------------------- shared helpers
def periods(val, test):
    return [('Oct', val), ('Nov', test[test._month == 11]), ('Dec', test[test._month == 12])]


def cutoff_from(scores, fraction=BUDGET):
    """Same rule as experiment.run_experiment: the k-th highest score, k=floor(n*fraction)."""
    scores = np.sort(np.asarray(scores))
    return float(scores[-max(1, int(np.floor(len(scores) * fraction)))])


def fit_primary(train, cols=ex.CONTEXT, name='hist_boost', seed=ex.SEED):
    old = ex.SEED
    ex.SEED = seed
    try:
        return ex.make_model(name, cols).fit(train[cols], train.Revenue)
    finally:
        ex.SEED = old


def psi(expected, actual, eps=1e-4):
    """Population stability index between two categorical share vectors (aligned index)."""
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def tag(df):
    df.insert(0, 'analysis_status', POST_HOC)
    return df


# ---------------------------------------------------------------- 1. feature information audit
def feature_information_audit(train, val, test):
    """Month and SpecialDay: how much information do they carry at evaluation time?"""
    rows = []
    for name, part in [('train', train), ('Oct', val), ('Nov', test[test._month == 11]),
                       ('Dec', test[test._month == 12])]:
        rows.append({'partition': name, 'n': len(part),
                     'month_values': ','.join(sorted(part.Month.unique(), key=MONTH_ORDER.index)),
                     'month_seen_in_training': bool(part.Month.isin(train.Month.unique()).all()),
                     'specialday_nonzero_rows': int((part.SpecialDay > 0).sum()),
                     'new_visitor_share': float((part.VisitorType == 'New_Visitor').mean()),
                     'purchase_rate': float(part.Revenue.mean())})
    by_month = (train.assign(sd=train.SpecialDay > 0).groupby('Month')
                .agg(n=('Revenue', 'size'), specialday_nonzero=('sd', 'sum'), purchase_rate=('Revenue', 'mean'))
                .reindex([m for m in MONTH_ORDER if m in set(train.Month)]).reset_index())
    by_month['share_of_training'] = by_month.n / by_month.n.sum()
    return tag(pd.DataFrame(rows)), tag(by_month)


# ---------------------------------------------------------------- 2. ablation without Month/SpecialDay
def ablation(train, val, test):
    sets = {'arrival_context (primary)': ex.CONTEXT,
            'arrival_context minus Month': [c for c in ex.CONTEXT if c != 'Month'],
            'arrival_context minus Month and SpecialDay': [c for c in ex.CONTEXT if c not in ('Month', 'SpecialDay')],
            'VisitorType only': ['VisitorType']}
    rows = []
    for label, cols in sets.items():
        for name in ['logistic', 'random_forest', 'hist_boost']:
            pipe = ex.make_model(name, cols).fit(train[cols], train.Revenue)
            cut = cutoff_from(pipe.predict_proba(val[cols])[:, 1])
            for period, part in periods(val, test):
                y, p = part.Revenue.to_numpy(), pipe.predict_proba(part[cols])[:, 1]
                fixed = ex.policy_metrics(y, p, p >= cut, BUDGET)
                rows.append({'feature_set': label, 'model': name, 'period': period,
                             'average_precision': average_precision_score(y, p),
                             'roc_auc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p),
                             'mean_score': float(p.mean()), 'purchase_rate': float(y.mean()),
                             'precision_top10': ex.metrics(y, p)['precision_at_10pct'],
                             'fixed_cutoff': cut, 'fixed_selected': fixed['selected'],
                             'budget': fixed['budget'], 'fixed_precision': fixed['precision']})
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 3. simple rule baselines
def rule_baselines(train, val, test, primary):
    """Does the model beat rules a manager could write without ML?"""
    rng_seed = ex.SEED
    rates_vt = train.groupby('VisitorType').Revenue.mean()
    rates_tt = train.groupby('TrafficType').Revenue.mean()
    rates_vt_tt = train.groupby(['VisitorType', 'TrafficType']).Revenue.mean()
    prior = train.Revenue.mean()

    def lookup(part, table, keys):
        idx = pd.MultiIndex.from_frame(part[keys]) if len(keys) > 1 else part[keys[0]]
        return table.reindex(idx).fillna(prior).to_numpy()

    rules = {
        'Model: hist_boost arrival-context (primary)': lambda part: primary.predict_proba(part[ex.CONTEXT])[:, 1],
        'Rule: new visitors first, random within group': lambda part: (part.VisitorType == 'New_Visitor').to_numpy(float),
        'Rule: training purchase rate by VisitorType': lambda part: lookup(part, rates_vt, ['VisitorType']),
        'Rule: training purchase rate by TrafficType': lambda part: lookup(part, rates_tt, ['TrafficType']),
        'Rule: training rate by VisitorType x TrafficType': lambda part: lookup(part, rates_vt_tt, ['VisitorType', 'TrafficType']),
        'Random selection (expected)': None}
    rows = []
    for label, fn in rules.items():
        for period, part in periods(val, test):
            y = part.Revenue.to_numpy()
            if fn is None:
                rows.append({'rule': label, 'period': period, 'precision_top10': float(y.mean()),
                             'buyers_top10': float(y.mean() * np.floor(len(y) * BUDGET)),
                             'average_precision': float(y.mean()), 'roc_auc': .5})
                continue
            s = fn(part)
            mask = ex.top_mask(s, BUDGET, rng_seed)
            rows.append({'rule': label, 'period': period, 'precision_top10': float(y[mask].mean()),
                         'buyers_top10': int(y[mask].sum()),
                         'average_precision': float(average_precision_score(y, s)),
                         'roc_auc': float(roc_auc_score(y, s))})
    return tag(pd.DataFrame(rows))


def rule_tie_spread(val, test, repeats=200):
    """'New visitors first' needs random tie-breaking; report the spread over tie-break seeds."""
    rows = []
    for period, part in periods(val, test):
        y = part.Revenue.to_numpy()
        s = (part.VisitorType == 'New_Visitor').to_numpy(float)
        prec = [y[ex.top_mask(s, BUDGET, seed)].mean() for seed in range(repeats)]
        rows.append({'period': period, 'repeats': repeats, 'precision_mean': float(np.mean(prec)),
                     'precision_p2_5': float(np.quantile(prec, .025)),
                     'precision_p97_5': float(np.quantile(prec, .975))})
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 4. uncertainty of the capacity finding
def cutoff_uncertainty(val, test, primary, repeats=2000):
    """Bootstrap October rows -> re-derive the 10% cutoff -> count Nov/Dec selections.

    Conditional on the fitted model: it captures uncertainty from estimating the cutoff
    on 549 October sessions, not model-refitting or temporal-dependence uncertainty."""
    rng = np.random.default_rng(ex.SEED)
    pval = primary.predict_proba(val[ex.CONTEXT])[:, 1]
    scores = {m: primary.predict_proba(part[ex.CONTEXT])[:, 1] for m, part in periods(val, test)[1:]}
    budgets = {m: int(np.floor(len(s) * BUDGET)) for m, s in scores.items()}
    draws = {m: [] for m in scores}
    for _ in range(repeats):
        cut = cutoff_from(pval[rng.integers(0, len(pval), len(pval))])
        for m, s in scores.items():
            draws[m].append(int((s >= cut).sum()))
    point = cutoff_from(pval)
    rows = []
    for m, d in draws.items():
        d = np.asarray(d)
        rows.append({'period': m, 'budget': budgets[m], 'selected_point': int((scores[m] >= point).sum()),
                     'selected_p2_5': float(np.quantile(d, .025)), 'selected_median': float(np.median(d)),
                     'selected_p97_5': float(np.quantile(d, .975)),
                     'share_of_draws_under_budget': float((d < budgets[m]).mean()),
                     'share_of_draws_over_budget': float((d > budgets[m]).mean()),
                     'distinct_score_values_in_month': int(len(np.unique(scores[m]))),
                     'repeats': repeats})
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 5. seed sensitivity
def seed_sensitivity(train, val, test, seeds=range(161, 171)):
    rows = []
    for seed in seeds:
        aps = {}
        for name in ['logistic', 'random_forest', 'hist_boost']:
            pipe = fit_primary(train, name=name, seed=seed)
            p = pipe.predict_proba(val[ex.CONTEXT])[:, 1]
            aps[name] = (average_precision_score(val.Revenue, p), brier_score_loss(val.Revenue, p), pipe, p)
        winner = sorted(aps, key=lambda k: (-aps[k][0], aps[k][1], k))[0]
        pipe, pval = aps[winner][2], aps[winner][3]
        cut = cutoff_from(pval)
        row = {'seed': seed, 'winner': winner, **{f'oct_ap_{k}': v[0] for k, v in aps.items()}}
        for m, part in periods(val, test)[1:]:
            p = pipe.predict_proba(part[ex.CONTEXT])[:, 1]
            row[f'{m}_fixed_selected'] = int((p >= cut).sum())
            row[f'{m}_budget'] = int(np.floor(len(p) * BUDGET))
            row[f'{m}_batch_precision'] = float(part.Revenue.to_numpy()[ex.top_mask(p, BUDGET, seed)].mean())
        rows.append(row)
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 6. recalibration
def recalibration(val, test, primary):
    """Fit Platt and isotonic maps on October predictions only; evaluate on Nov/Dec.

    Ranking (and so top-10% selection) is unchanged by Platt and nearly unchanged by isotonic;
    the question is whether predicted buyer volume becomes more reliable."""
    pval = primary.predict_proba(val[ex.CONTEXT])[:, 1]
    yval = val.Revenue.to_numpy()
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(pval / (1 - pval)).reshape(-1, 1), yval)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4).fit(pval, yval)
    maps = {'uncalibrated': lambda p: p,
            'platt (fit on Oct)': lambda p: platt.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1],
            'isotonic (fit on Oct)': lambda p: iso.predict(p)}
    rows = []
    for period, part in periods(val, test)[1:]:
        y = part.Revenue.to_numpy()
        raw = primary.predict_proba(part[ex.CONTEXT])[:, 1]
        mask = ex.top_mask(raw, BUDGET)
        for label, f in maps.items():
            p = f(raw)
            rows.append({'period': period, 'calibration': label, 'brier': brier_score_loss(y, p),
                         'log_loss': log_loss(y, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1]),
                         'predicted_buyers_all': float(p.sum()), 'actual_buyers_all': int(y.sum()),
                         'predicted_buyers_top10': float(p[mask].sum()), 'actual_buyers_top10': int(y[mask].sum()),
                         'mean_predicted': float(p.mean()), 'observed_rate': float(y.mean())})
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 7. drift diagnostics
def drift(train, val, test, primary):
    cats = ['VisitorType', 'TrafficType', 'Browser', 'OperatingSystems', 'Region', 'Weekend']
    rows = []
    for ref_name, ref in [('train', train), ('Oct', val)]:
        for period, part in periods(val, test):
            if period == ref_name:
                continue
            for c in cats:
                levels = sorted(set(ref[c]) | set(part[c]))
                e = ref[c].value_counts().reindex(levels, fill_value=0)
                a = part[c].value_counts().reindex(levels, fill_value=0)
                rows.append({'reference': ref_name, 'period': period, 'feature': c, 'psi': psi(e, a)})
            # Score distribution PSI on reference deciles.
            sref = primary.predict_proba(ref[ex.CONTEXT])[:, 1]
            sp = primary.predict_proba(part[ex.CONTEXT])[:, 1]
            edges = np.unique(np.quantile(sref, np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            e = np.histogram(sref, edges)[0]
            a = np.histogram(sp, edges)[0]
            rows.append({'reference': ref_name, 'period': period, 'feature': 'model score (decile bins)', 'psi': psi(e, a)})
    psi_df = tag(pd.DataFrame(rows))
    # What does the cutoff actually select? Composition of the selected group by VisitorType.
    cut = cutoff_from(primary.predict_proba(val[ex.CONTEXT])[:, 1])
    comp = []
    for period, part in periods(val, test):
        p = primary.predict_proba(part[ex.CONTEXT])[:, 1]
        for vt in sorted(part.VisitorType.unique()):
            g = (part.VisitorType == vt).to_numpy()
            comp.append({'period': period, 'visitor_type': vt, 'n': int(g.sum()),
                         'share_of_month': float(g.mean()),
                         'mean_score': float(p[g].mean()),
                         'share_above_october_cutoff': float((p[g] >= cut).mean()),
                         'selected_by_fixed_cutoff': int((p[g] >= cut).sum())})
    return psi_df, tag(pd.DataFrame(comp))


# ---------------------------------------------------------------- 8. permutation importance
def importance(val, test, primary, repeats=30):
    rows = []
    for period, part in periods(val, test):
        r = permutation_importance(primary, part[ex.CONTEXT], part.Revenue, scoring='average_precision',
                                   n_repeats=repeats, random_state=ex.SEED, n_jobs=1)
        for i, c in enumerate(ex.CONTEXT):
            rows.append({'period': period, 'feature': c, 'ap_drop_mean': float(r.importances_mean[i]),
                         'ap_drop_std': float(r.importances_std[i]), 'repeats': repeats})
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 9. label-free rolling cutoff policy
def rolling_policy(val, test, primary):
    """A deployable middle policy: each month's cutoff is the previous month's 10% score quantile.

    Uses only previous-month SCORES (no purchase labels), so it could run in production.
    Nov uses Oct (identical to the frozen cutoff); Dec uses Nov. One informative month only."""
    sc = {m: primary.predict_proba(part[ex.CONTEXT])[:, 1] for m, part in periods(val, test)}
    ys = {m: part.Revenue.to_numpy() for m, part in periods(val, test)}
    frozen = cutoff_from(sc['Oct'])
    rows = []
    for m, prev in [('Nov', 'Oct'), ('Dec', 'Nov')]:
        y, p = ys[m], sc[m]
        for label, mask in [('fixed_october_cutoff', p >= frozen),
                            ('rolling_previous_month_cutoff', p >= cutoff_from(sc[prev])),
                            ('batch_top_budget (offline reference)', ex.top_mask(p, BUDGET))]:
            rows.append({'period': m, 'policy': label, **ex.policy_metrics(y, p, mask, BUDGET)})
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 10. capacity curve and cost scenarios
def capacity_curve(val, test, primary, fractions=np.round(np.arange(.02, .31, .02), 2)):
    pval = primary.predict_proba(val[ex.CONTEXT])[:, 1]
    rows = []
    for f in fractions:
        cut = cutoff_from(pval, f)
        for m, part in periods(val, test)[1:]:
            p = primary.predict_proba(part[ex.CONTEXT])[:, 1]
            k = int((p >= cut).sum())
            rows.append({'target_fraction': float(f), 'period': m, 'selected_rate': k / len(p),
                         'selected': k, 'budget': int(np.floor(len(p) * f))})
    return tag(pd.DataFrame(rows))


def cost_scenarios(policy_csv, ratios=(.5, 1, 2, 4)):
    """Capacity-mismatch cost in units of one idle slot: idle + r * overflow.

    r is an ASSUMED ratio of the cost of one overflow session (overtime, delay, unserved
    customer) to one idle slot. No real cost data exists in this project."""
    pol = pd.read_csv(policy_csv)
    pol = pol[np.isclose(pol.fraction, BUDGET) & pol.period.isin(['Nov', 'Dec'])]
    rows = []
    for r in ratios:
        for _, x in pol.iterrows():
            rows.append({'overflow_to_idle_cost_ratio': r, 'period': x.period, 'policy': x.policy,
                         'unused_capacity': int(x.unused_capacity), 'over_budget': int(x.over_budget),
                         'mismatch_cost_idle_slot_units': float(x.unused_capacity + r * x.over_budget)})
    return tag(pd.DataFrame(rows))


# ---------------------------------------------------------------- 11. train-only temporal tuning
def temporal_tuning(train, val, test):
    """Hyperparameters chosen by expanding-window CV inside the training months only.

    Folds: validate on each of June, Jul, Aug, Sep using all earlier training months."""
    grid = {'logistic': [{'C': c} for c in (.01, .1, 1, 10)],
            'random_forest': [{'min_samples_leaf': m, 'n_estimators': 200} for m in (5, 10, 25, 50)],
            'hist_boost': [{'max_leaf_nodes': l, 'learning_rate': lr, 'max_iter': it}
                           for l in (7, 15, 31) for lr in (.03, .1) for it in (75, 150)]}
    folds = [m for m in (6, 7, 8, 9) if (train._month == m).any()]
    cv_rows = []
    best = {}
    for name, params in grid.items():
        scored = []
        for prm in params:
            aps = []
            for m in folds:
                tr, va = train[train._month < m], train[train._month == m]
                if tr.Revenue.nunique() < 2 or va.Revenue.nunique() < 2:
                    continue
                pipe = ex.make_model(name, ex.CONTEXT)
                pipe.set_params(**{f'model__{k}': v for k, v in prm.items()})
                pipe.fit(tr[ex.CONTEXT], tr.Revenue)
                aps.append(average_precision_score(va.Revenue, pipe.predict_proba(va[ex.CONTEXT])[:, 1]))
            cv_rows.append({'model': name, 'params': json.dumps(prm), 'folds': len(aps),
                            'cv_mean_ap': float(np.mean(aps)), 'cv_std_ap': float(np.std(aps))})
            scored.append((float(np.mean(aps)), json.dumps(prm, sort_keys=True), prm))
        best[name] = sorted(scored, key=lambda t: (-t[0], t[1]))[0][2]
    eval_rows = []
    for name, prm in best.items():
        pipe = ex.make_model(name, ex.CONTEXT)
        pipe.set_params(**{f'model__{k}': v for k, v in prm.items()})
        pipe.fit(train[ex.CONTEXT], train.Revenue)
        cut = cutoff_from(pipe.predict_proba(val[ex.CONTEXT])[:, 1])
        for period, part in periods(val, test):
            y, p = part.Revenue.to_numpy(), pipe.predict_proba(part[ex.CONTEXT])[:, 1]
            eval_rows.append({'model': name, 'tuned_params': json.dumps(prm), 'period': period,
                              'average_precision': average_precision_score(y, p), 'roc_auc': roc_auc_score(y, p),
                              'precision_top10': ex.metrics(y, p)['precision_at_10pct'],
                              'fixed_selected': int((p >= cut).sum()), 'budget': int(np.floor(len(p) * BUDGET))})
    return tag(pd.DataFrame(cv_rows)), tag(pd.DataFrame(eval_rows))


# ---------------------------------------------------------------- figures
def figures(out, curve, comp, imp, rules, val, test, primary):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([0, .3], [0, .3], 'k--', lw=1, label='Selected = target')
    for m, c in [('Nov', '#167d9a'), ('Dec', '#e89438')]:
        d = curve[curve.period == m]
        ax.plot(d.target_fraction, d.selected_rate, 'o-', color=c, ms=3, label=f'{m}: frozen October cutoff')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel='Target selection share (cutoff set on October)', ylabel='Share actually selected',
           title='Capacity drift across all budgets')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / '04_capacity_curve.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    piv = imp.pivot(index='feature', columns='period', values='ap_drop_mean')[['Oct', 'Nov', 'Dec']]
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    piv.plot.barh(ax=ax, color=['#888888', '#167d9a', '#e89438'])
    ax.axvline(0, color='black', lw=.8)
    ax.set(xlabel='Drop in average precision when feature is shuffled', ylabel='',
           title='Permutation importance, primary model')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / '05_permutation_importance.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    piv = rules.pivot(index='rule', columns='period', values='precision_top10')[['Oct', 'Nov', 'Dec']]
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    piv.plot.barh(ax=ax, color=['#888888', '#167d9a', '#e89438'])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel='Purchase rate among the top 10% selected', ylabel='', title='Model versus simple rules')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / '06_rule_baselines.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    piv = comp.pivot(index='period', columns='visitor_type', values='share_above_october_cutoff').reindex(['Oct', 'Nov', 'Dec'])
    piv.plot.bar(ax=ax, color=['#167d9a', '#888888', '#e89438'])
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.tick_params(axis='x', labelrotation=0)
    ax.set(xlabel='', ylabel='Share above frozen cutoff', title='Who the October cutoff selects')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / '07_selection_by_visitor_type.png', dpi=180); plt.close(fig)


# ---------------------------------------------------------------- driver
def run(raw, output_dir='results_extended', policy_csv='results/policy_comparison.csv',
        locked_json='results/locked_decisions.json', quick=False):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df, split, _ = ex.prepare(raw)
    train, val, test = split['train'], split['validation'], split['test']
    primary = fit_primary(train)
    cut = cutoff_from(primary.predict_proba(val[ex.CONTEXT])[:, 1])
    if Path(locked_json).exists() and not quick:
        locked = json.loads(Path(locked_json).read_text())['thresholds']['0.1']
        if not np.isclose(cut, locked, rtol=0, atol=1e-9):
            raise RuntimeError(f'Refit primary cutoff {cut} != locked {locked}; extended results would not '
                               'describe the published model. Check data file and package versions.')
    n_boot = 200 if quick else 2000
    seeds = range(161, 163) if quick else range(161, 171)

    info, train_months = feature_information_audit(train, val, test)
    info.to_csv(out / 'feature_information_audit.csv', index=False)
    train_months.to_csv(out / 'training_month_composition.csv', index=False)
    ablation(train, val, test).to_csv(out / 'ablation_month_specialday.csv', index=False)
    rules = rule_baselines(train, val, test, primary); rules.to_csv(out / 'rule_baselines.csv', index=False)
    rule_tie_spread(val, test, 20 if quick else 200).to_csv(out / 'rule_new_visitor_tie_spread.csv', index=False)
    cutoff_uncertainty(val, test, primary, n_boot).to_csv(out / 'cutoff_uncertainty.csv', index=False)
    seed_sensitivity(train, val, test, seeds).to_csv(out / 'seed_sensitivity.csv', index=False)
    recalibration(val, test, primary).to_csv(out / 'recalibration.csv', index=False)
    psi_df, comp = drift(train, val, test, primary)
    psi_df.to_csv(out / 'drift_psi.csv', index=False); comp.to_csv(out / 'selection_by_visitor_type.csv', index=False)
    imp = importance(val, test, primary, 5 if quick else 30); imp.to_csv(out / 'permutation_importance.csv', index=False)
    rolling_policy(val, test, primary).to_csv(out / 'rolling_cutoff_policy.csv', index=False)
    curve = capacity_curve(val, test, primary); curve.to_csv(out / 'capacity_curve.csv', index=False)
    if Path(policy_csv).exists():
        cost_scenarios(policy_csv).to_csv(out / 'capacity_cost_scenarios.csv', index=False)
    if not quick:
        cv, tuned = temporal_tuning(train, val, test)
        cv.to_csv(out / 'temporal_cv_grid.csv', index=False); tuned.to_csv(out / 'temporal_cv_tuned_eval.csv', index=False)
    figures(out, curve, comp, imp, rules, val, test, primary)
    (out / 'README.md').write_text(
        '# Extended analyses: post-hoc\n\nGenerated by extended_analysis.py. Every table here was designed after '
        'the locked held-out results were seen. It explains, stresses and qualifies the primary result in '
        'results/; it does not replace it. See EXTENDED_FINDINGS.md in the repository root.\n', encoding='utf-8')
    print('Extended outputs:', out.resolve())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/source_dataset.csv')
    ap.add_argument('--output', default='results_extended')
    args = ap.parse_args()
    data, _ = ex.load_data(args.csv)
    run(data, args.output)
