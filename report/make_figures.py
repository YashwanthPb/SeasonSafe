"""Build the multi-panel figures used in report/seasonsafe.tex.

Run from the repository root after experiment.py and extended_analysis.py outputs exist:
    python report/make_figures.py
Reads results/, results_extended/ and refits the primary model (identical to the locked one).
Panels marked "post-hoc" use analyses designed after the held-out results were seen.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment as ex  # noqa: E402
import extended_analysis as ea  # noqa: E402

RES, EXT, OUT = ROOT / 'results', ROOT / 'results_extended', ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Categorical slots in fixed order (validated: scripts/validate_palette.js, light mode, all checks pass).
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
GREY, INK, MUTED, GRID = '#8f8e88', '#21211f', '#5f5e59', '#e4e3df'
MONTH_C = {'Oct': GREY, 'Nov': BLUE, 'Dec': ORANGE}
FULL, COL = 7.16, 3.5

plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.titleweight': 'bold', 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'legend.frameon': False,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5, 'axes.axisbelow': True,
    'lines.linewidth': 1.4, 'patch.linewidth': 0, 'savefig.dpi': 300, 'figure.dpi': 100,
    'font.family': 'DejaVu Sans'})


def title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', pad=4)


def bars(ax, x, h, w, color, label=None, **kw):
    return ax.bar(x, h, w, color=color, label=label, edgecolor='white', linewidth=.8, **kw)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    print('wrote', name)


# ------------------------------------------------------------------ data
raw = pd.read_csv(RES / 'source_dataset.csv')
df, split, _ = ex.prepare(raw)
train, val, test = split['train'], split['validation'], split['test']
primary = ea.fit_primary(train)
parts = dict(ea.periods(val, test))
scores = {m: primary.predict_proba(p[ex.CONTEXT])[:, 1] for m, p in parts.items()}
ys = {m: p.Revenue.to_numpy() for m, p in parts.items()}
cut = ea.cutoff_from(scores['Oct'])


# ------------------------------------------------------------------ Fig A: data overview (column)
def fig_data():
    mr = pd.read_csv(RES / 'monthly_purchase_rates.csv').sort_values('_month')
    role = np.where(mr._month <= 9, 'Train', np.where(mr._month == 10, 'Validation', 'Test'))
    colors = {'Train': GREY, 'Validation': YELLOW, 'Test': BLUE}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.55), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    x = np.arange(len(mr))
    bars(a1, x, mr['size'], .7, [colors[r] for r in role])
    for xi, v in zip(x, mr['size']):
        a1.text(xi, v + 60, f'{v:,}', ha='center', va='bottom', fontsize=5.6, color=MUTED)
    a1.set(ylabel='Sessions', ylim=(0, 4000)); title(a1, 'a', 'Sessions per month')
    bars(a2, x, mr['mean'], .7, [colors[r] for r in role])
    a2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    a2.set(ylabel='Purchase rate'); title(a2, 'b', 'Observed purchase rate')
    a2.set_xticks(x, mr.Month)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    a2.set_ylim(0, .34)
    a2.legend(handles, ['Train (Feb to Sep)', 'Validation (Oct)', 'Test (Nov, Dec)'], loc='upper left', ncol=3, fontsize=5.8)
    fig.tight_layout(h_pad=.6)
    save(fig, 'fA_data.png')


# ------------------------------------------------------------------ Fig B: capacity (full width)
def fig_capacity():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.15), gridspec_kw={'width_ratios': [1.05, 1, 1]})
    # (a) selected as % of budget by policy
    ax = axs[0]
    pol = pd.read_csv(RES / 'policy_comparison.csv')
    pol = pol[np.isclose(pol.fraction, .1)]
    roll = pd.read_csv(EXT / 'rolling_cutoff_policy.csv')
    rows = []
    for m, mm in [('Oct', 'Oct_validation'), ('Nov', 'Nov'), ('Dec', 'Dec')]:
        f = pol[(pol.period == mm) & (pol.policy == 'fixed_october_cutoff')].iloc[0]
        b = pol[(pol.period == mm) & (pol.policy == 'batch_top_budget')].iloc[0]
        r = roll[(roll.period == m) & (roll.policy == 'rolling_previous_month_cutoff')]
        rows.append((m, f.selected / f.budget, (r.selected.iloc[0] / r.budget.iloc[0]) if len(r) else np.nan,
                     b.selected / b.budget, f.selected, f.budget))
    x = np.arange(3); w = .26
    for i, (lab, col, idx) in enumerate([('Frozen October cutoff', BLUE, 1), ('Rolling prev.-month cutoff', AQUA, 2),
                                         ('Monthly top 10% (offline)', ORANGE, 3)]):
        vals = [r[idx] for r in rows]
        bars(ax, x + (i - 1) * w, vals, w, col, lab)
    ax.axhline(1, color=INK, lw=.8, ls='--')
    for xi, r in zip(x, rows):
        ax.text(xi - w, r[1] + .02, f'{r[4]}/{r[5]}', ha='center', va='bottom', fontsize=5.5, color=MUTED)
    ax.set_xticks(x, ['Oct (val.)', 'Nov', 'Dec']); ax.set_ylim(0, 1.75)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_ylabel('Selected / budget')
    ax.legend(loc='upper left', ncol=1, fontsize=5.6)
    ax.text(.5, 1.02, 'budget', fontsize=5.6, color=MUTED, ha='center', va='bottom')
    title(ax, 'a', 'Capacity use at a 10% budget')
    # (b) capacity curve
    ax = axs[1]
    cc = pd.read_csv(EXT / 'capacity_curve.csv')
    ax.plot([0, .3], [0, .3], color=INK, lw=.8, ls='--', label='Selected = target')
    for m in ['Nov', 'Dec']:
        d = cc[cc.period == m]
        ax.plot(d.target_fraction, d.selected_rate, '-o', ms=2.5, color=MONTH_C[m], label=m)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel='Target share (cutoff set on Oct)', ylabel='Share selected')
    ax.legend(loc='upper left'); title(ax, 'b', 'Every budget level (post-hoc)')
    # (c) bootstrap distribution of selected counts
    ax = axs[2]
    rng = np.random.default_rng(ex.SEED)
    pv = scores['Oct']
    draws = {m: [] for m in ['Nov', 'Dec']}
    for _ in range(2000):
        c = ea.cutoff_from(pv[rng.integers(0, len(pv), len(pv))])
        for m in draws:
            draws[m].append((scores[m] >= c).sum() / np.floor(len(scores[m]) * .1))
    edges = np.arange(.55, 1.42, .04)
    for m in ['Nov', 'Dec']:
        ax.hist(draws[m], bins=edges, color=MONTH_C[m], alpha=.55, label=m, edgecolor='white', linewidth=.5)
    ax.axvline(1, color=INK, lw=.8, ls='--')
    for m in ['Nov', 'Dec']:
        lo, hi = np.quantile(draws[m], [.025, .975])
        ax.text(.57, ax.get_ylim()[1] * (.9 if m == 'Nov' else .78), f'{m} 95%: {lo:.0%} to {hi:.0%}',
                fontsize=5.8, color=MONTH_C[m] if m == 'Nov' else '#b4461b')
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel='Selected / budget', ylabel='Bootstrap resamples')
    ax.legend(loc='upper right'); title(ax, 'c', 'Cutoff estimation noise (post-hoc)')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fB_capacity.png')


# ------------------------------------------------------------------ Fig C: model quality (full width)
def fig_models():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.15))
    # (a) October validation AP
    ax = axs[0]
    vm = pd.read_csv(RES / 'validation_models.csv')
    fams = [('arrival_context', 'Arrival context'), ('session_no_pagevalues', 'Session, no PageValues'),
            ('session_with_pagevalues', 'Session + PageValues')]
    models = [('logistic', 'Logistic', BLUE), ('random_forest', 'Random forest', ORANGE), ('hist_boost', 'Hist. boosting', AQUA)]
    x = np.arange(3); w = .26
    for i, (mk, ml, col) in enumerate(models):
        v = [vm[(vm.family == f) & (vm.model == mk)].average_precision.iloc[0] for f, _ in fams]
        bars(ax, x + (i - 1) * w, v, w, col, ml)
    ax.axhline(val.Revenue.mean(), color=INK, lw=.8, ls=':')
    ax.text(-.45, val.Revenue.mean() + .015, 'Oct base rate', fontsize=5.6, color=MUTED, ha='left')
    ax.set_xticks(x, ['Arrival\ncontext', 'Session,\nno PageValues', 'Session +\nPageValues'])
    ax.set(ylabel='Average precision', ylim=(0, .9)); ax.legend(loc='upper left')
    title(ax, 'a', 'October model selection')
    # (b) PR curves pooled Nov-Dec, hist_boost by family
    ax = axs[1]
    tp = pd.read_csv(RES / 'test_predictions.csv', float_precision='round_trip')
    for (f, lab), col in zip(fams, [BLUE, ORANGE, AQUA]):
        d = tp[(tp.family == f) & (tp.model == 'hist_boost')]
        p, r, _ = precision_recall_curve(d.y, d.p)
        ax.plot(r, p, color=col, label=lab, lw=1.2)
    base = tp[tp.family == 'arrival_context'].y.mean()
    ax.axhline(base, color=INK, lw=.8, ls=':')
    ax.text(.5, base - .06, 'random (base rate)', fontsize=5.6, color=MUTED, ha='center')
    ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02)); ax.legend(loc='upper right')
    title(ax, 'b', 'PR curves, Nov and Dec (boosting)')
    # (c) cumulative gains pooled Nov-Dec
    ax = axs[2]
    y = np.concatenate([ys['Nov'], ys['Dec']])
    s_model = np.concatenate([scores['Nov'], scores['Dec']])
    vt = pd.concat([parts['Nov'].VisitorType, parts['Dec'].VisitorType]).to_numpy()
    s_rule = (vt == 'New_Visitor').astype(float)
    fr = np.linspace(0, 1, 101)

    def gains(s):
        order = np.lexsort((np.random.default_rng(ex.SEED).random(len(s)), -s))
        cum = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
        return cum[(fr * len(y)).astype(int)]
    ax.plot(fr, fr, color=INK, lw=.8, ls=':', label='Random')
    ax.plot(fr, gains(s_rule), color=ORANGE, label='New visitors first')
    ax.plot(fr, gains(s_model), color=BLUE, label='Model (arrival)')
    sess = tp[(tp.family == 'session_with_pagevalues') & (tp.model == 'hist_boost')].set_index('row_id')
    ids = np.concatenate([parts['Nov']._row_id, parts['Dec']._row_id])
    ax.plot(fr, gains(sess.loc[ids].p.to_numpy()), color=AQUA, label='Session + PageValues')
    ax.axvline(.1, color=GRID, lw=1.2, zorder=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel='Share of sessions selected', ylabel='Share of buyers captured', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc='lower right', fontsize=5.6); title(ax, 'c', 'Cumulative gains, Nov and Dec')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fC_models.png')


# ------------------------------------------------------------------ Fig D: drift (full width)
def fig_drift():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.15), gridspec_kw={'width_ratios': [1, 1.1, 1]})
    # (a) score distributions
    ax = axs[0]
    edges = np.linspace(0, .55, 34)
    for m in ['Oct', 'Nov', 'Dec']:
        h, _ = np.histogram(scores[m], bins=edges)
        ax.step(edges[:-1], h / h.sum(), where='post', color=MONTH_C[m], label=m, lw=1.2)
    ax.axvline(cut, color=INK, lw=.8, ls='--')
    ax.text(cut + .007, ax.get_ylim()[1] * .55, 'frozen\nOctober\ncutoff', fontsize=5.6, color=MUTED, va='top', ha='left')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel='Predicted purchase probability', ylabel='Share of sessions'); ax.legend(loc='upper right')
    title(ax, 'a', 'Score distribution by month')
    # (b) traffic mix
    ax = axs[1]
    d = pd.concat([parts[m].assign(m=m) for m in ['Oct', 'Nov', 'Dec']])
    grp = d.TrafficType.astype(int).map(lambda t: 'Types 4, 5' if t in (4, 5) else 'Types 8, 10' if t in (8, 10)
                            else f'Type {t}' if t in (1, 2, 3) else 'Other')
    share = pd.crosstab(grp, d.m, normalize='columns')[['Oct', 'Nov', 'Dec']]
    share = share.loc[['Type 1', 'Type 2', 'Type 3', 'Types 4, 5', 'Types 8, 10', 'Other']]
    x = np.arange(len(share)); w = .26
    for i, m in enumerate(['Oct', 'Nov', 'Dec']):
        bars(ax, x + (i - 1) * w, share[m], w, MONTH_C[m], m)
    ax.set_xticks(x, [i.replace('Types ', '').replace('Type ', '') for i in share.index])
    ax.set_xlabel('Traffic type (code)')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_ylabel('Share of sessions')
    ax.legend(loc='upper right'); title(ax, 'b', 'Traffic-source mix shifts')
    # (c) PSI vs October
    ax = axs[2]
    psi = pd.read_csv(EXT / 'drift_psi.csv')
    psi = psi[psi.reference == 'Oct']
    feats = ['Weekend', 'Region', 'Browser', 'OperatingSystems', 'VisitorType', 'TrafficType']
    y = np.arange(len(feats)); h = .38
    for i, m in enumerate(['Nov', 'Dec']):
        v = [psi[(psi.period == m) & (psi.feature == f)].psi.iloc[0] for f in feats]
        ax.barh(y + (i - .5) * h, v, h, color=MONTH_C[m], label=f'{m} vs Oct', edgecolor='white', linewidth=.8)
    ax.set_xscale('log'); ax.set_xlim(1e-4, 8)
    ax.axvline(.25, color=INK, lw=.8, ls='--')
    ax.text(.28, 1.6, '0.25', fontsize=5.4, color=MUTED)
    ax.set_yticks(y, ['Weekend', 'Region', 'Browser', 'OS', 'VisitorType', 'TrafficType'])
    ax.set_xlabel('Population stability index (log scale)'); ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right'); title(ax, 'c', 'Feature drift vs October (post-hoc)')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fD_drift.png')


# ------------------------------------------------------------------ Fig E: what the model learned (full width)
def fig_learned():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.15))
    # (a) share above cutoff by visitor type
    ax = axs[0]
    comp = pd.read_csv(EXT / 'selection_by_visitor_type.csv')
    x = np.arange(3); w = .36
    for i, (vt, lab, col) in enumerate([('New_Visitor', 'New visitors', BLUE), ('Returning_Visitor', 'Returning visitors', ORANGE)]):
        v = [comp[(comp.period == m) & (comp.visitor_type == vt)].share_above_october_cutoff.iloc[0] for m in ['Oct', 'Nov', 'Dec']]
        bb = bars(ax, x + (i - .5) * w, v, w, col, lab)
        for r, val_ in zip(bb, v):
            ax.text(r.get_x() + r.get_width() / 2, val_ + .01, f'{val_:.0%}', ha='center', fontsize=5.6, color=MUTED)
    ax.set_xticks(x, ['Oct', 'Nov', 'Dec']); ax.set_ylim(0, .75)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_ylabel('Share above frozen cutoff')
    ax.legend(loc='upper left', ncol=1); title(ax, 'a', 'Who the cutoff selects')
    # (b) permutation importance
    ax = axs[1]
    imp = pd.read_csv(EXT / 'permutation_importance.csv')
    piv = imp.pivot(index='feature', columns='period', values='ap_drop_mean')[['Oct', 'Nov', 'Dec']]
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    y = np.arange(len(piv)); h = .26
    for i, m in enumerate(['Oct', 'Nov', 'Dec']):
        ax.barh(y + (i - 1) * h, piv[m], h, color=MONTH_C[m], label=m, edgecolor='white', linewidth=.6)
    ax.axvline(0, color=INK, lw=.6)
    ax.set_yticks(y, [{'OperatingSystems': 'OS'}.get(f, f) for f in piv.index]); ax.grid(axis='y', visible=False)
    ax.set_xlabel('Drop in AP when shuffled'); ax.legend(loc='lower right')
    ax.text(.001, 0, 'exactly 0', fontsize=5.6, color=MUTED, va='center'); ax.text(.001, 1, 'exactly 0', fontsize=5.6, color=MUTED, va='center')
    title(ax, 'b', 'Permutation importance (post-hoc)')
    # (c) ablation: selected/budget
    ax = axs[2]
    ab = pd.read_csv(EXT / 'ablation_month_specialday.csv')
    sets = [('arrival_context (primary)', 'Primary'), ('arrival_context minus Month', '− Month'),
            ('arrival_context minus Month and SpecialDay', '− Month,\nSpecialDay')]
    x = np.arange(3); w = .36
    for i, m in enumerate(['Nov', 'Dec']):
        v = [(lambda r: r.fixed_selected / r.budget)(ab[(ab.feature_set == s) & (ab.model == 'hist_boost') & (ab.period == m)].iloc[0]) for s, _ in sets]
        bars(ax, x + (i - .5) * w, v, w, MONTH_C[m], f'{m}, boosting')
        lg = [(lambda r: r.fixed_selected / r.budget)(ab[(ab.feature_set == s) & (ab.model == 'logistic') & (ab.period == m)].iloc[0]) for s, _ in sets]
        ax.scatter(x + (i - .5) * w, lg, marker='D', s=12, facecolor='white', edgecolor=INK, linewidth=.8, zorder=3,
                   label='Logistic regression' if i == 0 else None)
    ax.axhline(1, color=INK, lw=.8, ls='--')
    ax.set_xticks(x, [l for _, l in sets]); ax.set_ylim(0, 1.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_ylabel('Selected / budget')
    ax.legend(loc='upper right', fontsize=5.6); title(ax, 'c', 'Dropping the unseen Month (post-hoc)')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fE_learned.png')


# ------------------------------------------------------------------ Fig F: probabilities and alternatives (full width)
def fig_probs():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw={'width_ratios': [.9, 1, 1.2]})
    # (a) reliability
    ax = axs[0]
    ax.plot([0, .5], [0, .5], color=INK, lw=.8, ls='--')
    for m in ['Oct', 'Nov', 'Dec']:
        o, p = calibration_curve(ys[m], scores[m], n_bins=8, strategy='quantile')
        ax.plot(p, o, '-o', ms=3, color=MONTH_C[m], label=m)
    ax.set(xlabel='Mean predicted probability', ylabel='Observed purchase rate', xlim=(0, .5), ylim=(0, .5))
    ax.legend(loc='upper left'); title(ax, 'a', 'Reliability by month')
    # (b) predicted vs actual buyers in top 10%
    ax = axs[1]
    rc = pd.read_csv(EXT / 'recalibration.csv')
    cals = [('uncalibrated', 'Uncalibrated', BLUE), ('platt (fit on Oct)', 'Platt (Oct)', ORANGE),
            ('isotonic (fit on Oct)', 'Isotonic (Oct)', AQUA)]
    x = np.arange(2); w = .22
    for i, (k, lab, col) in enumerate(cals):
        v = [rc[(rc.period == m) & (rc.calibration == k)].predicted_buyers_top10.iloc[0] for m in ['Nov', 'Dec']]
        bars(ax, x + (i - 1.5) * w, v, w, col, lab)
    act = [rc[(rc.period == m) & (rc.calibration == 'uncalibrated')].actual_buyers_top10.iloc[0] for m in ['Nov', 'Dec']]
    bars(ax, x + 1.5 * w, act, w, INK, 'Actual')
    ax.set_xticks(x, ['Nov (top 299)', 'Dec (top 172)']); ax.set_ylabel('Buyers in selected group'); ax.set_ylim(0, 140)
    ax.legend(loc='upper right', fontsize=5.6, ncol=2); title(ax, 'b', 'Predicted vs actual buyers (post-hoc)')
    # (c) rule baselines
    ax = axs[2]
    rb = pd.read_csv(EXT / 'rule_baselines.csv')
    rules = [('Random selection (expected)', 'Random'),
             ('Rule: training purchase rate by TrafficType', 'Rule: TrafficType rate'),
             ('Rule: training rate by VisitorType x TrafficType', 'Rule: Visitor × Traffic'),
             ('Rule: new visitors first, random within group', 'Rule: new visitors first'),
             ('Model: hist_boost arrival-context (primary)', 'Model (primary)')]
    y = np.arange(len(rules)); h = .36
    for i, m in enumerate(['Nov', 'Dec']):
        v = [rb[(rb.rule == r) & (rb.period == m)].precision_top10.iloc[0] for r, _ in rules]
        ax.barh(y + (i - .5) * h, v, h, color=MONTH_C[m], label=m, edgecolor='white', linewidth=.8)
    ax.set_yticks(y, [l for _, l in rules]); ax.grid(axis='y', visible=False)
    ax.set_xlim(0, .46); ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_xlabel('Purchase rate in top 10%')
    ax.legend(loc='lower right'); title(ax, 'c', 'Model vs simple rules (post-hoc)')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fF_probs.png')


if __name__ == '__main__':
    fig_data(); fig_capacity(); fig_models(); fig_drift(); fig_learned(); fig_probs()
