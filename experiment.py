"""SeasonSafe research scaffold. AI-generated; not a final assessed submission.
No empirical performance is claimed until run on the cited UCI data.
"""
from pathlib import Path
import io, json, hashlib, platform, urllib.request, zipfile
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

SEED = 161
MONTHS = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'June':6,'Jun':6,
          'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
CAT = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
CONTEXT = CAT + ['SpecialDay']
BEHAVIOR = ['Administrative','Administrative_Duration','Informational',
            'Informational_Duration','ProductRelated','ProductRelated_Duration',
            'BounceRates','ExitRates']
FEATURES = {'arrival_context':CONTEXT,
            'session_no_pagevalues':CONTEXT+BEHAVIOR,
            'session_with_pagevalues':CONTEXT+BEHAVIOR+['PageValues']}
URL = 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'

def load_data(csv_path=None):
    if csv_path:
        raw = Path(csv_path).read_bytes()
    else:
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = [n for n in archive.namelist() if n.endswith('online_shoppers_intention.csv')]
                if len(names) != 1:
                    raise ValueError('Expected one official dataset CSV.')
                raw = archive.read(names[0])
        except Exception as exc:
            raise RuntimeError('Download failed. Download dataset 468 from UCI, unzip it, '
                               'upload online_shoppers_intention.csv to Colab, and set CSV_PATH.') from exc
    return pd.read_csv(io.BytesIO(raw)), hashlib.sha256(raw).hexdigest()

def prepare(raw):
    df = raw.copy()
    expected = set(FEATURES['session_with_pagevalues'] + ['Revenue'])
    if set(df.columns) != expected:
        raise ValueError(f'Unexpected schema. Missing={expected-set(df.columns)}; extra={set(df.columns)-expected}')
    df['_row_id'] = np.arange(len(df))
    for c in ['Revenue','Weekend']:
        values = df[c].astype(str).str.strip().str.lower()
        mapped = values.map({'true':1,'false':0,'1':1,'0':0})
        if mapped.isna().any():
            raise ValueError(f'Invalid or missing boolean in {c}')
        df[c] = mapped.astype(int)
    df['_month'] = df['Month'].map(MONTHS)
    if df['_month'].isna().any():
        raise ValueError('Unrecognized month label.')
    if df['Revenue'].nunique() != 2:
        raise ValueError('Both target classes are required.')
    for c in FEATURES['session_with_pagevalues']:
        if c not in CAT:
            df[c] = pd.to_numeric(df[c], errors='raise')
            if np.isinf(df[c].to_numpy(dtype=float)).any() or (df[c].dropna() < 0).any():
                raise ValueError(f'Invalid numeric values in {c}')
    duplicate_count = int(raw.duplicated().sum())
    # Keep identical observations: no session ID proves they are duplicate records.
    # Month is part of each feature set, so exact feature duplicates cannot cross month splits.
    for c in CAT:
        df[c] = df[c].fillna('Unknown').astype(str)
    groups = {'train':df[df['_month'] <= 9],
              'validation':df[df['_month'] == 10],
              'test':df[df['_month'] >= 11]}
    for name, group in groups.items():
        if group.empty or group.Revenue.nunique() < 2:
            raise ValueError(f'{name} requires rows and both target classes.')
    ids = [set(g._row_id) for g in groups.values()]
    assert not any(ids[i] & ids[j] for i in range(3) for j in range(i+1,3))
    return df, groups, duplicate_count

def make_model(name, columns, quick=False):
    cats = [c for c in columns if c in CAT]
    nums = [c for c in columns if c not in CAT]
    numeric = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    prep = ColumnTransformer([('numeric',numeric,nums),
          ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    models = {
        'logistic':LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        'random_forest':RandomForestClassifier(n_estimators=30 if quick else 200,
                      min_samples_leaf=10, max_features='sqrt', n_jobs=-1, random_state=SEED),
        'hist_boost':HistGradientBoostingClassifier(max_iter=20 if quick else 150,
                      max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False,random_state=SEED)
    }
    return Pipeline([('preprocess',prep),('model',models[name])])

def top_mask(scores, fraction=.10, seed=SEED):
    scores = np.asarray(scores)
    n = len(scores)
    k = max(1, int(np.floor(n*fraction)))
    # Independent seeded tie-breaking avoids favoring original CSV ordering.
    tie = np.random.default_rng(seed).random(n)
    order = np.lexsort((tie,-scores))
    mask = np.zeros(n,dtype=bool)
    mask[order[:k]] = True
    return mask

def metrics(y, p):
    y,p = np.asarray(y),np.asarray(p)
    mask = top_mask(p)
    prevalence = float(y.mean())
    return {'n':len(y),'purchase_rate':prevalence,
            'average_precision':float(average_precision_score(y,p)),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
            'brier':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,np.clip(p,1e-8,1-1e-8),labels=[0,1])),
            'precision_at_10pct':float(y[mask].mean()),
            'recall_at_10pct':float(y[mask].sum()/y.sum()) if y.sum() else np.nan,
            'lift_at_10pct':float(y[mask].mean()/prevalence) if prevalence else np.nan}

def policy_metrics(y,p,mask,fraction):
    y,p,mask = np.asarray(y),np.asarray(p),np.asarray(mask)
    n,k = len(y),int(mask.sum())
    budget = max(1,int(np.floor(n*fraction)))
    return {'n':n,'budget':budget,'selected':k,'selected_rate':k/n,
            'over_budget':max(0,k-budget),'unused_capacity':max(0,budget-k),
            'buyers_selected':int(y[mask].sum()),
            'precision':float(y[mask].mean()) if k else np.nan,
            'recall':float(y[mask].sum()/y.sum()) if y.sum() else np.nan,
            'predicted_buyers_selected':float(p[mask].sum()),
            'actual_minus_predicted_buyers':float(y[mask].sum()-p[mask].sum())}

def paired_bootstrap(y,p,threshold,fraction,repeats=500):
    # Conditional on this fitted model. Row-resampling is approximate, not a time-series guarantee.
    rng = np.random.default_rng(SEED)
    y,p = np.asarray(y),np.asarray(p)
    diffs=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(y),size=len(y)); yy,pp=y[idx],p[idx]
        a=top_mask(pp,fraction); b=pp>=threshold
        if b.any():
            diffs.append(float(yy[a].mean()-yy[b].mean()))
    if len(diffs)<max(10,repeats//2):
        return {'difference_mean':np.nan,'ci_low':np.nan,'ci_high':np.nan,'valid_replicates':len(diffs)}
    return {'difference_mean':float(np.mean(diffs)),
            'ci_low':float(np.quantile(diffs,.025)),'ci_high':float(np.quantile(diffs,.975)),
            'valid_replicates':len(diffs)}

def run_experiment(raw, output_dir='results', data_hash='not supplied', quick=False):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    df,split,duplicates=prepare(raw)
    raw.to_csv(out/'source_dataset.csv',index=False)
    df[FEATURES['session_with_pagevalues']+['Revenue']].to_csv(out/'cleaned_dataset.csv',index=False)
    (out/'DATA_SOURCE.txt').write_text('UCI Online Shoppers Purchasing Intention Dataset (468). '
        'Sakar and Kastro (2018). DOI: 10.24432/C5F88Q. CC BY 4.0. '
        'https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset\n'
        'Check locked_decisions.json: quick_test_only=true means synthetic test data, NOT UCI data.\n',encoding='utf-8')
    train,val,test=split['train'],split['validation'],split['test']
    # October is for model choice; November/December labels are untouched until choices are saved.
    candidates={}; validation=[]
    for family,cols in FEATURES.items():
        for name in ['logistic','random_forest','hist_boost']:
            pipe=make_model(name,cols,quick).fit(train[cols],train.Revenue)
            p=pipe.predict_proba(val[cols])[:,1]
            candidates[(family,name)]=pipe
            validation.append({'family':family,'model':name,**metrics(val.Revenue,p)})
    validation=pd.DataFrame(validation)
    validation.to_csv(out/'validation_models.csv',index=False)
    # Primary policy uses arrival-context variables only. Other families are retrospective diagnostics.
    chosen=validation[validation.family=='arrival_context'].sort_values(
                ['average_precision','brier','model'],ascending=[False,True,True]).iloc[0]['model']
    primary=candidates[('arrival_context',chosen)]
    pval=primary.predict_proba(val[CONTEXT])[:,1]
    budgets=[.05,.10,.20]
    thresholds={str(b):float(np.sort(pval)[-max(1,int(np.floor(len(pval)*b)))]) for b in budgets}
    manifest={'seed':SEED,'data_sha256':data_hash,'primary_family':'arrival_context',
              'selected_model':chosen,'selected_using':'October average precision; Brier tie-break',
              'train_months':'available months February-September (any January also included)',
              'validation_month':'October','test_months':['November','December'],
              'thresholds':thresholds,'duplicate_rows_retained':duplicates,
              'quick_test_only':quick,'python':platform.python_version(),'sklearn':sklearn.__version__,
              'numpy':np.__version__,'pandas':pd.__version__}
    (out/'locked_decisions.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]; predictions=[]
    for (family,name),pipe in candidates.items():
        cols=FEATURES[family]
        for period,part in [('Nov',test[test._month==11]),('Dec',test[test._month==12]),('pooled',test)]:
            if part.empty: continue
            p=pipe.predict_proba(part[cols])[:,1]
            rows.append({'family':family,'model':name,'period':period,**metrics(part.Revenue,p)})
            if period=='pooled':
                predictions.append(pd.DataFrame({'row_id':part._row_id,'Month':part.Month,
                    'VisitorType':part.VisitorType,'y':part.Revenue,'p':p,'family':family,'model':name}))
    results=pd.DataFrame(rows); results.to_csv(out/'test_models.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True); pred.to_csv(out/'test_predictions.csv',index=False)
    policies=[]; intervals=[]; segments=[]; baseline=[]
    for period,part in [('Oct_validation',val),('Nov',test[test._month==11]),('Dec',test[test._month==12])]:
        if part.empty: continue
        y=part.Revenue.to_numpy(); p=primary.predict_proba(part[CONTEXT])[:,1]
        constant=np.repeat(train.Revenue.mean(),len(y))
        baseline.append({'period':period,**metrics(y,constant)})
        for b in budgets:
            threshold=thresholds[str(b)]
            for policy,mask in [('fixed_october_cutoff',p>=threshold),('batch_top_budget',top_mask(p,b))]:
                policies.append({'period':period,'fraction':b,'policy':policy,**policy_metrics(y,p,mask,b)})
                for visitor in sorted(part.VisitorType.unique()):
                    g=part.VisitorType.to_numpy()==visitor; selected=mask & g
                    segments.append({'period':period,'fraction':b,'policy':policy,'visitor_type':visitor,
                       'n':int(g.sum()),'selected':int(selected.sum()),
                       'selection_rate':float(selected.sum()/g.sum()),
                       'selected_precision':float(y[selected].mean()) if selected.any() else np.nan,
                       'purchase_rate':float(y[g].mean()),'small_group':bool(g.sum()<100)})
            if period!='Oct_validation':
                intervals.append({'period':period,'fraction':b,**paired_bootstrap(y,p,threshold,b,30 if quick else 500)})
    policy_df=pd.DataFrame(policies); policy_df.to_csv(out/'policy_comparison.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'policy_precision_difference_intervals.csv',index=False)
    pd.DataFrame(segments).to_csv(out/'visitor_segment_audit.csv',index=False)
    pd.DataFrame(baseline).to_csv(out/'constant_baseline.csv',index=False)
    summary=df.groupby(['_month','Month']).Revenue.agg(['size','mean']).reset_index()
    summary.to_csv(out/'monthly_purchase_rates.csv',index=False)
    pd.DataFrame([{'split':name,'n':len(g),'purchases':int(g.Revenue.sum())} for name,g in split.items()]).to_csv(out/'split_audit.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(summary.Month,summary['mean'],color='#167d9a')
    ax.set(ylabel='Observed purchase rate',title='Purchase rate across available months')
    fig.tight_layout(); fig.savefig(out/'01_monthly_purchase_rates.png',dpi=180); plt.close(fig)
    ten=policy_df[np.isclose(policy_df.fraction,.1)]
    chart=ten.pivot(index='period',columns='policy',values='selected_rate').reindex(['Oct_validation','Nov','Dec'])
    chart=chart.rename(index={'Oct_validation':'October (validation)','Nov':'November','Dec':'December'},
                       columns={'batch_top_budget':'Top 10% each month','fixed_october_cutoff':'Fixed October cutoff'})
    ax=chart.plot.bar(figsize=(8,4),color=['#167d9a','#e89438'])
    ax.axhline(.1,color='black',linestyle='--',label='10% capacity')
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis='x',labelrotation=0)
    ax.set(ylabel='Share selected',xlabel='',title='Does the October cutoff preserve capacity?')
    ax.legend(fontsize=8); ax.figure.tight_layout(); ax.figure.savefig(out/'02_capacity_drift.png',dpi=180); plt.close(ax.figure)
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],'k--',label='Perfect calibration')
    for label,part in [('October',val),('November',test[test._month==11]),('December',test[test._month==12])]:
        if part.empty: continue
        p=primary.predict_proba(part[CONTEXT])[:,1]
        observed,predicted=calibration_curve(part.Revenue,p,n_bins=8,strategy='quantile')
        ax.plot(predicted,observed,'o-',label=label)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed purchase rate',title='Probability reliability by month')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'03_reliability.png',dpi=180); plt.close(fig)
    note='''# Results interpretation checklist
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
'''
    (out/'INTERPRET_RESULTS.md').write_text(note,encoding='utf-8')
    print('Selected arrival-context model:',chosen)
    print('Outputs:',out.resolve())
    return validation, results, policy_df

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default=None)
    parser.add_argument('--output',default=None,
                        help="Default: 'results', or 'sensitivity_deduplicated' with --dedup")
    parser.add_argument('--dedup',action='store_true',
                        help='Pre-specified sensitivity: drop exact duplicate rows before the identical pipeline')
    args=parser.parse_args()
    data,sha=load_data(args.csv)
    if args.dedup:
        before=len(data)
        data=data.drop_duplicates().reset_index(drop=True)
        # Hash the transformed table so the manifest never claims to be the official file.
        sha='deduplicated:'+hashlib.sha256(data.to_csv(index=False).encode('utf-8')).hexdigest()
        print(f'Sensitivity run: removed {before-len(data)} exact duplicate rows (transformed data).')
    run_experiment(data,args.output or ('sensitivity_deduplicated' if args.dedup else 'results'),sha)
