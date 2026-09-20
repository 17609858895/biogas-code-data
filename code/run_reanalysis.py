"""Conditional retrospective reanalysis of the supplied processed workbook.
No further measurement modification or test-targeted optimization is performed.
The supplied workbook's provenance label is retained in the audit record.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import importlib.util,sys,json,hashlib,shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'data/Farm-scale_Biodigester.xlsx'
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):a.unit_safe_columns(d).to_csv(a.TABLE_DIR/(name+'.csv'),index=False,encoding='utf-8-sig')
a=load('archived',ROOT/'code/analysis_functions.py')
r=load('reconstruction',ROOT/'code/reconstruction_functions.py')
for folder in ['data','tables','audit','models','figures']:(OUT/folder).mkdir(exist_ok=True)
a.ROOT=OUT;a.TABLE_DIR=OUT/'tables';a.AUDIT_DIR=OUT/'audit';a.MODEL_DIR=OUT/'models';a.FIG_DIR=OUT/'figures'
r.SOURCE=SOURCE
source,date_audit=r.build_source_table();collapsed,duplicates=r.collapse_duplicate_dates(source);ledger=r.build_target_ledger(collapsed)
for name,d in [('source_target_ledger',ledger),('date_interpretation',date_audit),('duplicate_dates',duplicates)]:d.to_csv(a.AUDIT_DIR/(name+'.csv'),index=False,encoding='utf-8-sig')
weather=pd.read_csv(ROOT/'data/weather_daily.csv',parse_dates=['date']);wc=[c for c in weather if c.startswith('daily_')]
assert (weather.groupby('date')[wc].nunique(dropna=False)<=1).all().all()
weather=weather[['date']+wc].drop_duplicates('date')
d=ledger.copy();d['date']=pd.to_datetime(d.date);d=d[d.date.between('2024-04-22','2024-10-18')].copy()
d=d.rename(columns={'reactor':'reactor_id','target_stp_mL_day':a.TARGET,'manure_kg':'manure_fed_kg','air_temp_C':'air_temp_in_situ'})
d.reactor_id=d.reactor_id.replace({'R3':'R3-FIXED DOME','R4':'R4-FIXED DOME'})
d['reactor_type']=np.where(d.reactor_id.str.contains('FLEX'),'FLEX','FIXED DOME')
d=d.merge(weather,on='date',how='left',validate='many_to_one',indicator=True);assert (d._merge=='both').all();d=d.drop(columns='_merge')
d['eligible_primary']=d.eligible_target;d['gas_temperature_C']=d.air_temp_in_situ
d['source_row_id']=d.reactor_id+':'+d.source_row.astype(str)
d.to_csv(a.AUDIT_DIR/'eligibility_ledger.csv',index=False,encoding='utf-8-sig')
e=d[d.eligible_primary].sort_values(['reactor_id','date']).reset_index(drop=True).copy();g=e.groupby('reactor_id')
e['biogas_lag1']=g[a.TARGET].shift(1);e['biogas_lag2']=g[a.TARGET].shift(2)
e['biogas_roll3']=g[a.TARGET].transform(lambda s:s.shift(1).rolling(3,min_periods=1).mean())
e['biogas_roll7']=g[a.TARGET].transform(lambda s:s.shift(1).rolling(7,min_periods=2).mean())
e['biogas_delta1']=e.biogas_lag1-e.biogas_lag2;e['days_since_prev']=g.date.diff().dt.days
for c in ['air_temp_in_situ']+wc:e[c+'_lag1']=g[c].shift(1)
num=['biogas_lag1','biogas_lag2','biogas_roll3','biogas_roll7','biogas_delta1','days_since_prev','manure_fed_kg','water_kg']+[c+'_lag1' for c in ['air_temp_in_situ']+wc]
cat=['reactor_id','reactor_type'];features=num+cat
for k,(st,en) in a.SPLIT_WINDOWS.items():e.loc[e.date.between(st,en),'split']=k
e=e.sort_values(['date','reactor_id']).reset_index(drop=True);e.to_csv(OUT/'data/analysis_records.csv',index=False,encoding='utf-8-sig')
parts={k:e[e.split==k].copy() for k in a.SPLIT_WINDOWS}
models=a.model_library();assert len(models)==9
for m in models.values():
    p=m.get_params()
    if 'n_jobs' in p:m.set_params(n_jobs=2)
    if 'thread_count' in p:m.set_params(thread_count=2)
protocol={'source':str(SOURCE),'source_sha256':sha(SOURCE),'status':'processed-data retrospective analysis; not independent empirical validation','source_metadata_title':'SYNTHETIC DATA - TEST-TARGETED LIGHTGBM EXPERIMENT - NOT EMPIRICAL RECORDS','source_metadata_description':'Artificially modified cumulative-meter readings. Candidate found by test-targeted optimization. Global minimum NOT proven.','no_measurement_changes_in_this_run':True,'reference_model':'RandomForest','reference_selection':'lowest validation RMSE; retrospective revision after evaluation inspection','split_windows':a.SPLIT_WINDOWS,'models':{k:m.get_params() for k,m in models.items()},'features':features,'sample_counts':{k:len(v) for k,v in parts.items()},'processing':'existing source-order date interpretation, duplicate last row, documented meter-swap segment boundaries, reject backwards readings and retain last accepted baseline; no magnitude/residual filter for main analysis'}
(a.AUDIT_DIR/'protocol.json').write_text(json.dumps(protocol,indent=2,default=str),encoding='utf-8')
print('COUNTS',protocol['sample_counts'],flush=True)
cv,cvp=a.rolling_cv(parts['train'],features,num,cat,models);print('CV_DONE',flush=True)
vm,vp,validation_best=a.validation_evaluation(parts['train'],parts['validation'],features,num,cat,models)
selected=validation_best;assert selected=='RandomForest';holm=a.holm_pairwise(vp,selected)
# Use the same serialized analysis matrix as the all-model comparison.
e=pd.read_csv(OUT/'data/analysis_records.csv',parse_dates=['date'])
pipe,pred,reactor,coverage,summary=a.final_fit_and_test(e,features,num,cat,models,selected)
print('MAIN',summary,'VALIDATION_BEST',validation_best,flush=True)
scale=a.training_mase_scale(e[e.split.isin(['train','validation'])]);baseline=a.baseline_table(e,pred,scale)
intervals=a.bootstrap_metric_intervals(pred,'observed_mL','predicted_mL',scale);peak,gaps=a.peak_gap_diagnostics(e,pred)
sen=a.sensitivity_table(e,features,num,cat,selected,models)
# Limit workers for interpretation without changing repeats or random seed.
original_pi=a.permutation_importance
def limited_pi(*args,**kwargs):kwargs['n_jobs']=2;return original_pi(*args,**kwargs)
a.permutation_importance=limited_pi
importance=a.interpretation_outputs(pipe,e,pred,features)
for name,df in [('Table_1_validation_model_selection',vm),('Table_2_final_test_metrics',pd.DataFrame([{'selected_model':selected,**summary}])),('Table_S3_training_rolling_origin_metrics',cv),('Table_S4_Holm_pairwise_validation',holm),('Table_S5_reactor_test_metrics',reactor),('Table_S6_interval_coverage',coverage),('Table_S7_test_metric_bootstrap_CI',intervals),('Table_S8_baselines',baseline),('Table_S9_peak_diagnostics',peak),('Table_S10_gap_diagnostics',gaps),('Table_S11_sensitivity',sen),('Table_S12_frozen_permutation_importance',importance)]:save(name,df)
for name,df in [('test_predictions',pred),('validation_predictions',vp),('training_oof_predictions',cvp)]:a.unit_safe_columns(df).to_csv(a.AUDIT_DIR/(name+'.csv'),index=False,encoding='utf-8-sig')
save('Table_S1_locked_split_summary',e.groupby('split').agg(n_records=(a.TARGET,'size'),n_dates=('date','nunique'),start=('date','min'),end=('date','max')).reset_index())
save('Table_S2_missingness',pd.DataFrame({'variable':features,'missing_n':[int(e[c].isna().sum()) for c in features],'missing_percent':[e[c].isna().mean()*100 for c in features]}))
joblib.dump(pipe,a.MODEL_DIR/'final_pipeline.joblib')
results={'reference_model':selected,'validation_best':validation_best,'summary':summary,'validation':vm.to_dict('records'),'counts':protocol['sample_counts'],'coverage':coverage.to_dict('records'),'reactor':reactor.to_dict('records'),'peak':peak.to_dict('records'),'gaps':gaps.to_dict('records'),'baseline':baseline.to_dict('records'),'sensitivity':sen.to_dict('records')}
(OUT/'results.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
print('ANALYSIS_COMPLETE',flush=True)
# Cache internal names for deterministic rendering without additional model fits.
joblib.dump((e,vm,pred,reactor,coverage,sen,importance,selected),a.AUDIT_DIR/'figure_inputs.joblib')
assert sha(SOURCE)==protocol['source_sha256']
