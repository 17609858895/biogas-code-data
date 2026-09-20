"""R2-7: date-cluster percentile intervals conditional on the existing fitted analysis.

Run with the original ML environment. No source workbook or primary predictions
are changed. Scenario refits reproduce the existing deterministic configurations.
"""
from pathlib import Path
import argparse, importlib.util, sys, json, hashlib, os
os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'results'
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=DATA/'uncertainty')
OUT=parser.parse_args().output
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('analysis',ROOT/'code/analysis_functions.py')
a=importlib.util.module_from_spec(spec);sys.modules['analysis']=a;spec.loader.exec_module(a)
e=pd.read_csv(DATA/'data/analysis_records.csv',parse_dates=['date'])
dev=e[e.split.isin(['train','validation'])].copy()
test=e[e.split=='test'].copy()
pred=pd.read_csv(DATA/'audit/test_predictions.csv',parse_dates=['date'])
scale=a.training_mase_scale(dev)
dates=np.array(sorted(pred.date.unique()))
assert len(dates)==23 and len(pred)==86
draws=np.random.default_rng(42).choice(len(dates),size=(1000,len(dates)),replace=True)
counts=np.stack([np.bincount(d,minlength=len(dates)) for d in draws])
pd.DataFrame(draws,columns=[f'draw_{i+1}' for i in range(len(dates))]).to_csv(OUT/'bootstrap_date_indices.csv',index=False)
rows=[];repframes=[]

def metrics(frame):
    """Apply the common date multiplicities to each retained subgroup."""
    y=frame.observed_mL_d.to_numpy();p=frame.predicted_mL_d.to_numpy()
    idx=pd.Index(dates).get_indexer(frame.date)
    assert (idx>=0).all()
    w=np.vstack([np.ones(len(frame)),counts[:,idx]])
    n=w.sum(axis=1);error=p-y
    with np.errstate(divide='ignore',invalid='ignore'):
        mse=w@(error**2)/n;mae=w@np.abs(error)/n
        var=w@(y*y)-(w@y)**2/n
        result={'R2':np.where(var>1e-12,1-(w@(error**2))/var,np.nan),
                'RMSE':np.sqrt(mse),'MAE':mae,'MASE':mae/scale,
                'Bias':w@error/n,'MAPE':w@(np.abs(error)/np.maximum(np.abs(y),1e-6))*100/n}
        if 'covered90' in frame:
            for level in [90,95]:result[f'Coverage{level}']=w@frame[f'covered{level}'].astype(float).to_numpy()/n
    return result

def record(domain,name,frame,result,selected):
    for metric in selected:
        values=result[metric];valid=values[1:][np.isfinite(values[1:])]
        lo,hi=np.quantile(valid,[.025,.975])
        rows.append(dict(domain=domain,group=name,metric=metric,estimate=float(values[0]),
                         ci95_low=float(lo),ci95_high=float(hi),valid_replicates=len(valid),
                         n_records=len(frame),n_dates=frame.date.nunique()))
    repframes.append(pd.DataFrame({'domain':domain,'group':name,'replicate':np.arange(1000),
                                   **{k:result[k][1:] for k in selected}}))

core=['R2','RMSE','MAE','MASE']
for group,frame in [('Overall',pred)]+list(pred.groupby('reactor_id')):
    result=metrics(frame)
    record('reactor',group,frame,result,core+['Coverage90','Coverage95'])

threshold=float(dev[a.TARGET].quantile(.9))
for group,frame in [('All records',pred),('Peak records',pred[pred.observed_mL_d>=threshold]),
                    ('1-day history',pred[pred.days_since_prev<=1]),('Longer gaps',pred[pred.days_since_prev>1])]:
    record('diagnostic',group,frame,metrics(frame),['RMSE','MAE','Bias'])

models=pd.read_csv(DATA/'audit/all_model_evaluation_predictions.csv',parse_dates=['date'])
frames={name:g.copy() for name,g in models.groupby('model')}
for name,column in [('Persistence','biogas_lag1'),('Rolling-3','biogas_roll3'),('Rolling-7','biogas_roll7')]:
    fr=test[['date','reactor_id',a.TARGET,column]].rename(columns={a.TARGET:'observed_mL_d',column:'predicted_mL_d'})
    frames[name]=fr
allmetrics={name:metrics(fr) for name,fr in frames.items()}
mlnames=sorted(models.model.unique())
criteria=['R2','RMSE','MAE','MAPE']
weightframe=pd.read_csv(DATA/'tables/TOPSIS_weights.csv')
weights=weightframe.entropy_weight.to_numpy()
assert weightframe.criterion.tolist()==['R2','RMSE_mL','MAE_mL','MAPE_percent']
# Entropy weights define the published descriptive score and are held fixed.
# Norms and ideal/nadir values are recomputed for each replicate across all models.
x=np.stack([np.stack([allmetrics[n][c] for c in criteria],axis=-1) for n in mlnames],axis=1)
v=x/np.sqrt((x*x).sum(axis=1,keepdims=True))*weights
ideal=v.min(axis=1);ideal[:,0]=v[:,:,0].max(axis=1)
nadir=v.max(axis=1);nadir[:,0]=v[:,:,0].min(axis=1)
dp=np.sqrt(((v-ideal[:,None,:])**2).sum(axis=2));dn=np.sqrt(((v-nadir[:,None,:])**2).sum(axis=2))
topsis=dn/(dp+dn+1e-12)
for i,n in enumerate(mlnames):allmetrics[n]['TOPSIS']=topsis[:,i]
reference=pd.read_csv(DATA/'tables/Table_1_all_models_evaluation.csv')
for name,fr in frames.items():
    result=allmetrics[name]
    result['Skill_P']=1-(result['RMSE']/allmetrics['Persistence']['RMSE'])**2
    result['Skill_R7']=1-(result['RMSE']/allmetrics['Rolling-7']['RMSE'])**2
    old=reference[reference.model==name].iloc[0]
    for key,col in [('R2','R2'),('RMSE','RMSE_mL_d'),('MAE','MAE_mL_d'),('MASE','MASE'),('Skill_P','Skill_P'),('Skill_R7','Skill_R7')]:
        np.testing.assert_allclose(result[key][0],old[col],rtol=1e-10,atol=1e-10)
    if name in mlnames:np.testing.assert_allclose(result['TOPSIS'][0],old.TOPSIS,rtol=1e-10)
    record('comparison',name,fr,result,core+['Skill_P','Skill_R7']+(['TOPSIS','MAPE'] if name in mlnames else []))

protocol=json.loads((DATA/'audit/protocol.json').read_text(encoding='utf-8'))
features=protocol['features'];cat=['reactor_id','reactor_type'];num=[c for c in features if c not in cat]
short=[c for c in ['biogas_lag1','biogas_roll3','days_since_prev','manure_fed_kg','water_kg'] if c in features]
no_weather=[c for c in features if not c.startswith('daily_') and c!='air_temp_in_situ_lag1']
q1,q3=e[a.TARGET].quantile([.25,.75]);upper=q3+3*(q3-q1)
scenarios={'Primary':(dev,test,features),'No weather':(dev,test,no_weather),
           'Parsimonious history + operation':(dev,test,short+cat),
           'Consecutive only':(dev[dev.days_since_prev<=1],test[test.days_since_prev<=1],features),
           'Exclude R4':(dev[dev.reactor_id!='R4-FIXED DOME'],test[test.reactor_id!='R4-FIXED DOME'],features),
           '3xIQR sensitivity':(dev[dev[a.TARGET]<=upper],test[test[a.TARGET]<=upper],features)}
oldsc=pd.read_csv(DATA/'tables/Table_S11_sensitivity.csv').rename(columns={'RMSE_mL_d':'RMSE','MAE_mL_d':'MAE'})
scpred=[]
for name,(fit,ev,cols) in scenarios.items():
    model=a.model_library()['RandomForest'].set_params(n_jobs=2)
    pipe=a.make_pipeline(clone(model),[c for c in cols if c not in cat],cat)
    pipe.fit(fit[cols],fit[a.TARGET])
    fr=ev[['date','reactor_id',a.TARGET]].rename(columns={a.TARGET:'observed_mL_d'}).copy()
    fr['predicted_mL_d']=pipe.predict(ev[cols]);fr['scenario']=name
    scpred.append(fr)
    result=metrics(fr);old=oldsc[oldsc.scenario==name].iloc[0]
    for key,col in [('R2','R2'),('RMSE','RMSE'),('MAE','MAE'),('MASE','MASE')]:
        np.testing.assert_allclose(result[key][0],old[col],atol=0.000051,rtol=0)
    record('scenario',name,fr,result,core)
    print(name,'matches existing results',flush=True)

summary=pd.DataFrame(rows)
# Check all four original pooled intervals remain identical at archived precision.
oldci=pd.read_csv(DATA/'tables/Table_S7_test_metric_bootstrap_CI.csv').rename(columns={'ci95_low':'ci95 low','ci95_high':'ci95 high'})
for (_,old),key in zip(oldci.iterrows(),core):
    got=summary[(summary.domain=='reactor')&(summary.group=='Overall')&(summary.metric==key)].iloc[0]
    np.testing.assert_allclose([got.estimate,got.ci95_low,got.ci95_high],[old['estimate'],old['ci95 low'],old['ci95 high']],atol=.000051,rtol=0)
summary.to_csv(OUT/'metric_intervals.csv',index=False,encoding='utf-8-sig')
pd.concat(repframes,ignore_index=True).to_csv(OUT/'bootstrap_replicates.csv.gz',index=False,compression='gzip')
pd.concat(scpred,ignore_index=True).to_csv(OUT/'scenario_predictions.csv',index=False,encoding='utf-8-sig')
inputs=[DATA/'audit/test_predictions.csv',DATA/'audit/all_model_evaluation_predictions.csv',DATA/'data/analysis_records.csv',DATA/'audit/protocol.json',DATA/'tables/TOPSIS_weights.csv']
manifest={'seed':42,'replicates':1000,'evaluation_dates':[str(pd.Timestamp(d).date()) for d in dates],
          'resampling':'23 dates with replacement; shared date multiplicities across all models, reactors and scenarios; retain all records in each selected date and subset',
          'percentiles':[2.5,97.5],'MASE_scale_mL_d':scale,'peak_threshold_mL_d':threshold,
          'fixed':['fitted models','data preparation','development MASE scale','calibration band widths','TOPSIS entropy weights','subset definitions'],
          'TOPSIS':'Recompute column norms and ideal/nadir for each replicate; keep published entropy weights fixed.',
          'boundary_coverage':'A group with no observed misses has a degenerate [1,1] nonparametric bootstrap interval; this is not a coverage guarantee.',
          'widths':'914.904519713331 and 1390.9223494257671 mL/day are fixed calibrated band widths, not independently sampled performance metrics.',
          'excluded_uncertainty':['consecutive-day dependence','data preparation','parameter estimation and model selection','calibration-quantile estimation'],
          'checks':{'all_existing_point_estimates_match':True,'original_pooled_intervals_match':True,'minimum_valid_replicates':int(summary.valid_replicates.min())},
          'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
(OUT/'protocol.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary[summary.domain=='reactor'].to_string(index=False))
