"""Conditional development-only temporal selection and ALE; CHONG LIU.

Uses unchanged author-confirmed corrected processed records (2026-10-19). This is a
retrospective diagnostic and cannot establish independent farm validation.
Run with the project's pinned Python 3.9 environment. No evaluation or
calibration row enters any estimator, selection score, or ALE reference set.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import importlib.util, sys, json, hashlib, platform, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/revision2';OUT.mkdir(parents=True,exist_ok=True)
REPO=ROOT
SOURCE=ROOT/'results/data/analysis_records.csv'
PROTOCOL=ROOT/'results/audit/protocol.json'
MODULE=REPO/'code/analysis_functions.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,frame):frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def metric(y,p):return {'RMSE_mL_d':float(np.sqrt(mean_squared_error(y,p))),'MAE_mL_d':float(mean_absolute_error(y,p)),'R2':float(r2_score(y,p))}
sp=importlib.util.spec_from_file_location('fixed_analysis_functions',MODULE)
a=importlib.util.module_from_spec(sp);sys.modules[sp.name]=a;sp.loader.exec_module(a)
original=json.loads(PROTOCOL.read_text(encoding='utf-8'))
records=pd.read_csv(SOURCE,parse_dates=['date'])
dev=records[records.split.isin(['train','validation'])].copy().sort_values(['date','reactor_id']).reset_index(drop=True)
assert len(dev)==424 and dev.date.max()==pd.Timestamp('2024-09-04')
assert not dev.split.isin(['calibration','test']).any()
features=original['features'];categorical=['reactor_id','reactor_type'];numeric=[x for x in features if x not in categorical]
models=a.model_library();assert len(models)==9
for model in models.values():
    params=model.get_params()
    if 'n_jobs' in params:model.set_params(n_jobs=2)
    if 'thread_count' in params or type(model).__name__=='CatBoostRegressor':model.set_params(thread_count=2)
dates=np.array(sorted(dev.date.unique()))
outer_splits=list(TimeSeriesSplit(n_splits=3).split(dates))
protocol={
 'author':'CHONG LIU','scope':'retrospective diagnostic on author-confirmed corrected processed records (confirmation 2026-10-19)',
 'independent_empirical_validation':False,
 'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,PROTOCOL,MODULE]},
 'data_used':'development only (original training and validation); no calibration/evaluation data in fitting, selection, preprocessing or ALE',
 'development_n':len(dev),'development_dates_n':len(dates),'development_start':str(dev.date.min().date()),'development_end':str(dev.date.max().date()),
 'outer_design':'3 expanding-window splits of unique development dates, sklearn TimeSeriesSplit(n_splits=3)',
 'inner_design':'3 expanding-window splits of unique dates inside each outer training window',
 'selection_rule':'minimum RMSE pooled across all inner validation rows; exact ties broken alphabetically by model name',
 'candidate_settings':'nine unchanged archived configurations; no hyperparameter search',
 'preprocessing':'median imputation with missing indicators, standardization, categorical mode imputation and one-hot encoding estimated separately in every training fold',
 'prediction_interpretation':'retrospective sequential covariate evaluation; lagged outcomes in a validation block may use preceding observed outcomes in that block; this is not a fixed-origin multi-step forecast',
 'outer_role':'evaluate selection procedure and compare temporal rank stability; outer scores do not select a new evaluation-window winner',
 'random_state':42,'threads_per_estimator':2,'features':features,
 'ALE_selection':'four highest positive validation-only RF permutation importances among numeric features; model fit on original training329 rows; importance computed on original validation95 rows; 30 repeats, random_state42; fixed before development refit',
 'posthoc_ALE_features':['biogas_roll7','biogas_roll3','daily_mean_air_temp_lag1','biogas_lag2'],
 'ALE_reference':'refitted RF on all424 development rows; nonmissing plotted-feature rows; disjoint empirical-quantile bins, interpolated empirical centering; curves for8/10/15bins, 10bins plotted',
 'python':sys.version,'platform':platform.platform(),
 'models':{k:v.get_params() for k,v in models.items()},
}
(OUT/'selection_protocol.json').write_text(json.dumps(protocol,indent=2,default=str),encoding='utf-8')
outer_rows=[];inner_rows=[];selected_rows=[];predictions=[];fold_rows=[]
started=time.monotonic()
for outer,(ot,ov) in enumerate(outer_splits,1):
    train=dev[dev.date.isin(dates[ot])];valid=dev[dev.date.isin(dates[ov])]
    inner_dates=np.array(sorted(train.date.unique()));pooled={k:[[],[]] for k in models}
    assert train.date.max()<valid.date.min()
    fold_rows.append({'outer_fold':outer,'train_n':len(train),'test_n':len(valid),'train_start':train.date.min(),'train_end':train.date.max(),'test_start':valid.date.min(),'test_end':valid.date.max()})
    for inner,(it,iv) in enumerate(TimeSeriesSplit(n_splits=3).split(inner_dates),1):
        tr=train[train.date.isin(inner_dates[it])];va=train[train.date.isin(inner_dates[iv])]
        assert tr.date.max()<va.date.min() and va.date.max()<valid.date.min()
        for name,estimator in models.items():
            pipe=a.make_pipeline(clone(estimator),numeric,categorical)
            pipe.fit(tr[features],tr[a.TARGET]);p=pipe.predict(va[features])
            pooled[name][0].extend(va[a.TARGET].tolist());pooled[name][1].extend(p.tolist())
            inner_rows.append({'outer_fold':outer,'inner_fold':inner,'model':name,'train_n':len(tr),'validation_n':len(va),'train_end':tr.date.max(),'validation_start':va.date.min(),'validation_end':va.date.max(),**metric(va[a.TARGET],p)})
    inner_summary=[{'model':name,**metric(y,p)} for name,(y,p) in pooled.items()]
    best=sorted(inner_summary,key=lambda r:(r['RMSE_mL_d'],r['model']))[0]['model']
    for row in inner_summary:selected_rows.append({'outer_fold':outer,'selected':row['model']==best,**row})
    for name,estimator in models.items():
        pipe=a.make_pipeline(clone(estimator),numeric,categorical)
        pipe.fit(train[features],train[a.TARGET]);p=pipe.predict(valid[features])
        outer_rows.append({'outer_fold':outer,'model':name,'selected_by_inner':name==best,**metric(valid[a.TARGET],p)})
        for i,pr in zip(valid.index,p):
            predictions.append({'outer_fold':outer,'model':name,'selected_by_inner':name==best,'date':dev.loc[i,'date'],'reactor_id':dev.loc[i,'reactor_id'],'observed_mL_d':dev.loc[i,a.TARGET],'predicted_mL_d':float(pr)})
    print('OUTER_DONE',outer,'selected',best,'elapsed_s',round(time.monotonic()-started,1),flush=True)
outer=pd.DataFrame(outer_rows);inner=pd.DataFrame(inner_rows);selection=pd.DataFrame(selected_rows);pred=pd.DataFrame(predictions)
save('nested_outer_metrics',outer);save('nested_inner_metrics',inner);save('nested_inner_selection',selection);save('nested_outer_predictions',pred);save('nested_fold_windows',pd.DataFrame(fold_rows))
summary=[]
for name,g in pred.groupby('model'):
    summary.append({'model':name,'n_outer_predictions':len(g),**metric(g.observed_mL_d,g.predicted_mL_d),'inner_selection_count':int(selection[(selection.model==name)&selection.selected].shape[0])})
g=pred[pred.selected_by_inner]
summary.append({'model':'Nested selected procedure','n_outer_predictions':len(g),**metric(g.observed_mL_d,g.predicted_mL_d),'inner_selection_count':3})
save('nested_pooled_summary',pd.DataFrame(summary))

# Predefine variables using only the original validation window.
tr=dev[dev.split=='train'];va=dev[dev.split=='validation']
rf=a.make_pipeline(clone(models['RandomForest']),numeric,categorical)
rf.fit(tr[features],tr[a.TARGET])
with warnings.catch_warnings():
    warnings.simplefilter('ignore',UserWarning)
    pi=permutation_importance(rf,va[features],va[a.TARGET],scoring='neg_root_mean_squared_error',n_repeats=30,random_state=42,n_jobs=2)
importance=pd.DataFrame({'feature':features,'validation_RMSE_increase_mL_d':pi.importances_mean,'permutation_SD_mL_d':pi.importances_std}).sort_values(['validation_RMSE_increase_mL_d','feature'],ascending=[False,True])
save('validation_only_RF_permutation_importance',importance)
chosen=importance[importance.feature.isin(numeric)].head(4).feature.tolist()
assert len(chosen)==4
rf=a.make_pipeline(clone(models['RandomForest']),numeric,categorical);rf.fit(dev[features],dev[a.TARGET])
joblib.dump(rf,OUT/'development_RF_for_ALE.joblib')

def ale(model,x,f,bins=10):
    z=x[x[f].notna()].copy();s=z[f].to_numpy(float)
    edges=np.unique(np.quantile(s,np.linspace(0,1,bins+1)))
    ids=np.clip(np.searchsorted(edges,s,side='left')-1,0,len(edges)-2)
    changes=[];counts=[]
    for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        low=z[ids==k].copy();high=low.copy();counts.append(len(low));assert len(low)>0
        low[f]=lo;high[f]=hi
        changes.append(float(np.mean(model.predict(high)-model.predict(low))))
    raw=np.r_[0,np.cumsum(changes)];curve=raw-np.mean(np.interp(s,edges,raw))
    assert sum(counts)==len(s) and abs(np.mean(np.interp(s,edges,curve)))<1e-8
    return pd.DataFrame({'feature':f,'bins_requested':bins,'x':edges,'ale_mL_d':curve}),pd.DataFrame({'feature':f,'bins_requested':bins,'lower':edges[:-1],'upper':edges[1:],'n':counts,'mean_local_difference_mL_d':changes})
class Linear:
    def predict(self,x):return 3*x.a-2*x.b+7
t=pd.DataFrame({'a':np.arange(51,dtype=float),'b':np.arange(51,dtype=float)*.8})
c,_=ale(Linear(),t,'a');assert np.allclose(c.ale_mL_d,3*(c.x-t.a.mean()))
curves=[];supports=[];ale_summary=[]
sets={'validation_predefined':chosen,'posthoc_evaluation_selected':protocol['posthoc_ALE_features']}
for version,fs in sets.items():
    for f in fs:
        for bins in [8,10,15]:
            c,n=ale(rf,dev[features],f,bins);c.insert(0,'version',version);n.insert(0,'version',version);curves.append(c);supports.append(n)
            ale_summary.append({'version':version,'feature':f,'bins':bins,'n':int(dev[f].notna().sum()),'amplitude_mL_d':float(c.ale_mL_d.max()-c.ale_mL_d.min()),'net_change_mL_d':float(c.ale_mL_d.iloc[-1]-c.ale_mL_d.iloc[0]),'min_mL_d':float(c.ale_mL_d.min()),'max_mL_d':float(c.ale_mL_d.max())})
curves=pd.concat(curves,ignore_index=True);supports=pd.concat(supports,ignore_index=True)
save('ALE_curves',curves);save('ALE_bin_support',supports);save('ALE_summary',pd.DataFrame(ale_summary))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.labelsize':13,'axes.labelweight':'bold','xtick.labelsize':11.5,'ytick.labelsize':11.5,'axes.linewidth':1.5,'xtick.major.size':6,'ytick.major.size':6,'xtick.major.width':1.4,'ytick.major.width':1.4,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})
labels={'biogas_roll7':r'7-record mean (mL d$^{-1}$)','biogas_roll3':r'3-record mean (mL d$^{-1}$)','biogas_lag2':r'Lag 2 (mL d$^{-1}$)','biogas_lag1':r'Lag 1 (mL d$^{-1}$)','daily_mean_air_temp_lag1':'Mean air temperature (°C)','daily_max_air_temp_lag1':'Maximum air temperature (°C)','daily_min_air_temp_lag1':'Minimum air temperature (°C)','air_temp_in_situ_lag1':'In situ air temperature (°C)','biogas_delta1':r'Rate difference (mL d$^{-1}$)','daily_temp_range_lag1':'Air-temperature range (°C)','daily_solar_lag1':'Solar radiation','daily_vpd_lag1':'Vapour-pressure deficit (kPa)','daily_wind_lag1':'Wind speed','daily_precip_lag1':'Precipitation','daily_atm_p_lag1':'Atmospheric pressure','days_since_prev':'History age (d)','manure_fed_kg':'Manure fed (kg)','water_kg':'Water added (kg)'}
def draw(ax,version,f,letter):
    c=curves[(curves.version==version)&(curves.feature==f)&(curves.bins_requested==10)]
    s=dev[f].dropna();ax.axhline(0,color='#2B2B2B',lw=1,ls='--',zorder=0)
    ax.plot(c.x,c.ale_mL_d,color='#8491B4',marker='o',ms=5,lw=2.2)
    ax.plot(s,np.full(len(s),.025),'|',color='#4DBBD5',ms=5,alpha=.45,transform=ax.get_xaxis_transform())
    ax.set_xlabel(labels.get(f,f),labelpad=7);ax.set_ylabel(r'ALE (mL d$^{-1}$)',labelpad=7)
    ax.spines[['top','right']].set_visible(False);ax.grid(False)
    for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=16)
    ax.margins(x=.03,y=.12)
def export(fig,folder,stem):
    fig.savefig(folder/(stem+'.png'),dpi=600,bbox_inches='tight',facecolor='white',metadata={'Author':'CHONG LIU'})
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight',facecolor='white',metadata={'Author':'CHONG LIU','Creator':'CHONG LIU'})
for version,fs in sets.items():
    stem='Fig09_ALE_validation_predefined' if version=='validation_predefined' else 'FigS1_ALE_posthoc'
    folder=OUT/'figures'/stem;sub=folder/'subfigures';sub.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,f in enumerate(fs):draw(axes.flat[i],version,f,chr(97+i))
    fig.subplots_adjust(left=.095,right=.985,bottom=.10,top=.95,hspace=.42,wspace=.32)
    export(fig,folder,stem);plt.close(fig)
    for i,f in enumerate(fs):
        fig,ax=plt.subplots(figsize=(6,4));draw(ax,version,f,chr(97+i));fig.tight_layout(pad=1.8)
        export(fig,sub,stem+'_'+chr(97+i));plt.close(fig)
    (sub/'README.md').write_text('Each panel is independently rendered on its own canvas from the same analysis table; panels are not crops. Author: CHONG LIU.\n',encoding='utf-8')

result={'scope':protocol['scope'],'chosen_ALE_features':chosen,'inner_choice_scores':selection[selection.selected].to_dict('records'),'outer_selected':outer[outer.selected_by_inner].to_dict('records'),'nested_summary':summary,'ALE_10bin_summary':[r for r in ale_summary if r['bins']==10],'elapsed_seconds':time.monotonic()-started}
(OUT/'selection_results.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
