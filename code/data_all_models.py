"""Fit all nine fixed models on the same development data and evaluate identical rows."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import importlib.util,sys,json,joblib
import pandas as pd,numpy as np
from sklearn.base import clone
OUT=Path(__file__).resolve().parent.parent;DATA=OUT/'results'
sp=importlib.util.spec_from_file_location('plots',OUT/'code/analysis_functions.py');a=importlib.util.module_from_spec(sp);sys.modules['plots']=a;sp.loader.exec_module(a)
e=pd.read_csv(DATA/'data/analysis_records.csv',parse_dates=['date'])
protocol=json.loads((DATA/'audit/protocol.json').read_text('utf8'));features=protocol['features'];cat=['reactor_id','reactor_type'];num=[c for c in features if c not in cat]
dev=e[e.split.isin(['train','validation'])].copy();test=e[e.split=='test'].copy()
assert len(dev)==424 and len(test)==86
scale=a.training_mase_scale(dev);rows=[];predictions=[]
for name,model in a.model_library().items():
    p=model.get_params()
    if 'n_jobs' in p:model.set_params(n_jobs=2)
    if 'thread_count' in p:model.set_params(thread_count=2)
    pipe=a.make_pipeline(clone(model),num,cat);pipe.fit(dev[features],dev[a.TARGET]);pred=pipe.predict(test[features])
    metric=a.regression_metrics(test[a.TARGET].to_numpy(),pred,scale)
    y=test[a.TARGET].to_numpy();denom=np.where(np.abs(y)<1e-6,np.nan,np.abs(y))
    metric['MAPE_percent']=float(np.nanmean(np.abs((y-pred)/denom))*100)
    rows.append({'type':'ML','model':name,'n_records':len(test),**metric})
    frame=test[['date','reactor_id',a.TARGET]].rename(columns={a.TARGET:'observed_mL_d'}).copy();frame['model']=name;frame['predicted_mL_d']=pred;predictions.append(frame)
    joblib.dump(pipe,DATA/'models'/f'{name}_development_fit.joblib')
    print(name,metric,flush=True)
for name,column in [('Persistence','biogas_lag1'),('Rolling-3','biogas_roll3'),('Rolling-7','biogas_roll7')]:
    assert test[column].notna().all()
    rows.append({'type':'Baseline','model':name,'n_records':len(test),**a.regression_metrics(test[a.TARGET].to_numpy(),test[column].to_numpy(),scale)})
result=pd.DataFrame(rows);p=result.loc[result.model=='Persistence','RMSE_mL'].iloc[0];r7=result.loc[result.model=='Rolling-7','RMSE_mL'].iloc[0]
# Baseline skill is relative mean-squared-error reduction.
result['Skill_P']=1-(result.RMSE_mL/p)**2
result['Skill_R7']=1-(result.RMSE_mL/r7)**2
ml=result[result.type=='ML'];criteria=['R2','RMSE_mL','MAE_mL','MAPE_percent']
x=ml[criteria].astype(float);norm=x/np.sqrt((x*x).sum(axis=0));p=norm/(norm.sum(axis=0)+1e-12)
assert (p>=0).all().all()
entropy=-(p*np.log(p+1e-12)).sum(axis=0)/np.log(len(ml)+1e-12)
weights=(1-entropy)/(1-entropy).sum();v=norm.mul(weights,axis=1)
ideal=pd.Series({c:v[c].max() if c=='R2' else v[c].min() for c in criteria})
nadir=pd.Series({c:v[c].min() if c=='R2' else v[c].max() for c in criteria})
dpos=np.sqrt(((v-ideal)**2).sum(axis=1));dneg=np.sqrt(((v-nadir)**2).sum(axis=1))
result.loc[ml.index,'TOPSIS']=dneg/(dpos+dneg+1e-12)
pd.DataFrame({'criterion':criteria,'entropy_weight':weights.values}).to_csv(DATA/'tables/TOPSIS_weights.csv',index=False,encoding='utf-8-sig')
result=result.sort_values(['RMSE_mL','MAE_mL','model']).reset_index(drop=True);result.insert(0,'rank',np.arange(1,len(result)+1))
a.unit_safe_columns(result).to_csv(DATA/'tables/Table_1_all_models_evaluation.csv',index=False,encoding='utf-8-sig')
pd.concat(predictions,ignore_index=True).to_csv(DATA/'audit/all_model_evaluation_predictions.csv',index=False,encoding='utf-8-sig')
assert len(result)==12
old=pd.read_csv(DATA/'audit/test_predictions.csv');new=predictions[[x.model.iloc[0] for x in predictions].index(protocol['reference_model'])]
np.testing.assert_allclose(old.predicted_mL_d,new.predicted_mL_d,rtol=1e-10,atol=1e-10)
(DATA/'audit/all_model_protocol.json').write_text(json.dumps({'development_records':424,'calibration_records_excluded_from_fitting':36,'common_evaluation_records':86,'ranking':'descriptive ascending evaluation RMSE, not used to select or tune a model','skill_definition':'1 - model MSE / baseline MSE','topsis':{'criteria':criteria,'NSE_excluded':True,'normalization':'column Euclidean norm','weighting':'Shannon entropy across the nine ML models','direction':'maximize R2, minimize RMSE MAE MAPE','model_selection_use':False},'validation_results_kept_separate':True},indent=2),encoding='utf8')
print(result.to_string(index=False))
