"""Independently recompute outer predictions and reconcile table roles; CHONG LIU."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
OUT=Path(__file__).resolve().parent.parent/'results/revision2'
pred=pd.read_csv(OUT/'nested_outer_predictions.csv')
outer=pd.read_csv(OUT/'nested_outer_metrics.csv')
choice=pd.read_csv(OUT/'nested_inner_selection.csv')
summary=pd.read_csv(OUT/'nested_pooled_summary.csv')
report={'author':'CHONG LIU','all_outer_model_cells_recalculated':0,'selected_outer':[]}
for (fold,model),g in pred.groupby(['outer_fold','model']):
    row=outer[(outer.outer_fold==fold)&(outer.model==model)].iloc[0]
    calculated={'RMSE_mL_d':np.sqrt(mean_squared_error(g.observed_mL_d,g.predicted_mL_d)),'MAE_mL_d':mean_absolute_error(g.observed_mL_d,g.predicted_mL_d),'R2':r2_score(g.observed_mL_d,g.predicted_mL_d)}
    for key,value in calculated.items():assert np.isclose(value,row[key],rtol=1e-12,atol=1e-12)
    selected=choice[(choice.outer_fold==fold)&choice.selected].iloc[0]['model']
    assert bool(row.selected_by_inner)==(model==selected)
    report['all_outer_model_cells_recalculated']+=1
    if model==selected:report['selected_outer'].append({'outer_fold':int(fold),'model':model,'n':len(g),**calculated})
g=pred[pred.selected_by_inner]
calculated={'RMSE_mL_d':np.sqrt(mean_squared_error(g.observed_mL_d,g.predicted_mL_d)),'MAE_mL_d':mean_absolute_error(g.observed_mL_d,g.predicted_mL_d),'R2':r2_score(g.observed_mL_d,g.predicted_mL_d)}
row=summary[summary.model=='Nested selected procedure'].iloc[0]
for key,value in calculated.items():assert np.isclose(value,row[key],rtol=1e-12,atol=1e-12)
weighted=np.sqrt(sum(r['n']*r['RMSE_mL_d']**2 for r in report['selected_outer'])/len(g))
assert np.isclose(weighted,calculated['RMSE_mL_d'],rtol=1e-12)
report['selected_pooled']={'n':len(g),**calculated,'weighted_fold_RMSE':weighted}
report['clarification']='Inner-choice RMSE is the pooled inner-validation score used to choose an algorithm. Outer-selected RMSE is held-out error after refit and is the only fold quantity pooled for nested performance.'
p=OUT/'selection_results.json';d=json.loads(p.read_text('utf-8'))
d['inner_choice_scores']=choice[choice.selected].to_dict('records')
d['outer_selected']=report['selected_outer']
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'selection_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2))
