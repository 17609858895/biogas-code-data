"""Export full-precision CSVs using the final manuscript table numbering."""
from pathlib import Path
import json
import pandas as pd

root=Path(__file__).resolve().parent.parent/'results'
tables=root/'tables';out=root/'publication_tables';out.mkdir(exist_ok=True)
def save(df,name):df.to_csv(out/name,index=False,encoding='utf-8-sig')
save(pd.read_csv(tables/'Table_1_all_models_evaluation.csv'),'Table_1.csv')
split=pd.read_csv(tables/'Table_S1_locked_split_summary.csv')
save(split.set_index('split').loc[['train','validation','calibration','test']].reset_index(),'Table_S1.csv')
cv=pd.read_csv(tables/'Table_S3_training_rolling_origin_metrics.csv')
cols=['R2','RMSE_mL_d','MAE_mL_d','MASE']
stats=cv.groupby('model')[cols].agg(['mean','std'])
stats.columns=['CV_'+a+'_'+b for a,b in stats.columns]
val=pd.read_csv(tables/'Table_1_validation_model_selection.csv')
val=val.rename(columns={c:'Validation_'+c for c in cols})
save(val.merge(stats,on='model').sort_values('Validation_RMSE_mL_d'),'Table_S2.csv')
save(pd.read_csv(tables/'Table_S4_Holm_pairwise_validation.csv'),'Table_S3.csv')
intervals=pd.read_csv(root/'uncertainty/metric_intervals.csv')
protocol=json.loads((root/'uncertainty/protocol.json').read_text('utf-8'))
scenario_n=pd.read_csv(tables/'Table_S11_sensitivity.csv').set_index('scenario')
for number,domain in [(4,'reactor'),(5,'comparison'),(6,'diagnostic'),(7,'scenario')]:
    rows=[]
    for group,g in intervals[intervals.domain==domain].groupby('group',sort=False):
        row={'group':group,'n_records':int(g.iloc[0].n_records),'n_dates':int(g.iloc[0].n_dates)}
        for r in g.itertuples():
            row.update({r.metric:r.estimate,r.metric+'_ci95_low':r.ci95_low,r.metric+'_ci95_high':r.ci95_high})
        if domain=='reactor':
            fit=pd.read_csv(tables/'Table_2_final_test_metrics.csv').iloc[0]
            row.update(fixed_width90_mL_d=2*fit.q90_mL_d,fixed_width95_mL_d=2*fit.q95_mL_d)
        if domain=='scenario':row['fit_n']=int(scenario_n.loc[group,'train_n'])
        rows.append(row)
    save(pd.DataFrame(rows),f'Table_S{number}.csv')
print('Final Table 1 and Tables S1-S7: '+str(out))
