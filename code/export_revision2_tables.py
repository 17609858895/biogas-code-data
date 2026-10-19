"""Export added publication tables from explicit curated or computed sources."""
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / 'results/revision2'
OUT = ROOT / 'results/publication_tables'
OUT.mkdir(parents=True,exist_ok=True)

for source_name, output_name in [('Table_2_literature_comparison.csv','Table_2.csv'),
                                  ('Table_S8_issue_time_availability.csv','Table_S8.csv')]:
    source = ROOT / 'data' / source_name
    if not source.is_file():
        raise FileNotFoundError('Missing explicitly curated source table: '+str(source))
    shutil.copy2(source,OUT/output_name)

folds = pd.read_csv(REV/'nested_fold_windows.csv')
outer = pd.read_csv(REV/'nested_outer_metrics.csv')
selected = outer[outer.selected_by_inner].merge(folds,on='outer_fold',validate='one_to_one')
selected = selected[['outer_fold','train_start','train_end','test_start','test_end',
                      'train_n','test_n','model','RMSE_mL_d','MAE_mL_d','R2']]
selected['model'] = selected['model'].replace({'RandomForest':'RF'})
selected.to_csv(OUT/'Table_S9.csv',index=False,encoding='utf-8-sig')
pooled = pd.read_csv(REV/'nested_pooled_summary.csv')
pooled['model'] = pooled['model'].replace({'RandomForest':'RF','Nested selected procedure':'Nested selection'})
pooled.to_csv(OUT/'Table_S10.csv',index=False,encoding='utf-8-sig')
ale = pd.read_csv(REV/'ALE_summary.csv')
ale = ale.loc[ale.bins==10,['version','feature','n','amplitude_mL_d']]
ale.to_csv(OUT/'Table_S11.csv',index=False,encoding='utf-8-sig')
print('Exported Table 2 and Tables S8-S11; Table 2/S8 are curated source tables, S9-S11 are computational outputs.')
