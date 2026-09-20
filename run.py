"""Reproduce the final manuscript analysis from the supplied data.

Validated with Python 3.9.21. Install: python -m pip install -r requirements.txt
Run everything: python run.py
Numerical analysis only: python run.py --no-figures
After analysis, re-render figures only: python run.py --figures-only

Inputs: data/Farm-scale_Biodigester.xlsx (supplied workbook, byte-for-byte)
and data/weather_daily.csv (the daily meteorological inputs used by the model).
All outputs are generated under results/; no model cache or manuscript is needed.
Nine fixed model configurations are defined in code/analysis_functions.py.
The lowest validation RMSE selects RF; evaluation rankings are descriptive.
The final analysis includes SHAP, permutation importance, ALE, baseline/TOPSIS
comparison and 1000 date-cluster bootstrap replicates (including reactor metrics,
coverage and sensitivity scenarios). Calibration records are excluded from fits.

This is a retrospective analysis of processed records. Preparation involved the
existing evaluation window, so the evaluation is not independent validation.
Bootstrap intervals condition on prepared data, models and calibration bands;
they preserve same-date clustering, not dependence between consecutive dates.
"""
from pathlib import Path
import argparse,os,subprocess,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--no-figures',action='store_true')
    group.add_argument('--figures-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    for n in ['Farm-scale_Biodigester.xlsx','weather_daily.csv']:
        if not (root/'data'/n).is_file():parser.error('Missing data/'+n)
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    steps=[] if args.figures_only else ['run_reanalysis.py','data_all_models.py','uncertainty_analysis.py','export_tables.py']
    if not args.no_figures:steps+=['refine_figures.py','ale_analysis.py']
    for step in steps:
        print('Running '+step,flush=True)
        subprocess.run([sys.executable,str(root/'code'/step)],cwd=root,env=env,check=True)
    print('Completed. Outputs: '+str(root/'results'))

if __name__=='__main__':main()
