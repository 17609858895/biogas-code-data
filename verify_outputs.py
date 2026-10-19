"""Compare independently regenerated outputs with the archived result snapshot."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
REF = ROOT / 'reference_results'
OUT = ROOT / 'results'

def compare_csv(reference, actual, base=REF):
    row = {'path':reference.relative_to(base).as_posix(), 'exists':actual.is_file()}
    if not actual.is_file():
        row['pass'] = False
        return row
    x = pd.read_csv(reference)
    y = pd.read_csv(actual)
    if reference.name == 'Table_S12_frozen_permutation_importance.csv':
        row['original_row_order_matches'] = x['feature'].tolist() == y['feature'].tolist()
        row['comparison_key'] = 'feature; near-zero importance ties may change row order at machine precision'
        x = x.sort_values('feature').reset_index(drop=True)
        y = y.sort_values('feature').reset_index(drop=True)
    row.update(reference_shape=list(x.shape), regenerated_shape=list(y.shape))
    row['schema_matches'] = list(x.columns) == list(y.columns) and x.shape == y.shape
    differences = []
    max_delta = 0.0
    if row['schema_matches']:
        for col in x:
            if pd.api.types.is_numeric_dtype(x[col]) and pd.api.types.is_numeric_dtype(y[col]):
                xv = x[col].to_numpy(dtype=float)
                yv = y[col].to_numpy(dtype=float)
                same = np.isclose(xv, yv, rtol=1e-9, atol=1e-8, equal_nan=True)
                finite = np.isfinite(xv) & np.isfinite(yv)
                if finite.any():
                    max_delta = max(max_delta, float(np.max(np.abs(xv[finite]-yv[finite]))))
            else:
                same = x[col].fillna('<NA>').astype(str).to_numpy() == y[col].fillna('<NA>').astype(str).to_numpy()
            if not same.all():
                differences.append({'column':col, 'different_cells':int((~same).sum())})
    row.update(max_absolute_numeric_difference=max_delta,
               differences=differences, **{'pass':row['schema_matches'] and not differences})
    return row

def figure_map():
    mapping = []
    for n, stem in [(1,'Fig01_locked_split'),(2,'Fig02_association_analysis'),
                    (3,'Fig03_validation_selection'),(4,'Fig04_test_prediction'),
                    (5,'Fig05_residual_reliability'),(6,'Fig06_calibration_coverage'),
                    (7,'Fig07_frozen_model_importance'),(8,'Fig08_sensitivity'),
                    (9,'Fig09_ALE_reproduction/Fig09_ALE')]:
        item = {'figure':n, 'png':f'figures/{stem}.png', 'pdf':f'figures/{stem}.pdf'}
        imagepath = OUT / item['png']
        refpath = REF / item['png']
        item['png_exists'] = imagepath.is_file()
        item['pdf_exists'] = (OUT / item['pdf']).is_file()
        if imagepath.is_file():
            with Image.open(imagepath) as img:
                item['size_pixels'] = list(img.size)
                item['regenerated_png_sha256'] = hashlib.sha256(imagepath.read_bytes()).hexdigest()
                if refpath.is_file():
                    with Image.open(refpath) as ref:
                        item['same_pixel_dimensions_as_archive'] = img.size == ref.size
                        item['pixels_identical_to_archive'] = (img.size == ref.size and
                            ImageChops.difference(img.convert('RGB'),ref.convert('RGB')).getbbox() is None)
        item['pass'] = item['png_exists'] and item['pdf_exists']
        item['publication_role'] = 'archived prior Fig. 9' if n == 9 else 'main figure'
        mapping.append(item)
    for number,stem in [(9,'Fig09_ALE_validation_predefined'),('S1','FigS1_ALE_posthoc')]:
        relative = f'figures/{stem}/{stem}'
        actual = OUT/'revision2'/(relative+'.png')
        reference = ROOT/'reference_revision2'/(relative+'.png')
        item = {'figure':number,'png':'revision2/'+relative+'.png',
                'pdf':'revision2/'+relative+'.pdf','publication_role':'main figure' if number==9 else 'supplementary figure',
                'png_exists':actual.is_file(),'pdf_exists':(OUT/'revision2'/(relative+'.pdf')).is_file()}
        if actual.is_file() and reference.is_file():
            with Image.open(actual) as img, Image.open(reference) as ref:
                item['size_pixels'] = list(img.size)
                item['pixels_identical_to_archive'] = img.size==ref.size and ImageChops.difference(img.convert('RGB'),ref.convert('RGB')).getbbox() is None
        item['pass'] = item['png_exists'] and item['pdf_exists']
        mapping.append(item)
    return mapping

def main():
    csvs = [compare_csv(p, OUT / p.relative_to(REF)) for p in sorted(REF.rglob('*.csv'))]
    # Gzip headers have timestamps; compare decompressed tabular content.
    csvs += [compare_csv(p, OUT / p.relative_to(REF)) for p in sorted(REF.rglob('*.csv.gz'))]
    revref = ROOT/'reference_revision2'
    revision_csvs = [compare_csv(p,OUT/'revision2'/p.relative_to(revref),base=revref)
                     for p in sorted(revref.glob('*.csv'))]
    for row in revision_csvs:
        row['path'] = 'revision2/'+row['path']
    csvs += revision_csvs
    figures = figure_map()
    publication = []
    for filename in ['Table_1.csv','Table_2.csv']+[f'Table_S{i}.csv' for i in range(1,12)]:
        path = OUT/'publication_tables'/filename
        item = {'table':filename,'exists':path.is_file()}
        if path.is_file():
            item['shape'] = list(pd.read_csv(path).shape)
        item['pass'] = item['exists'] and item['shape'][0]>0 and item['shape'][1]>0
        publication.append(item)
    for source_name,filename in [('Table_2_literature_comparison.csv','Table_2.csv'),
                                 ('Table_S8_issue_time_availability.csv','Table_S8.csv')]:
        item = next(x for x in publication if x['table']==filename)
        item['curated_source_byte_identical'] = (ROOT/'data'/source_name).read_bytes()==(OUT/'publication_tables'/filename).read_bytes()
        item['pass'] = item['pass'] and item['curated_source_byte_identical']
    summary = json.loads((OUT / 'results.json').read_text('utf8'))['summary']
    checks = []
    for name, expected in [('R2',0.6608918004148903),('RMSE_mL',182.28939153478336),
                           ('MAE_mL',133.86511471460037),('MASE',0.8470178193356521)]:
        checks.append({'metric':name, 'submitted_full_precision':expected,
                       'regenerated':summary[name],
                       'pass':bool(np.isclose(summary[name],expected,rtol=1e-9,atol=1e-8))})
    report = {'checked_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'technical reproduction of fixed author-confirmed corrected inputs',
              'corrections_author_confirmed':'2026-10-19',
              'independent_field_correction_logs_available':False,
              'tolerance':{'rtol':1e-9,'atol':1e-8},
              'tables_and_data':csvs, 'figures':figures, 'submitted_metric_checks':checks,
              'publication_figure_pairs':10,'additional_archived_figure_pairs':1,
              'publication_tables':publication,
              'pass':all(x['pass'] for x in csvs+figures+checks+publication)}
    (ROOT / 'evidence/regeneration_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'csv_files':len(csvs), 'csv_pass':sum(x['pass'] for x in csvs),
                       'figures':len(figures),'figure_pairs_pass':sum(x['pass'] for x in figures),
                       'pixel_identical':sum(x.get('pixels_identical_to_archive',False) for x in figures),
                       'publication_tables':len(publication),'publication_tables_pass':sum(x['pass'] for x in publication),
                       'submitted_metric_checks':checks, 'pass':report['pass']},indent=2))
    if not report['pass']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
