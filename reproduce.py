"""Reproduce the author-confirmed corrected workbook analysis."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent

def preflight():
    rows = []
    for line in (ROOT / 'requirements.lock.txt').read_text('utf8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, required = line.strip().split('==')
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        rows.append({'package':name, 'required':required, 'installed':installed,
                     'matches':installed == required})
    expected = json.loads((ROOT / 'evidence/snapshot_manifest.json').read_text('utf8'))
    hashes = []
    for item in expected['files']:
        if item['path'].startswith('data/') or item['path'].startswith('code/'):
            p = ROOT / item['path']
            actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            hashes.append({'path':item['path'], 'matches':actual == item['sha256'],
                           'expected_sha256':item['sha256'], 'actual_sha256':actual})
    source = ROOT / 'data/Farm-scale_Biodigester.xlsx'
    report = {'time_utc':datetime.now(timezone.utc).isoformat(),
              'python':sys.version, 'platform':platform.platform(),
              'python_matches':platform.python_version() == '3.9.21',
              'requirements':rows, 'input_code_hashes':hashes,
              'current_workbook_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'provenance_status':'author-confirmed corrected derivative, 2026-10-19',
              'correction_rationale':'Author confirms original records contained errors and directs analysis of the corrected workbook.',
              'independent_field_correction_logs_available':False,
              'technical_preflight_pass':all(r['matches'] for r in rows+hashes)
                                         and platform.python_version() == '3.9.21'}
    (ROOT / 'evidence/preflight.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--no-figures', action='store_true')
    p.add_argument('--figures-only', action='store_true')
    args = p.parse_args()
    report = preflight()
    print(json.dumps(report, indent=2), flush=True)
    if not report['technical_preflight_pass']:
        raise SystemExit('Environment or snapshot mismatch; inspect evidence/preflight.json.')
    if args.preflight_only:
        return
    command = [sys.executable, str(ROOT / 'run_baseline.py')]
    if args.no_figures:
        command.append('--no-figures')
    if args.figures_only:
        command.append('--figures-only')
    run = {'started_utc':datetime.now(timezone.utc).isoformat(), 'command':command,
           'author_confirmed_corrected_data':True,
           'interpretation':'technical reproduction of the fixed author-confirmed corrected derivative; corrections are author-attested, without independent field correction logs'}
    (ROOT / 'evidence/run_status.json').write_text(json.dumps(run, indent=2), encoding='utf8')
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        if args.figures_only:
            extra_steps = ['render_revision2_ale.py']
        else:
            extra_steps = ['revision2_selection.py','verify_revision2.py','export_revision2_tables.py']
        for step in extra_steps:
            print('Running '+step,flush=True)
            subprocess.run([sys.executable,str(ROOT/'code'/step)],cwd=ROOT,check=True)
        if not args.no_figures:
            subprocess.run([sys.executable,str(ROOT/'verify_outputs.py')],cwd=ROOT,check=True)
        run['exit_code'] = 0
    except subprocess.CalledProcessError as exc:
        run['exit_code'] = exc.returncode
        raise
    finally:
        run['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (ROOT / 'evidence/run_status.json').write_text(json.dumps(run, indent=2), encoding='utf8')

if __name__ == '__main__':
    main()
