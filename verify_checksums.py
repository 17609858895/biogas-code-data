"""Write or verify SHA-256 manifests for the package snapshot."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    p.add_argument('--generated', action='store_true', help='Verify generated result-file bytes as well.')
    args = p.parse_args()
    if args.write:
        static = []
        generated = []
        snapshot = []
        for f in sorted(ROOT.rglob('*')):
            if not f.is_file() or '__pycache__' in f.parts:
                continue
            rel = f.relative_to(ROOT)
            if rel.parts[0] in ['code','data']:
                snapshot.append({'path':rel.as_posix(),'size':f.stat().st_size,'sha256':sha(f)})
            if rel.parts[0] == 'results':
                generated.append(f)
            elif rel.parts[0] != 'evidence' and f.name not in ['checksums.sha256','generated_checksums.sha256']:
                static.append(f)
        for filename, files in [('checksums.sha256',static),('generated_checksums.sha256',generated)]:
            (ROOT / filename).write_text(''.join(sha(f)+'  '+f.relative_to(ROOT).as_posix()+'\n' for f in files),encoding='utf8')
        (ROOT/'evidence/snapshot_manifest.json').write_text(json.dumps({'scope':'final corrected inputs and analysis code',
             'files':snapshot},indent=2),encoding='utf8')
    checked = []
    for filename in ['checksums.sha256']+(['generated_checksums.sha256'] if args.generated else []):
        for line in (ROOT / filename).read_text('utf8').splitlines():
            expected, name = line.split('  ', 1)
            f = ROOT / name
            checked.append({'path':name, 'matches':f.is_file() and sha(f)==expected})
    report = {'checked_files':len(checked), 'mismatches':[x['path'] for x in checked if not x['matches']],
              'pass':all(x['matches'] for x in checked)}
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
