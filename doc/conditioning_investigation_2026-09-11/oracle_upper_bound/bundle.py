"""Package reports and Stage-1 supports; omit large trainable/Adam checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    paths = sorted(path for path in a.root.rglob('*') if path.is_file() and
                   path.suffix in ('.json', '.jsonl', '.npz') and '.partial.' not in path.name)
    output = a.root / 'oracle_upper_bound_bundle.zip'
    temporary = output.with_suffix('.tmp.zip')
    manifest = {}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            relative = str(path.relative_to(a.root))
            content = path.read_bytes()
            manifest[relative] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
            z.writestr(relative, content)
        z.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    temporary.replace(output)
    print(output)


if __name__ == '__main__':
    main()
