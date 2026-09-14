"""Find moved experiment source files used by launchers and source-hash checks.

Data paths and production-source paths keep their ordinary Path behavior.
This changes file locations, never the contents or expected hashes of a report.
"""
from functools import lru_cache
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _sources():
    result = {}
    for path in SCRIPTS.rglob('*.py'):
        if path.name in result:
            raise ValueError(f'Ambiguous experiment source filename: {path.name}')
        result[path.name] = path
    return result


def source_path(base, relative):
    path = Path(base) / relative
    if path.suffix != '.py' or path.exists():
        return path
    return _sources().get(path.name, path)
