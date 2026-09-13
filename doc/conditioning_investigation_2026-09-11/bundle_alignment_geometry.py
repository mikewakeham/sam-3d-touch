"""Bundle existing Stage-1 supports for the pose-versus-shape analysis.

Standard library only; no model loading, inference, or training. Copies occupancy
NPY members from existing NPZ files, omitting latent arrays to reduce transfer size.
The original files are never modified. Full report validation runs before writing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from analyze_alignment_tolerance import analyze

HERE = Path(__file__).resolve().parent
FIELDS = ('predicted_occupancy.npy', 'target_occupancy.npy')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def beneath(root, name):
    root = root.resolve()
    path = (root / name).resolve()
    path.relative_to(root)
    return path


def collect(root, camera_dir=None, include_camera=True):
    analysis = analyze(root)
    reports = {f'alignment/{i}/results.json': root/str(i)/'results.json' for i in range(4)}
    arrays, cases = {}, []
    for shard in range(4):
        report = json.loads((root/str(shard)/'results.json').read_text())
        for policy, condition, axis, degrees in report['settings']['cases']:
            for group, batch in enumerate(report['input_batches']):
                for draw in range(report['settings']['sampling_draws']):
                    name = f'{policy}_{condition}_g{group}_d{draw}.npz'
                    path = beneath(root/str(shard), name)
                    prefix = f'alignment/{shard}/{Path(name).stem}'
                    arrays[prefix] = path
                    cases.append(dict(kind='alignment', prefix=prefix, shard=shard,
                        policy=policy, condition=condition, axis=axis, degrees=degrees,
                        group=group, draw=draw, sample_ids=batch['sample_ids']))

    camera_status = 'not requested'
    if include_camera:
        if camera_dir is None:
            drop = json.loads((HERE/'visual_dropout_returned_manual/oracle/results.json').read_text())
            camera_dir = Path(drop['settings']['baseline_fit_dir']).parent/'camera'
        reference = HERE/'multiple_view_returned_46083371/camera.json'
        report_path = camera_dir/'results.json'
        if not report_path.is_file():
            camera_status = f'unavailable: no results.json at {camera_dir}; no reconstruction run requested'
        else:
            report = json.loads(report_path.read_text())
            if report != json.loads(reference.read_text()):
                raise ValueError(f'Camera report differs from the historical reference: {report_path}')
            camera_arrays, camera_cases = {}, []
            for row in report['sampled']:
                path = beneath(camera_dir, row['artifact'])
                prefix = f'camera/{Path(row["artifact"]).stem}'
                if prefix in camera_arrays:
                    raise ValueError(f'Duplicate camera artifact: {prefix}')
                camera_arrays[prefix] = path
                camera_cases.append(dict(kind='natural_camera', prefix=prefix,
                    group=row['group'], draw=row['noise_draw'], sample_ids=[row['sample_id']]))
            missing = [str(p) for p in camera_arrays.values() if not p.is_file()]
            if missing:
                camera_status = f'unavailable: {len(missing)} of {len(camera_arrays)} saved NPZs missing; whole camera comparison omitted'
            else:
                if len(camera_arrays) != 56:
                    raise ValueError('Expected 56 historical camera predictions')
                reports['camera/results.json'] = report_path
                arrays.update(camera_arrays); cases.extend(camera_cases)
                camera_status = 'complete: 56 saved natural camera predictions'

    # Preflight every required file/member before creating an output archive.
    missing = [str(p) for p in arrays.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required alignment NPZs missing:\n'+'\n'.join(missing))
    for path in arrays.values():
        with zipfile.ZipFile(path) as source:
            for field in FIELDS:
                if source.namelist().count(field) != 1:
                    raise ValueError(f'Missing/duplicate {field} in {path}')
                if source.getinfo(field).file_size > 2 * 1024 * 1024:
                    raise ValueError(f'Unexpected occupancy member size: {path}/{field}')
                # Reads check the ZIP CRC. Array dimensions/dtypes are checked
                # by the later numerical analysis, not inferred from filenames.
                if not source.read(field).startswith(b'\x93NUMPY'):
                    raise ValueError(f'Not an NPY member: {path}/{field}')
    return reports, arrays, dict(format_version=1, scope='Existing Stage-1 occupancy arrays only; '
        'all alignment conditions, views and draws. No new model execution. '
        'Source NPZ latents omitted; NPY occupancy members copied byte-for-byte.',
        camera_status=camera_status, cases=cases, validated_analysis=analysis)


def write_bundle(output, reports, arrays, metadata):
    if output.exists():
        raise FileExistsError(output)
    metadata = {**metadata, 'files': [], 'source_npz': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            created = True
            for name, path in reports.items():
                payload = path.read_bytes()
                archive.writestr(name, payload)
                metadata['files'].append(dict(archive_path=name, source_path=str(path),
                    bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
            for prefix, path in arrays.items():
                metadata['source_npz'].append(dict(prefix=prefix, source_path=str(path),
                    bytes=path.stat().st_size, sha256=sha(path)))
                with zipfile.ZipFile(path) as source:
                    for field in FIELDS:
                        payload = source.read(field); name = prefix+'/'+field
                        archive.writestr(name, payload)
                        metadata['files'].append(dict(archive_path=name, source_path=str(path),
                            source_member=field, bytes=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest()))
            archive.writestr('bundle_manifest.json', json.dumps(metadata, indent=2)+'\n')
        with zipfile.ZipFile(output) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f'Bundle integrity failure: {bad}')
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment-root', type=Path, required=True)
    parser.add_argument('--camera-fit-dir', type=Path)
    parser.add_argument('--skip-camera', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, arrays, metadata = collect(args.alignment_root, args.camera_fit_dir, not args.skip_camera)
    print(metadata['camera_status'], flush=True)
    print(f'Validated {len(arrays)} saved NPZs; copying occupancy arrays.', flush=True)
    write_bundle(args.output, reports, arrays, metadata)
    print(f'Saved {args.output} ({args.output.stat().st_size/1024**2:.2f} MiB)', flush=True)


if __name__ == '__main__':
    main()
