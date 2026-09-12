"""Compare the sample-independent surface training control to completed real arms."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from analyze_object_transfer import object_losses, validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPLITS = ('fit', 'held_view', 'held_object')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_constant(report, baseline):
    assert report['settings']['model_key'] == 'oracle_constant'
    # Reuse the original no-dropout protocol validator. Only its enum changes;
    # keep the actual report unmodified and explicitly validate the intervention.
    protocol_view = copy.deepcopy(report)
    protocol_view['settings']['model_key'] = 'oracle'
    values = validate(protocol_view)
    for field in ('plan', 'plan_sha256', 'schedule', 'schedule_sha256', 'reference_sha256',
                  'initial_generator_sha256', 'initial_model_sha256', 'initial_trainable_ca_sha256',
                  'initial_encoder_sha256', 'trainable_parameters', 'input_batches'):
        assert report[field] == baseline[field], field
    assert report['initial_real_replay_passed'] and report['initial_real_replay'] == baseline['assessments'][0]['rows']
    assert report['constant_context_verified']
    assert report['constant_token_counts'] == {'visual': 7528, 'surface': 1024}
    bank = report['constant_bank']
    assert (bank['batch'], bank['row']) == (0, 0)
    assert bank['shape'] == [1, 1024, 32]
    assert bank['dtype'] in ('torch.float32', 'torch.bfloat16', 'torch.float16')
    assert len(bank['sha256']) == 64 and set(bank['sha256']) <= set('0123456789abcdef')
    assert bank['sample_id'] == baseline['input_batches'][0]['sample_ids'][0]
    assert bank['object_id'] == baseline['input_batches'][0]['object_ids'][0]
    assert bank['object_id'] in report['plan']['training_object_ids']
    assert bank['object_id'] not in report['plan']['held_object_ids']
    rows = {(r['batch'], r['swapped_surface']): r for r in report['final_rows']}
    for batch in range(32):
        for field in ('scalar_losses', 'per_object_losses'):
            assert rows[batch, False][field] == rows[batch, True][field]
    return values


def paired(delta):
    return dict(mean_delta=mean(delta), median_delta=median(delta),
                objects_improved=sum(v < 0 for v in delta), objects=len(delta))


def analyze(constant, baselines):
    assert set(baselines) == {'image', 'camera', 'oracle'}
    values = {k: validate(r) for k, r in baselines.items()}
    values['constant'] = validate_constant(constant, baselines['oracle'])
    for key, report in baselines.items():
        assert report['settings']['model_key'] == key
        for field in ('plan', 'initial_generator_sha256', 'initial_trainable_ca_sha256', 'schedule_sha256'):
            assert report[field] == constant[field]
        for name, digest in report['source_sha256'].items():
            assert constant['source_sha256'][name] == digest
        for actual, real in zip(constant['input_batches'], report['input_batches']):
            assert {k: v for k, v in actual.items() if k != 'features_sha256'} == {
                k: v for k, v in real.items() if k != 'features_sha256'}
    summary = {}
    for split in SPLITS:
        ids = sorted(values['image'][split])
        summary[split] = dict(cells={k: dict(mean_loss=mean(v['correct'] for v in groups[split].values()),
                                            median_loss=median(v['correct'] for v in groups[split].values()))
                                    for k, groups in values.items()}, paired_comparisons={})
        for treatment, baseline in [('constant', 'image'), ('oracle', 'constant'), ('camera', 'constant')]:
            summary[split]['paired_comparisons'][f'{treatment}_minus_{baseline}'] = paired([
                values[treatment][split][o]['correct']-values[baseline][split][o]['correct'] for o in ids])
    curves, trajectory_comparisons = {}, {}
    reports = {**baselines, 'constant': constant}
    for key, report in reports.items():
        curves[key] = []
        for assessment in report['assessments']:
            objects = object_losses(report, assessment['rows'], 4, False)
            curves[key].append(dict(step=assessment['step'], **{
                split: mean(v['correct'] for v in objects[split].values()) for split in SPLITS}))
    for i, step in enumerate((0, 256, 512, 1024)):
        objects = {k: object_losses(r, r['assessments'][i]['rows'], 4, False) for k, r in reports.items()}
        trajectory_comparisons[step] = {}
        for split in SPLITS:
            ids = sorted(objects['image'][split])
            trajectory_comparisons[step][split] = {
                f'{t}_minus_{b}': paired([objects[t][split][o]['correct']-objects[b][split][o]['correct'] for o in ids])
                for t, b in [('constant', 'image'), ('oracle', 'constant'), ('camera', 'constant')]}
    return dict(summary=summary, curves=curves, trajectory_comparisons=trajectory_comparisons, per_object=values,
                limits='Same 16 fitted/16 held identities and one seed; development pool. The fixed bank '
                       'is one actual training surface, repeated for every input from the start. It carries '
                       'no sample-specific surface information. Different constant banks may behave differently. '
                       'A weak/negative effect is not an equivalence proof or evidence that VecSetX loses geometry. '
                       'Mean/median and object-level effects matter; no draws-as-independent-object inference. '
                       'Monitor bank 600000 and final bank 700000 match the original real-surface training reports; '
                       'do not compare absolute values with the checkpoint probe bank 800000. No production fix established.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('constant_report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    constant = json.loads(args.constant_report.read_text())
    baselines, hashes = {}, {'constant': sha(args.constant_report)}
    for key in ('image', 'camera', 'oracle'):
        path = HERE/'object_transfer_returned_manual'/key/'results.json'
        baselines[key] = json.loads(path.read_text()); hashes[key] = sha(path)
    assert constant['baseline_report_sha256'] == hashes['oracle']
    for name, digest in constant['source_sha256'].items():
        assert sha(REPO/name) == digest, name
    result = analyze(constant, baselines); result['report_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))
