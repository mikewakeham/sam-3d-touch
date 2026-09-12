"""Analyze the four matched image-anchor surface-adaptation arms; CPU only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from analyze_object_transfer import object_losses, validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = ('separate_oracle', 'separate_constant', 'separate_camera', 'joint_oracle')
SPLITS = ('fit', 'held_view', 'held_object')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_arm(report, anchor, frame):
    key = report['settings']['model_key']
    assert key in MODELS
    separate = key.startswith('separate_')
    protocol_view = copy.deepcopy(report)
    protocol_view['settings']['model_key'] = 'camera' if key == 'separate_camera' else 'oracle'
    values = validate(protocol_view)
    for field in ('plan', 'plan_sha256', 'schedule', 'schedule_sha256', 'reference_sha256'):
        assert report[field] == anchor[field]
    assert report['anchor_model_sha256'] == anchor['final_model_sha256']
    assert report['input_batches'] == frame['input_batches']
    assert report['initial_encoder_sha256'] == frame['initial_encoder_sha256']
    assert report['initial_removed_replay_passed']
    assert report['initial_removed_replay'] == anchor['assessments'][-1]['rows']
    expected_module_checks = dict(zero_residual_exact=True, absent_surface_exact=True,
        independent_sum_exact=True, output_gradient_nonzero=True, visual_parameters_no_grad=True,
        hidden_state_gradient_finite=True, wrong_length_rejected=True)
    assert report['module_checks'] == expected_module_checks
    assert report['initial_frozen_parameters_sha256'] == report['final_frozen_parameters_sha256']
    assert report['interface_context_checks'] == ({'visual': 7528, 'surface': 1024} if separate else {'joint': 8552})
    assert report['initial_zero_residual_exact'] == (report['assessments'][0]['rows'] == anchor['assessments'][-1]['rows'])
    assert report['final_removed_anchor_exact'] == (report['final_removed_rows'] == anchor['final_rows'])
    if separate:
        assert report['initial_zero_residual_exact'] and report['final_removed_anchor_exact']
        assert set(report['trainable_parameters']) == {'touch_encoder', 'surface_attention'}
    else:
        assert report['trainable_parameters'] == frame['trainable_parameters']
    assert report['trainable_parameters']['touch_encoder'] == frame['trainable_parameters']['touch_encoder']
    object_losses(report, report['final_removed_rows'], 8, False)
    bank = report['constant_bank']
    assert (bank['batch'], bank['row']) == (0, 0)
    assert bank['sample_id'] == frame['input_batches'][0]['sample_ids'][0]
    assert bank['object_id'] == frame['input_batches'][0]['object_ids'][0]
    if key == 'separate_constant':
        rows = {(r['batch'], r['swapped_surface']): r for r in report['final_rows']}
        for b in range(32):
            for name in ('scalar_losses', 'per_object_losses'):
                assert rows[b, False][name] == rows[b, True][name]
    return values


def paired(delta):
    return dict(mean_delta=mean(delta), median_delta=median(delta),
                objects_improved=sum(d < 0 for d in delta), objects=len(delta))


def analyze(reports, anchor, frames):
    assert set(reports) == set(MODELS)
    validate(anchor)
    for frame in frames.values():
        validate(frame)
    values = {'image_anchor': object_losses(anchor, anchor['final_rows'], 8, False)}
    for key, report in reports.items():
        assert report['settings']['model_key'] == key
        values[key] = validate_arm(report, anchor, frames['camera' if key == 'separate_camera' else 'oracle'])
        for field in ('anchor_model_sha256', 'anchor_report_sha256', 'anchor_checkpoint_sha256',
                      'source_sha256', 'schedule_sha256', 'plan_sha256'):
            assert report[field] == reports['separate_oracle'][field]
        if key.startswith('separate_'):
            for field in ('initial_generator_sha256', 'initial_model_sha256', 'initial_trainable_ca_sha256',
                          'initial_frozen_parameters_sha256', 'trainable_parameters'):
                assert report[field] == reports['separate_oracle'][field], field
    assert reports['separate_oracle']['input_batches'] == reports['separate_constant']['input_batches']
    assert reports['separate_oracle']['constant_bank'] == reports['separate_constant']['constant_bank']
    summary = {}
    pairs = [(k, 'image_anchor') for k in MODELS] + [
        ('separate_oracle', 'separate_constant'), ('separate_oracle', 'joint_oracle'),
        ('separate_camera', 'separate_oracle'), ('separate_camera', 'separate_constant')]
    for split in SPLITS:
        ids = sorted(values['image_anchor'][split])
        cells = {}
        for key, groups in values.items():
            objects = groups[split]
            cells[key] = dict(mean_loss=mean(v['correct'] for v in objects.values()),
                              median_loss=median(v['correct'] for v in objects.values()))
            if key != 'image_anchor':
                benefit = [v['wrong']-v['correct'] for v in objects.values()]
                removed = object_losses(reports[key], reports[key]['final_removed_rows'], 8, False)[split]
                cells[key].update(mean_wrong_minus_correct=mean(benefit),
                    objects_preferring_correct=sum(d > 0 for d in benefit),
                    mean_removed_loss=mean(v['correct'] for v in removed.values()))
        summary[split] = dict(cells=cells, paired_comparisons={f'{t}_minus_{b}': paired([
            values[t][split][o]['correct']-values[b][split][o]['correct'] for o in ids]) for t, b in pairs})
    curves = {}
    for key, report in reports.items():
        curves[key] = []
        for assessment in report['assessments']:
            objects = object_losses(report, assessment['rows'], 4, False)
            curves[key].append(dict(step=assessment['step'], **{
                split: mean(v['correct'] for v in objects[split].values()) for split in SPLITS}))
    anchor_monitor = object_losses(anchor, anchor['assessments'][-1]['rows'], 4, False)
    return dict(summary=summary, curves=curves, per_object=values,
                anchor_monitor={s: mean(v['correct'] for v in anchor_monitor[s].values()) for s in SPLITS},
                limits='16 fitted/16 previously inspected held identities; one seed. All arms start from the same '
                       'completed image-1024 anchor and receive 1024 additional surface-adaptation updates. '
                       'Separate versus joint changes freezing, attention normalization, output initialization '
                       'and the location of trainable parameters together; it is a constructive bundle, not '
                       'a one-factor mechanism attribution. Constant is an oracle-frame fixed training bank; '
                       'camera-versus-constant alone does not isolate frame from content. Frozen fallback '
                       'does not guarantee helpful predictions with surfaces present. Require held-object '
                       'benefit over both image and constant, plus correct-surface utility; fit or fallback '
                       'preservation alone is not success. No broad-training or impossibility conclusion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    anchor_path = HERE/'object_transfer_returned_manual/image/results.json'
    anchor = json.loads(anchor_path.read_text())
    frames = {k: json.loads((HERE/'object_transfer_returned_manual'/k/'results.json').read_text()) for k in ('camera', 'oracle')}
    reports, hashes = {}, {}
    for key in MODELS:
        path = args.fit_root/key/'results.json'
        reports[key] = json.loads(path.read_text()); hashes[key] = sha(path)
        assert reports[key]['anchor_report_sha256'] == sha(anchor_path)
        frame_path = HERE/'object_transfer_returned_manual'/('camera' if key == 'separate_camera' else 'oracle')/'results.json'
        assert reports[key]['frame_report_sha256'] == sha(frame_path)
        for name, digest in reports[key]['source_sha256'].items():
            assert sha(REPO/name) == digest, name
    result = analyze(reports, anchor, frames); result['report_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))
