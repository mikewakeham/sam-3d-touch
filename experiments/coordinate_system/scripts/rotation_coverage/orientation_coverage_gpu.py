"""Prepare a rotation bank, then run one matched orientation-coverage arm.

GT grids are labels only. Observed points are rotated and re-encoded by frozen
VecSetX. All visual-present updates retain their original coherent inputs.
"""

# Allow both direct CLI execution and imports across experiment groups.
import sys as _sys
from pathlib import Path as _Path
_repo = next(p for p in _Path(__file__).resolve().parents if (p / 'train.py').is_file() and (p / 'sam3d_objects').is_dir())
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))
from experiments.coordinate_system.scripts.shared.paths import source_path as _source_path

import argparse
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

HERE=Path(__file__).resolve().parent
REPO=next(p for p in Path(__file__).resolve().parents if (_source_path(p, 'train.py')).is_file() and (p / 'sam3d_objects').is_dir())
sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.utils.checkpoint
from omegaconf import OmegaConf
from dataloader import build_dataloader,collate_touch_batch,load_data_config
from train import TouchTrainingModel,amp,build_optimizer,build_stage1_pipeline,prepare_batch,trainable_state_dict
from experiments.coordinate_system.scripts.tiny_fit.tiny_fit_gpu import parameter_digest
from experiments.coordinate_system.scripts.tiny_fit.probe_tiny_fit_views import choose_views
from experiments.noisy_target.evaluate import tensor_digest
from experiments.coordinate_system.scripts.shared_orientation.scope.continue_shared_orientation_gpu import optimizer_digest
from experiments.coordinate_system.scripts.shared_orientation.scope.shared_orientation_scope_protocol import additional_shape_parameter
from experiments.coordinate_system.scripts.rotation_coverage.orientation_coverage_protocol import rotations, indices, rotate_grid, inverse_frame, schedule
from experiments.coordinate_system.scripts.shared_orientation.shared_orientation_protocol import transform_points
from experiments.coordinate_system.scripts.shared.rollout_gpu import occupancy, geometry_metrics
from experiments.coordinate_system.scripts.shared.pose_shape_geometry import points, metrics
from experiments.coordinate_system.scripts.shared.bundle_alignment_geometry import write_bundle


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def seed(value=29):random.seed(value);np.random.seed(value);torch.manual_seed(value)


def read_refs(args):
    fit=json.loads((args.fit_dir/'results.json').read_text())
    probe=json.loads((args.visual_probe_dir/'results.json').read_text())
    assert fit['complete'] and fit['settings']['arm']=='shape_path'
    assert fit['settings']['base_step']==fit['settings']['additional_steps']==1000
    assert probe['complete'] and probe['parameters_unchanged'] and probe['settings']['training_updates']==0
    assert sha(args.fit_dir/'results.json')==probe['fit_report_sha256']
    assert fit['final_all_parameters_sha256']==probe['checkpoint_parameters_sha256']
    for name,value in probe['source_sha256'].items():assert sha(_source_path(REPO, name))==value,name
    for name in ['checkpoint_2000.pt']+[f'{c}_g{g}_d{d}.npz' for c in ('correct_surface','wrong_surface') for g in range(7) for d in range(2)]:
        if not (args.fit_dir/name).is_file():raise FileNotFoundError(args.fit_dir/name)
    assert sha(args.fit_dir/'checkpoint_2000.pt')==probe['checkpoint_sha256']
    single=json.loads((_source_path(HERE, 'tiny_fit_returned_46083371/camera/results.json')).read_text())
    pipeline_path=Path(single['settings']['pipeline_config']);data_path=Path(single['settings']['data_config'])
    assert pipeline_path.read_text()==single['pipeline_yaml'] and data_path.read_text()==single['data_yaml']
    cfg=OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent/cfg.ss_generator_config_path).read_text()==single['generator_yaml']
    payloads={};supports=[]
    with zipfile.ZipFile(args.fit_dir/'scope_shape_path_bundle.zip') as z:
        m=json.loads(z.read('bundle_manifest.json'));assert m['format_version']==4
        entries={e['archive_path']:e for e in m['files']}
        def payload(n):
            b=z.read(n);assert hashlib.sha256(b).hexdigest()==entries[n]['sha256'];return b
        assert json.loads(payload('scope_shape_path/results.json'))==fit
        for n in ['reference/parent_results.json','reference/original_target_occupancy.npy']+[f'physical/{f["sample_id"]}.npy' for f in fit['target_frames']]:payloads[n]=payload(n)
        for g in range(7):supports.append(np.load(io.BytesIO(payload(f'scope_shape_path/correct_surface_g{g}_d0/target_occupancy.npy')),allow_pickle=False))
    for k in ('input_batches','target_frames','target_quality'):assert fit[k]==probe[k]
    return fit,probe,single,cfg,payloads,supports


def construct(single,device):
    seed();pipeline=build_stage1_pipeline(Path(single['settings']['pipeline_config']),device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder=TouchEncoder(output_dim=pipeline.backbone.cond_channels,trainable=False,use_position=False).to(device).eval()
    return pipeline,encoder


def prepare_inputs(args,fit,single,pipeline,encoder,device,supports):
    data=load_data_config(Path(single['settings']['data_config']));data['dataset']['split']='train'
    ds=build_dataloader(data,4,0,shuffle=False,include_touch=True,oracle_point_frame=False).dataset
    groups=choose_views(ds.records,single['sample_ids'],count=6);result=[]
    for g,records in enumerate(groups):
        ds.records=records;batch=collate_touch_batch([ds[i] for i in range(4)])
        target,ca,kw,xyz,mask=prepare_batch(pipeline,batch,device,'bf16',True,False,False)
        assert len(ca)==1 and not kw and mask.all() and xyz.shape[1]==8192
        with torch.no_grad(),amp(device,'bf16'):
            pp,mm,_,_=encoder.prepare_points(xyz,mask);features=encoder.encoder.encode(pp,mm)['x'].detach()
        meta=dict(group=g,split='fit' if g<4 else 'reserved_view',sample_ids=[r['sample_id'] for r in records],
            image_sha256=tensor_digest(batch['image']),pointmap_sha256=tensor_digest(batch['pointmap']),
            target_sha256=tensor_digest(target['shape']),features_sha256=tensor_digest(features))
        assert meta=={k:v for k,v in fit['input_batches'][g].items() if k!='shared_target_sha256'}
        with np.load(args.fit_dir/f'correct_surface_g{g}_d0.npz',allow_pickle=False) as z:
            target['shape']=torch.from_numpy(z['target'].copy()).to(device)
            np.testing.assert_array_equal(z['target_occupancy'],supports[g])
        meta['shared_target_sha256']=tensor_digest(target['shape']);assert meta==fit['input_batches'][g]
        result.append(dict(target=target,visual=ca[0],features=features,points=pp,mask=mm))
    return result


def prepare_bank(args,refs,device):
    fit,probe,single,cfg,payloads,supports=refs
    args.output_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic();rr=rotations()
    parent=json.loads(payloads['reference/parent_results.json'])
    assert sha(args.encoder_checkpoint)==parent['encoder_checkpoint_sha256']
    source=_source_path(REPO, 'data_generation/objaverse-dexonomy/generate_target_latents.py')
    spec=importlib.util.spec_from_file_location('coverage_target_encoder',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=dict(complete=False,fit_report_sha256=sha(args.fit_dir/'results.json'),probe_report_sha256=sha(args.visual_probe_dir/'results.json'),
        source_sha256={**probe['source_sha256'],**{str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),_source_path(HERE, 'orientation_coverage_protocol.py'),_source_path(HERE, 'run_orientation_coverage_pair.py'),_source_path(HERE, 'run_shared_orientation_scope_pair.py')]}},
        encoder_checkpoint_sha256=sha(args.encoder_checkpoint),rotations=[r.tolist() for r in rr],
        input_batches=fit['input_batches'],identity_checks=[],quality=[],entries=[],decoded_artifacts=[],runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(device)),
        target_quality_policy=dict(roundtrip_iou_min=.95,common_precision_recall_2v_min=.95),
        scope='Only four fitted groups generate augmentation entries. Reserved groups are verified, never added to the training bank. '
        'Exact signed-axis grid rotations; re-encoded target labels and observed VecSetX features. No GT conditioner.')
    def save():(args.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    save();bank={};vae=module.load_encoder(args.encoder_checkpoint,device).requires_grad_(False)
    with torch.inference_mode():
        for g in range(4):
            ids=fit['input_batches'][g]['sample_ids']
            physical=[np.load(io.BytesIO(payloads[f'physical/{sid}.npy']),allow_pickle=False) for sid in ids]
            with np.load(args.fit_dir/f'correct_surface_g{g}_d0.npz',allow_pickle=False) as z:identity=z['target'].copy()
            for ri,r in enumerate(rr):
                grids=np.stack([rotate_grid(a,r) for a in physical]);latents=[]
                for i,grid in enumerate(grids):
                    x=torch.from_numpy(grid.astype(np.float32))[None,None].to(device)
                    mean=vae(x)['mean'][0].float().cpu().numpy()
                    flat=np.ascontiguousarray(mean.transpose(1,2,3,0).reshape(4096,8))
                    assert np.isfinite(flat).all();latents.append(flat)
                    if ri==0:
                        np.testing.assert_allclose(flat,identity[i],rtol=1e-4,atol=1e-5)
                        report['identity_checks'].append(dict(sample_id=ids[i],max_target_error=float(np.max(np.abs(flat-identity[i])))))
                # Exact historical identity labels remain the control, regardless
                # of allowable numerical replay drift in the encoder check.
                target=identity if ri==0 else np.stack(latents)
                bank[f'{g}/{ri}']=dict(target=torch.from_numpy(target),physical=torch.from_numpy(grids))
            save();print('Encoded target rotation bank group',g,flush=True)
    del vae;gc.collect();torch.cuda.empty_cache()
    pipeline,encoder=construct(single,device)
    prepared=prepare_inputs(args,fit,single,pipeline,encoder,device,supports)
    decoder=pipeline.init_ss_decoder(cfg.ss_decoder_config_path,cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    original=np.load(io.BytesIO(payloads['reference/original_target_occupancy.npy']),allow_pickle=False)
    frames={f['sample_id']:f for f in fit['target_frames']}
    with torch.no_grad():
        for g in range(4):
            p=prepared[g]
            for ri,r in enumerate(rr):
                key=f'{g}/{ri}';entry=bank[key];cols,signs=indices(r)
                rotated=p['points'][...,cols.tolist()]*torch.as_tensor(signs,device=device,dtype=p['points'].dtype)
                if ri==0:features=p['features']
                else:
                    with amp(device,'bf16'):features=encoder.encoder.encode(rotated,p['mask'])['x'].detach()
                entry['features']=features.cpu();decoded=[]
                for i,sid in enumerate(fit['input_batches'][g]['sample_ids']):
                    with amp(device,'bf16'):support=occupancy(decoder,entry['target'][i:i+1].to(device))
                    physical=entry['physical'][i].numpy();inv=inverse_frame(frames[sid]['object_from_output'],r)
                    common=metrics(transform_points(points(support),inv),points(original[i]))
                    q=dict(sample_id=sid,group=g,rotation_index=ri,roundtrip=geometry_metrics(support,physical),common_object_units=common)
                    report['quality'].append(q);decoded.append(support)
                    if ri==0:np.testing.assert_array_equal(support,supports[g][i])
                entry['support']=torch.from_numpy(np.stack(decoded))
                report['entries'].append(dict(key=key,group=g,rotation_index=ri,sample_ids=fit['input_batches'][g]['sample_ids'],
                    features_sha256=tensor_digest(entry['features']),target_sha256=tensor_digest(entry['target']),
                    physical_sha256=tensor_digest(entry['physical']),support_sha256=tensor_digest(entry['support']),
                    points_sha256=tensor_digest(rotated)))
            save();print('Encoded points and decoded labels group',g,flush=True)
    failed=[q for q in report['quality'] if q['roundtrip']['voxel_iou']<.95 or min(q['common_object_units']['precision_2v'],q['common_object_units']['recall_2v'])<.95]
    report['failed_quality']=failed;save()
    if failed:raise RuntimeError('Rotation-bank target quality failed before training; return results.partial.json')
    assert len(bank)==96 and len(report['quality'])==384
    for g in range(4):
        for ri in range(24):
            entry=bank[f'{g}/{ri}'];name=f'target_bank_g{g}_r{ri}.npz'
            np.savez_compressed(args.output_dir/name,predicted_occupancy=entry['support'].numpy(),target_occupancy=entry['physical'].numpy())
            report['decoded_artifacts'].append(dict(file=name,group=g,rotation_index=ri,sha256=sha(args.output_dir/name),sample_ids=fit['input_batches'][g]['sample_ids']))
    torch.save(bank,args.output_dir/'bank.pt');report['bank_sha256']=sha(args.output_dir/'bank.pt')
    report['complete']=True;report['elapsed_seconds']=time.monotonic()-started
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Rotation bank verified. Ready for matched arms.',flush=True)


def fit_arm(args,refs,device):
    fit,probe,single,cfg,payloads,supports=refs
    bank_report=json.loads((args.bank_dir/'results.json').read_text());assert bank_report['complete']
    assert bank_report['fit_report_sha256']==sha(args.fit_dir/'results.json')
    assert bank_report['probe_report_sha256']==sha(args.visual_probe_dir/'results.json')
    for n,v in bank_report['source_sha256'].items():assert sha(_source_path(REPO, n))==v,n
    assert sha(args.bank_dir/'bank.pt')==bank_report['bank_sha256']
    bank=torch.load(args.bank_dir/'bank.pt',map_location='cpu',weights_only=True,mmap=True)
    assert len(bank)==96
    for e in bank_report['entries']:
        for field in ('features','target','physical','support'):assert tensor_digest(bank[e['key']][field])==e[field+'_sha256']
    assert len(bank_report['decoded_artifacts'])==96
    for e in bank_report['decoded_artifacts']:
        assert sha(args.bank_dir/e['file'])==e['sha256']
        with np.load(args.bank_dir/e['file'],allow_pickle=False) as z:
            entry=bank[f'{e["group"]}/{e["rotation_index"]}']
            np.testing.assert_array_equal(z['predicted_occupancy'],entry['support'].numpy())
            np.testing.assert_array_equal(z['target_occupancy'],entry['physical'].numpy())
    pipeline,encoder=construct(single,device);model=TouchTrainingModel(pipeline.ss_generator,encoder,False,False)
    optimizer,optimized=build_optimizer(encoder,pipeline.backbone,SimpleNamespace(learning_rate=1e-4,cross_attention_learning_rate=1e-5,cross_attention_scope='full'))
    assert [n for n,p in model.named_parameters() if p.requires_grad]==fit['common_parameter_names']
    extra=[(n,p) for n,p in pipeline.backbone.named_parameters() if additional_shape_parameter(n)]
    assert [n for n,p in extra]==fit['additional_parameter_names']
    for _,p in extra:p.requires_grad_(True)
    optimizer.add_param_group(dict(params=[p for _,p in extra],lr=1e-5,name='additional_shape_path'));optimized+=[p for _,p in extra]
    assert len({id(p) for p in optimized})==len(optimized)
    assert {id(p) for p in optimized}=={id(p) for p in model.parameters() if p.requires_grad}
    assert {g['name']:sum(p.numel() for p in g['params']) for g in optimizer.param_groups}==fit['trainable_counts']
    assert len(pipeline.backbone.blocks)==24 and not pipeline.backbone.share_mod
    for b in pipeline.backbone.blocks:
        assert b.self_attn.protect_modality_list==['shape'];b.use_checkpoint=True
    saved=torch.load(args.fit_dir/'checkpoint_2000.pt',map_location='cpu',weights_only=True,mmap=True)
    assert saved['step']==2000 and saved['settings']==fit['settings'] and saved['conditioning_config']==model.conditioning_config
    assert saved['parent_report_sha256']==fit['parent_report_sha256']
    params=dict(model.named_parameters());assert set(saved['model'])=={n for n,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for n,v in saved['model'].items():params[n].copy_(v)
    initial=parameter_digest(model.named_parameters());assert initial==fit['final_all_parameters_sha256']
    optimizer.load_state_dict(saved['optimizer']);opt_hash=optimizer_digest(saved['optimizer'])
    assert optimizer_digest(optimizer.state_dict())==opt_hash
    assert [g['lr'] for g in optimizer.param_groups]==[1e-4,1e-5,1e-5]
    assert all(g['weight_decay']==0 for g in optimizer.param_groups)
    del saved
    prepared=prepare_inputs(args,fit,single,pipeline,encoder,device,supports)
    for g in range(4):
        assert tensor_digest(bank[f'{g}/0']['features'])==fit['input_batches'][g]['features_sha256']
        assert tensor_digest(bank[f'{g}/0']['target'])==fit['input_batches'][g]['shared_target_sha256']
    assert not any(p.requires_grad for p in encoder.encoder.parameters())
    frozen=parameter_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    args.output_dir.mkdir(parents=True,exist_ok=False);ss=schedule();rr=rotations()
    report=dict(settings=dict(arm=args.arm,base_step=2000,additional_steps=1000,seed=29,precision='bf16',visual_dropout_fraction=.5,
        target_convention='camera_axes_centered_unit_bbox',rotation_bank_size=24,sampling_steps=25,cfg=0),
        fit_report_sha256=sha(args.fit_dir/'results.json'),probe_report_sha256=sha(args.visual_probe_dir/'results.json'),
        bank_report_sha256=sha(args.bank_dir/'results.json'),source_sha256=bank_report['source_sha256'],
        initial_all_parameters_sha256=initial,restored_optimizer_sha256=opt_hash,
        schedule_sha256=hashlib.sha256(json.dumps(ss,sort_keys=True).encode()).hexdigest(),schedule=ss,
        input_batches=fit['input_batches'],target_frames=fit['target_frames'],target_quality=fit['target_quality'],
        trainable_counts={g['name']:sum(p.numel() for p in g['params']) for g in optimizer.param_groups},
        native=[],training=[],rows=[],artifacts=[],interface_checks={},complete=False,
        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(device)),
        scope='Matched step2000 resume with Adam continuity. Only dropped-visual update orientation changes. '
        'Natural reserved views never train. Full surfaces now; no new-object or arbitrary-angle equivariance claim.')
    def save():(args.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    save();gen=pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional==gen.self_consistency_prob==gen.fm_eps_max==0
    assert gen.loss_weights['shape']==1 and all(v==0 for k,v in gen.loss_weights.items() if k!='shape')

    def loss_at(g,zero=False,wrong=False,ri=0):
        p=prepared[g];target=p['target'];features=p['features'];visual=p['visual']
        if ri:
            assert zero and g<4
            entry=bank[f'{g}/{ri}'];target={**target,'shape':entry['target'].to(device)};features=entry['features'].to(device)
        if zero:visual=torch.zeros_like(visual)
        tokens=encoder.output_projection(features)+encoder.touch_embedding
        if wrong:tokens=tokens.roll(1,0)
        key='augmented_zero' if ri else 'natural_zero' if zero else 'natural_present'
        def check(module,inputs):
            if key not in report['interface_checks']:
                torch.testing.assert_close(inputs[0],torch.cat((visual,tokens.to(visual)),1).to(inputs[0]),rtol=0,atol=0)
                report['interface_checks'][key]=True
        hook=pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check)
        try:value,_=gen.loss(target,visual,touch_tokens=tokens)
        finally:hook.remove()
        assert torch.isfinite(value)
        return value

    def assess(step):
        rows=[]
        with torch.no_grad(),amp(device,'bf16'):
            gen.reverse_fn.training=True
            for v in ('present','zero'):
                for g in range(7):
                    for wrong in (False,True):
                        values=[]
                        for draw in range(8):
                            torch.manual_seed(400029+draw);random.seed(400029+draw)
                            values.append(float(loss_at(g,v=='zero',wrong)))
                        rows.append(dict(visual_mode=v,group=g,wrong=wrong,losses=values))
        report['native'].append(dict(step=step,rows=rows));save();return rows

    replay=assess(2000)
    expected={(r['visual_mode'],r['group'],r['wrong']):r['losses'] for r in probe['native']}
    a=np.array([r['losses'] for r in replay]);b=np.array([expected[r['visual_mode'],r['group'],r['wrong']] for r in replay])
    report['resume_replay']=dict(max_absolute_error=float(np.max(np.abs(a-b))),max_relative_error=float(np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-12))),rtol=.001,atol=.000001)
    save();np.testing.assert_allclose(a,b,rtol=.001,atol=.000001)
    print(args.arm,'initial replay and optimizer checks passed',flush=True);started=time.monotonic()
    for j,item in enumerate(ss,1):
        ri=item['rotation_index'] if args.arm=='augmented' else 0
        torch.manual_seed(item['noise_seed']);random.seed(item['noise_seed']);gen.reverse_fn.training=True
        optimizer.zero_grad(set_to_none=True)
        with amp(device,'bf16'):loss=loss_at(item['group'],item['visual_dropped'],ri=ri)
        loss.backward();assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(optimized,1.,error_if_nonfinite=True);optimizer.step()
        report['training'].append(dict(**item,applied_rotation_index=ri,loss=float(loss),preclip_gradient_norm=float(norm)))
        if j%20==0:save();print(args.arm,item['step'],'loss',float(loss),flush=True)
        if j in (300,1000):
            assess(item['step'])
            torch.save(dict(model=trainable_state_dict(model),optimizer=optimizer.state_dict(),step=item['step'],settings=report['settings'],
                conditioning_config=model.conditioning_config,fit_report_sha256=report['fit_report_sha256']),args.output_dir/f'checkpoint_{item["step"]}.pt')
    assert sum(r['applied_rotation_index']!=0 for r in report['training'])==(500 if args.arm=='augmented' else 0)
    assert all(r['visual_dropped'] for r in report['training'] if r['applied_rotation_index'])
    report['training_seconds']=time.monotonic()-started
    assert parameter_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen
    report['frozen_parameters_unchanged']=True;final=parameter_digest(model.named_parameters());report['final_all_parameters_sha256']=final
    decoder=pipeline.init_ss_decoder(cfg.ss_decoder_config_path,cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    gen.no_shortcut=True;gen.inference_steps=25;gen.rescale_t=float(cfg.get('ss_rescale_t',3))
    gen.reverse_fn.interval=list(cfg.get('ss_cfg_interval',[0,500]));gen.reverse_fn.strength=0;gen.reverse_fn.training=False
    original=np.load(io.BytesIO(payloads['reference/original_target_occupancy.npy']),allow_pickle=False)
    frames={f['sample_id']:f for f in fit['target_frames']};paired=json.loads((_source_path(HERE, 'camera_dropout_sampling_reference.json')).read_text())
    cases=[dict(group=g,rotation_index=0,visual_mode=v,condition=c,split='fit' if g<4 else 'reserved_view')
           for g in range(7) for v in ('present','zero') for c in ('correct_surface','wrong_surface')]
    cases += [dict(group=0,rotation_index=ri,visual_mode='zero',condition='correct_surface',split='augmented_fit_diagnostic') for ri in (1,7,13,19)]
    with torch.no_grad():
        for case in cases:
            g,ri=case['group'],case['rotation_index'];p=prepared[g];target=p['target'];features=p['features'];qs=supports[g]
            if ri:
                entry=bank[f'{g}/{ri}'];target={**target,'shape':entry['target'].to(device)};features=entry['features'].to(device);qs=entry['support'].numpy()
            visual=torch.zeros_like(p['visual']) if case['visual_mode']=='zero' else p['visual']
            with amp(device,'bf16'):tokens=encoder.output_projection(features)+encoder.touch_embedding
            if case['condition']=='wrong_surface':tokens=tokens.roll(1,0)
            for draw in range(2):
                torch.manual_seed(200029+draw);noise=gen._generate_x0(target)
                for i,sid in enumerate(fit['input_batches'][g]['sample_ids']):
                    assert {n:tensor_digest(v[i]) for n,v in noise.items()}==paired['noise'][sid.rsplit('_',1)[0]+'/'+str(draw)]
                check_key=f'sampled/{case["visual_mode"]}/{case["condition"]}/{bool(ri)}'
                def check(module,inputs):
                    if check_key not in report['interface_checks']:
                        torch.testing.assert_close(inputs[0],torch.cat((visual,tokens.to(visual)),1).to(inputs[0]),rtol=0,atol=0)
                        report['interface_checks'][check_key]=True
                hook=pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check)
                try:
                    with amp(device,'bf16'),patch.object(gen,'_generate_noise',side_effect=lambda *a,**k:{n:v.clone() for n,v in noise.items()}):
                        pred=gen({n:tuple(v.shape) for n,v in target.items()},device,visual,touch_tokens=tokens)['shape']
                finally:hook.remove()
                assert torch.isfinite(pred).all();outputs=[]
                for i,sid in enumerate(fit['input_batches'][g]['sample_ids']):
                    with amp(device,'bf16'):support=occupancy(decoder,pred[i:i+1])
                    outputs.append(support);inv=inverse_frame(frames[sid]['object_from_output'],rr[ri])
                    report['rows'].append(dict(**case,draw=draw,sample_id=sid,object_id=sid.rsplit('_',1)[0],object_from_output=inv.tolist(),
                        noise_sha256={n:tensor_digest(v[i]) for n,v in noise.items()},
                        latent_mse=float((pred[i].float()-target['shape'][i].float()).square().mean()),
                        **geometry_metrics(support,qs[i]),common_object_units=metrics(transform_points(points(support),inv),points(original[i]))))
                name=f'{case["visual_mode"]}_{case["condition"]}_g{g}_r{ri}_d{draw}.npz'
                np.savez_compressed(args.output_dir/name,prediction=pred.float().cpu().numpy(),target=target['shape'].cpu().numpy(),predicted_occupancy=np.stack(outputs),target_occupancy=qs)
                report['artifacts'].append(dict(file=name,**case,draw=draw,sample_ids=fit['input_batches'][g]['sample_ids']))
            save();print(args.arm,'sampled',case,flush=True)
    assert parameter_digest(model.named_parameters())==final
    assert len(report['rows'])==256 and len(report['artifacts'])==64
    assert all(report['interface_checks'].values()) and len(report['interface_checks'])==(8 if args.arm=='augmented' else 7)
    report['parameters_unchanged_during_sampling']=True;report['complete']=True
    result=args.output_dir/'results.json';result.write_text(json.dumps(report,indent=2)+'\n')
    refs_out={'coverage/results.json':result,'reference/fit_results.json':args.fit_dir/'results.json',
        'reference/visual_probe_results.json':args.visual_probe_dir/'results.json','reference/bank_results.json':args.bank_dir/'results.json'}
    for name,b in payloads.items():
        local=args.output_dir/'references'/name;local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(b);refs_out[name]=local
    arrays={f'coverage/{Path(a["file"]).stem}':args.output_dir/a['file'] for a in report['artifacts']}
    arrays.update({f'target_bank/{Path(a["file"]).stem}':args.bank_dir/a['file'] for a in bank_report['decoded_artifacts']})
    bundle=args.output_dir/f'coverage_{args.arm}_bundle.zip'
    write_bundle(bundle,refs_out,arrays,dict(format_version=6,scope=report['scope'],
        cases=[dict(kind='coverage',prefix=f'coverage/{Path(a["file"]).stem}',**a) for a in report['artifacts']],
        target_bank_cases=[dict(prefix=f'target_bank/{Path(a["file"]).stem}',**a) for a in bank_report['decoded_artifacts']]))
    print('Return',bundle,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['prepare','fit'],required=True)
    ap.add_argument('--arm',choices=['control','augmented'])
    ap.add_argument('--fit-dir',type=Path,required=True)
    ap.add_argument('--visual-probe-dir',type=Path,required=True)
    ap.add_argument('--bank-dir',type=Path)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--encoder-checkpoint',type=Path,default=Path('checkpoints/hf/ss_encoder.ckpt'))
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.phase=='fit' and (args.arm is None or args.bank_dir is None):ap.error('fit requires --arm and --bank-dir')
    refs=read_refs(args);device=torch.device('cuda',0);torch.cuda.set_device(device)
    if args.phase=='prepare':prepare_bank(args,refs,device)
    else:fit_arm(args,refs,device)


if __name__=='__main__':main()
