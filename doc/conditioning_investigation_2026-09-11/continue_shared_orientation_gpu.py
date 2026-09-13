"""Equal-exposure continuation: existing CA adaptation versus complete shape path.

Uses the finished shared-orientation run's targets and step1000 checkpoint.
No VAE encoding, new labels, new observations or optimizer reset for old groups.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.utils.checkpoint
from omegaconf import OmegaConf
from dataloader import build_dataloader,collate_touch_batch,load_data_config
from train import (TouchTrainingModel,amp,build_optimizer,build_stage1_pipeline,
                   prepare_batch,trainable_state_dict)
from tiny_fit_gpu import parameter_digest
from probe_tiny_fit_views import choose_views
from experiments.noisy_target.evaluate import tensor_digest
from rollout_gpu import occupancy,geometry_metrics
from pose_shape_geometry import points as support_points,metrics as proximity
from shared_orientation_protocol import transform_points
from shared_orientation_scope_protocol import additional_shape_parameter,continuation_schedule,next_noise_seed
from visual_dropout_protocol import schedule_digest
from bundle_alignment_geometry import write_bundle


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def optimizer_digest(state):
    """Stable content digest, independent of pickle storage/device addresses."""
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((str(x.dtype),tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(state);return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',choices=['current','shape_path'],required=True)
    ap.add_argument('--fit-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    old_path=args.fit_dir/'results.json';old=json.loads(old_path.read_text())
    assert old['complete'] and old['settings']['arm']=='camera_shared_orientation'
    assert old['settings']['steps']==1000 and old['target_quality_gate_passed']
    for name,value in old['source_sha256'].items():assert sha(REPO/name)==value,name
    for name,value in old['reference_sha256'].items():assert sha(HERE/name)==value,name
    checkpoint_path=args.fit_dir/'checkpoint_1000.pt'
    if not checkpoint_path.is_file():raise FileNotFoundError(checkpoint_path)
    single=json.loads((HERE/'tiny_fit_returned_46083371/camera/results.json').read_text())
    paired=json.loads((HERE/'camera_dropout_sampling_reference.json').read_text())
    pipeline_path=Path(single['settings']['pipeline_config']);data_path=Path(single['settings']['data_config'])
    assert pipeline_path.read_text()==single['pipeline_yaml'] and data_path.read_text()==single['data_yaml']
    cfg=OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent/cfg.ss_generator_config_path).read_text()==single['generator_yaml']
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    random.seed(29);np.random.seed(29);torch.manual_seed(29)
    pipeline=build_stage1_pipeline(pipeline_path,device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder=TouchEncoder(output_dim=pipeline.backbone.cond_channels,trainable=False,use_position=False).to(device).eval()
    model=TouchTrainingModel(pipeline.ss_generator,encoder,False,False)
    optimizer,optimized=build_optimizer(encoder,pipeline.backbone,SimpleNamespace(
        learning_rate=1e-4,cross_attention_learning_rate=1e-5,cross_attention_scope='full'))
    parameters=dict(model.named_parameters());common_names=[n for n,p in parameters.items() if p.requires_grad]
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert saved['step']==1000 and saved['settings']==old['settings']
    assert saved['conditioning_config']==model.conditioning_config and saved['source_sha256']==old['source_sha256']
    assert set(saved['model'])==set(common_names)
    with torch.no_grad():
        for name,value in saved['model'].items():parameters[name].copy_(value.to(parameters[name]))
    assert parameter_digest(model.named_parameters())==old['final_all_parameters_sha256']
    optimizer.load_state_dict(saved['optimizer'])
    expected_opt=optimizer_digest(saved['optimizer'])
    assert optimizer_digest(optimizer.state_dict())==expected_opt
    assert [g['lr'] for g in optimizer.param_groups]==[1e-4,1e-5]
    assert all(g['weight_decay']==0 for g in optimizer.param_groups)
    del saved
    extra=[(n,p) for n,p in pipeline.backbone.named_parameters() if additional_shape_parameter(n)]
    assert extra and all(not p.requires_grad for _,p in extra)
    assert len(pipeline.backbone.blocks)==24 and not pipeline.backbone.share_mod
    for b in pipeline.backbone.blocks:
        assert b.self_attn.protect_modality_list==['shape']
        b.use_checkpoint=True  # Both arms; non-reentrant, no model-function change.
    if args.arm=='shape_path':
        for _,p in extra:p.requires_grad_(True)
        optimizer.add_param_group(dict(params=[p for _,p in extra],lr=1e-5,name='additional_shape_path'))
        optimized=optimized+[p for _,p in extra]
    assert len({id(p) for p in optimized})==len(optimized)
    assert {id(p) for p in optimized}=={id(p) for p in model.parameters() if p.requires_grad}
    assert not any(p.requires_grad for p in encoder.encoder.parameters())
    if pipeline.ss_condition_embedder is not None:
        assert not any(p.requires_grad for p in pipeline.ss_condition_embedder.parameters())
    frozen_start=parameter_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    extra_start=parameter_digest(extra)
    schedule=continuation_schedule()
    report=dict(settings=dict(arm=args.arm,base_step=1000,additional_steps=1000,seed=29,precision='bf16',
        fit_objects=4,fit_views=4,reserved_views=3,visual_dropout_fraction=.5,activation_checkpointing=True,
        sampling_steps=25,cfg=0,target_convention='camera_axes_centered_unit_bbox'),
        parent_report_sha256=sha(old_path),parent_checkpoint_sha256=sha(checkpoint_path),
        source_sha256={**old['source_sha256'],**{str(p.relative_to(REPO)):sha(p) for p in
            [Path(__file__),HERE/'shared_orientation_scope_protocol.py',
             REPO/'sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py',
             REPO/'sam3d_objects/model/backbone/tdfy_dit/modules/transformer/modulated.py',
             REPO/'sam3d_objects/model/backbone/tdfy_dit/modules/attention/modules.py']}},
        initial_all_parameters_sha256=old['final_all_parameters_sha256'],
        restored_optimizer_sha256=expected_opt,common_parameter_names=common_names,
        additional_parameter_names=[n for n,p in extra] if args.arm=='shape_path' else [],
        trainable_counts={g['name']:sum(p.numel() for p in g['params']) for g in optimizer.param_groups},
        dropout_schedule_sha256=schedule_digest(schedule),input_batches=[],target_cache_sha256={},
        target_frames=old['target_frames'],target_quality=old['target_quality'],training=[],assessments=[],
        rows=[],artifacts=[],interface_checks={},complete=False,
        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(device)),
        scope='Matched 1000 additional updates from the same checkpoint. Shared optimizer moments retained. '
        'Only trainable generator scope changes; unchanged VecSetX, inputs, targets and layout attention protection.')
    args.output_dir.mkdir(parents=True)
    def save():(args.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    data=load_data_config(data_path);data['dataset']['split']='train'
    ds=build_dataloader(data,4,0,shuffle=False,include_touch=True,oracle_point_frame=False).dataset
    groups=choose_views(ds.records,single['sample_ids'],count=6)
    frames={f['sample_id']:f for f in old['target_frames']}
    prepared=[];target_supports=[]
    for group,records in enumerate(groups):
        ds.records=records;batch=collate_touch_batch([ds[i] for i in range(4)])
        targets,ca,kw,xyz,mask=prepare_batch(pipeline,batch,device,'bf16',True,False,False)
        assert len(ca)==1 and not kw and mask.all() and xyz.shape[1]==8192
        with torch.no_grad(),amp(device,'bf16'):
            pp,mm,_,_=encoder.prepare_points(xyz,mask)
            features=encoder.encoder.encode(pp,mm)['x'].detach()
        meta=dict(group=group,split='fit' if group<4 else 'reserved_view',
            sample_ids=[r['sample_id'] for r in records],image_sha256=tensor_digest(batch['image']),
            pointmap_sha256=tensor_digest(batch['pointmap']),target_sha256=tensor_digest(targets['shape']),
            features_sha256=tensor_digest(features))
        assert meta=={k:v for k,v in old['input_batches'][group].items() if k!='shared_target_sha256'}
        latent=[]
        for sid in meta['sample_ids']:
            path=args.fit_dir/f'target_{sid}.npz'
            with np.load(path,allow_pickle=False) as z:
                flat=z['latent'];assert flat.shape==(4096,8) and np.isfinite(flat).all()
                assert hashlib.sha256(flat.tobytes()).hexdigest()==frames[sid]['target_sha256']
                np.testing.assert_array_equal(z['object_from_output'],np.array(frames[sid]['object_from_output']))
                latent.append(torch.from_numpy(flat.copy()))
            report['target_cache_sha256'][sid]=sha(path)
        targets['shape']=torch.stack(latent).to(device)
        meta['shared_target_sha256']=tensor_digest(targets['shape'])
        assert meta==old['input_batches'][group];report['input_batches'].append(meta)
        prepared.append((targets,ca[0],features))
        with np.load(args.fit_dir/f'correct_surface_g{group}_d0.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['target'],targets['shape'].cpu().numpy())
            target_supports.append(z['target_occupancy'].copy())
    # Exact cached-array provenance is appropriate here: there is no floating
    # point recomputation. Anchor labels to the already returned geometry bundle.
    with zipfile.ZipFile(args.fit_dir/'shared_orientation_bundle.zip') as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        members={f['archive_path']:f for f in manifest['files']}
        assert json.loads(z.read('shared_orientation/results.json'))==old
        for group,q in enumerate(target_supports):
            name=f'shared_orientation/correct_surface_g{group}_d0/target_occupancy.npy'
            payload=z.read(name)
            assert hashlib.sha256(payload).hexdigest()==members[name]['sha256']
            np.testing.assert_array_equal(q,np.load(io.BytesIO(payload),allow_pickle=False))
    original_support=np.load(args.fit_dir/'original_target_occupancy.npy',allow_pickle=False)
    assert hashlib.sha256(original_support.tobytes()).hexdigest()==paired['target_support_content_sha256']
    gen=pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional==gen.self_consistency_prob==gen.fm_eps_max==0
    assert gen.loss_weights['shape']==1 and all(v==0 for k,v in gen.loss_weights.items() if k!='shape')

    def loss_at(group,drop=False,wrong=False):
        targets,visual,features=prepared[group]
        tokens=encoder.output_projection(features)+encoder.touch_embedding
        if wrong:tokens=tokens.roll(1,0)
        if drop:visual=torch.zeros_like(visual)
        key='visual_zero' if drop else 'visual_present'
        def check(module,inputs):
            torch.testing.assert_close(inputs[0],torch.cat((visual,tokens.to(visual)),1).to(inputs[0]),rtol=0,atol=0)
            report['interface_checks'][key]=True
        hook=None
        if key not in report['interface_checks']:
            hook=pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check)
        try:value,_=gen.loss(targets,visual,touch_tokens=tokens)
        finally:
            if hook is not None:hook.remove()
        assert torch.isfinite(value)
        return value

    def assess(step,groups_to_check=range(7)):
        py=random.getstate();rows=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad(),amp(device,'bf16'):
            gen.reverse_fn.training=True
            for group in groups_to_check:
                for wrong in (False,True):
                    values=[]
                    for draw in range(8):
                        torch.manual_seed(400000+29+draw);random.seed(400000+29+draw)
                        values.append(float(loss_at(group,wrong=wrong)))
                    rows.append(dict(group=group,wrong=wrong,losses=values))
        random.setstate(py);report['assessments'].append(dict(step=step,rows=rows));save()
        return rows

    replay=assess(1000,groups_to_check=[0])
    expected=[r for r in old['assessments'][-1]['rows'] if r['group']==0]
    actual=np.array([r['losses'] for r in replay]);expected_values=np.array([r['losses'] for r in expected])
    report['resume_replay']=dict(max_absolute_error=float(np.max(np.abs(actual-expected_values))),
        max_relative_error=float(np.max(np.abs(actual-expected_values)/np.maximum(np.abs(expected_values),1e-12))),
        rtol=1e-3,atol=1e-6)
    save();np.testing.assert_allclose(actual,expected_values,rtol=1e-3,atol=1e-6)
    print('Resume verified. Trainable groups:',report['trainable_counts'],flush=True)
    started=time.monotonic()
    for local_step in range(1,1001):
        step=1000+local_step;group=(local_step-1)%4
        torch.manual_seed(next_noise_seed(local_step));random.seed(next_noise_seed(local_step))
        optimizer.zero_grad(set_to_none=True);gen.reverse_fn.training=True
        with amp(device,'bf16'):loss=loss_at(group,drop=schedule[local_step-1])
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        extra_norm=None
        if local_step==1 or local_step%20==0:
            norms=[p.grad.detach().float().norm() for _,p in extra if p.grad is not None]
            extra_norm=float(torch.stack(norms).norm()) if norms else 0.
        norm=torch.nn.utils.clip_grad_norm_(optimized,1.,error_if_nonfinite=True)
        if local_step==1 and args.arm=='shape_path':assert extra_norm>0
        optimizer.step()
        report['training'].append(dict(step=step,group=group,visual_dropped=schedule[local_step-1],loss=float(loss),
            preclip_gradient_norm=float(norm),additional_gradient_norm=extra_norm))
        if local_step%20==0:
            print(args.arm,step,'loss',float(loss),'seconds',round(time.monotonic()-started,1),flush=True);save()
        if local_step in (300,1000):
            assess(step)
            torch.save(dict(model=trainable_state_dict(model),optimizer=optimizer.state_dict(),step=step,
                settings=report['settings'],conditioning_config=model.conditioning_config,
                parent_report_sha256=report['parent_report_sha256']),args.output_dir/f'checkpoint_{step}.pt')
    report['training_seconds']=time.monotonic()-started
    assert parameter_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_start
    report['frozen_parameters_unchanged']=True
    report['additional_parameters_changed']=parameter_digest(extra)!=extra_start
    assert report['additional_parameters_changed']==(args.arm=='shape_path')
    final_hash=parameter_digest(model.named_parameters());report['final_all_parameters_sha256']=final_hash
    save()
    decoder=pipeline.init_ss_decoder(cfg.ss_decoder_config_path,cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    gen.no_shortcut=True;gen.inference_steps=25;gen.rescale_t=float(cfg.get('ss_rescale_t',3))
    gen.reverse_fn.interval=list(cfg.get('ss_cfg_interval',[0,500]));gen.reverse_fn.strength=0;gen.reverse_fn.training=False
    with torch.no_grad():
        for group,(targets,visual,features) in enumerate(prepared):
            # Reuse the already verified decoded labels; cached latents matched
            # their original NPZs above. Do not regenerate or change the labels.
            for condition in ('correct_surface','wrong_surface'):
                with amp(device,'bf16'):tokens=encoder.output_projection(features)+encoder.touch_embedding
                if condition=='wrong_surface':tokens=tokens.roll(1,0)
                for draw in range(2):
                    torch.manual_seed(200000+29+draw);noise=gen._generate_x0(targets)
                    for i,sid in enumerate(report['input_batches'][group]['sample_ids']):
                        assert {n:tensor_digest(v[i]) for n,v in noise.items()}==paired['noise'][sid.rsplit('_',1)[0]+'/'+str(draw)]
                    with amp(device,'bf16'),patch.object(gen,'_generate_noise',side_effect=lambda *a,**k:{n:v.clone() for n,v in noise.items()}):
                        pred=gen({n:tuple(v.shape) for n,v in targets.items()},device,visual,touch_tokens=tokens)['shape']
                    assert torch.isfinite(pred).all();supports=[]
                    for i,sid in enumerate(report['input_batches'][group]['sample_ids']):
                        with amp(device,'bf16'):p=occupancy(decoder,pred[i:i+1])
                        supports.append(p);mapped=transform_points(support_points(p),np.array(frames[sid]['object_from_output']))
                        report['rows'].append(dict(condition=condition,group=group,draw=draw,sample_id=sid,
                            object_id=sid.rsplit('_',1)[0],split='fit' if group<4 else 'reserved_view',
                            noise_sha256={n:tensor_digest(v[i]) for n,v in noise.items()},
                            latent_mse=float((pred[i].float()-targets['shape'][i].float()).square().mean()),
                            **geometry_metrics(p,target_supports[group][i]),
                            common_object_units=proximity(mapped,support_points(original_support[i]))))
                    name=f'{condition}_g{group}_d{draw}.npz'
                    np.savez_compressed(args.output_dir/name,prediction=pred.float().cpu().numpy(),
                        target=targets['shape'].cpu().numpy(),predicted_occupancy=np.stack(supports),target_occupancy=target_supports[group])
                    report['artifacts'].append(dict(file=name,condition=condition,group=group,draw=draw,
                        sample_ids=report['input_batches'][group]['sample_ids']))
            save();print(args.arm,'sampling group',group,'done',flush=True)
    assert parameter_digest(model.named_parameters())==final_hash
    assert len(report['rows'])==112 and len(report['artifacts'])==28
    report['parameters_unchanged_during_sampling']=True;report['complete']=True
    result=args.output_dir/'results.json';result.write_text(json.dumps(report,indent=2)+'\n')
    prefix='scope_'+args.arm
    arrays={f'{prefix}/{Path(a["file"]).stem}':args.output_dir/a['file'] for a in report['artifacts']}
    bundle_reports={f'{prefix}/results.json':result,'reference/parent_results.json':old_path,
        'reference/original_target_occupancy.npy':args.fit_dir/'original_target_occupancy.npy',
        **{f'physical/{sid}.npy':args.fit_dir/f'physical_{sid}.npy' for sid in frames}}
    bundle=args.output_dir/f'{prefix}_bundle.zip'
    write_bundle(bundle,bundle_reports,arrays,dict(format_version=4,scope=report['scope'],cases=[dict(
        kind=prefix,prefix=f'{prefix}/{Path(a["file"]).stem}',**a) for a in report['artifacts']]))
    print('Return',bundle,flush=True)


if __name__=='__main__':main()
