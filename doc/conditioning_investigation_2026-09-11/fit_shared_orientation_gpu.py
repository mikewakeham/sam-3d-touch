"""Short target-frame intervention; unchanged camera conditioning and VecSetX.

Regenerate and validate labels before training. Original targets/data are never
overwritten. Failures before training leave a diagnostic partial report.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import build_dataloader,collate_touch_batch,load_data_config
from train import (TouchTrainingModel,amp,build_optimizer,build_stage1_pipeline,
                   prepare_batch,trainable_state_dict)
from tiny_fit_gpu import parameter_digest
from probe_tiny_fit_views import choose_views
from visual_dropout_protocol import dropout_schedule,schedule_digest
from rollout_gpu import occupancy,geometry_metrics
from experiments.noisy_target.evaluate import tensor_digest
from pose_shape_geometry import points as support_points,metrics as proximity
from shared_orientation_protocol import camera_oriented_target,transform_points
from bundle_alignment_geometry import write_bundle


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--encoder-checkpoint',type=Path,default=Path('checkpoints/hf/ss_encoder.ckpt'))
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    refs=[HERE/'tiny_fit_returned_46083371/camera/results.json',
          HERE/'multiple_view_returned_46083371/camera.json',
          HERE/'visual_dropout_returned_manual/camera/results.json',
          HERE/'camera_dropout_sampling_reference.json']
    single,multi,drop,paired=[json.loads(p.read_text()) for p in refs]
    for n,h in drop['source_sha256'].items():assert sha(REPO/n)==h,n
    assert drop['reference_sha256']=={'single':sha(refs[0]),'multiple':sha(refs[1])}
    pipeline_path=Path(single['settings']['pipeline_config']);data_path=Path(single['settings']['data_config'])
    assert pipeline_path.read_text()==single['pipeline_yaml'] and data_path.read_text()==single['data_yaml']
    cfg=OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent/cfg.ss_generator_config_path).read_text()==single['generator_yaml']
    data=load_data_config(data_path);data['dataset']['split']='train'
    ds=build_dataloader(data,4,0,shuffle=False,include_touch=True,oracle_point_frame=False).dataset
    groups=choose_views(ds.records,single['sample_ids'],count=6)
    assert len(groups)==7
    for group,records in enumerate(groups):
        assert [r['sample_id'] for r in records]==drop['input_batches'][group]['sample_ids']
        for r in records:
            for p in [ds.root/'objects'/r['object_id']/'model.obj',
                      ds.resolve_path(r['object_transform_path']),ds.resolve_path(r['camera_path'])]:
                if not p.is_file():raise FileNotFoundError(p)
    if not args.encoder_checkpoint.is_file():raise FileNotFoundError(args.encoder_checkpoint)
    generator_source=REPO/'data_generation/objaverse-dexonomy/generate_target_latents.py'
    spec=importlib.util.spec_from_file_location('target_generation',generator_source)
    target_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(target_module)
    schedule=dropout_schedule(29)
    assert schedule_digest(schedule)==drop['dropout_schedule_sha256']
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    report=dict(settings=dict(arm='camera_shared_orientation',seed=29,steps=1000,precision='bf16',
        fit_objects=4,fit_views=4,reserved_views=3,visual_dropout_fraction=.5,
        target_convention='camera_axes_centered_unit_bbox',sampling_steps=25,cfg=0),
        reference_sha256={str(p.relative_to(HERE)):sha(p) for p in refs},
        source_sha256={**drop['source_sha256'],**{str(p.relative_to(REPO)):sha(p) for p in
            [Path(__file__),HERE/'shared_orientation_protocol.py',generator_source,
             HERE/'pose_shape_geometry.py',HERE/'rollout_gpu.py',HERE/'bundle_alignment_geometry.py']}},
        encoder_checkpoint_sha256=sha(args.encoder_checkpoint),dropout_schedule_sha256=schedule_digest(schedule),
        target_checks=[],target_frames=[],input_batches=[],original_loss_replay=[],
        assessments=[],training=[],rows=[],artifacts=[],interface_checks={},complete=False,
        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(device)),
        scope='Only labels change. Same camera VecSetX features/visuals/initialization/update budget as camera dropout. '
              'Four fitted identities; no new-object or production-frame commitment. GT transforms only label construction and diagnostic scoring.')
    args.output_dir.mkdir(parents=True)
    def save():
        (args.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    # Target encoder is freed before the flow model is allocated. No interpolation
    # of latent channels, occupied-center rotation, or silent mesh clipping.
    vae=target_module.load_encoder(args.encoder_checkpoint,device).requires_grad_(False)
    meshes={};new_targets={};physical={};frames={}
    with torch.inference_mode():
        for record in groups[0]:
            oid=record['object_id'];mp=ds.root/'objects'/oid/'model.obj'
            tp=ds.resolve_path(record['object_transform_path'])
            mesh=target_module.load_normalized_mesh(mp,tp);meshes[oid]=mesh
            grid=target_module.voxelize_mesh(mesh)
            mean=vae(grid[None].to(device))['mean'][0].float().cpu().numpy()
            flat=np.ascontiguousarray(mean.transpose(1,2,3,0).reshape(4096,8))
            original=ds.load_target(ds.resolve_path(record['target_path']))
            np.testing.assert_allclose(flat,original,rtol=1e-4,atol=1e-5)
            report['target_checks'].append(dict(object_id=oid,mesh_sha256=sha(mp),transform_sha256=sha(tp),
                original_regeneration_max_error=float(np.max(np.abs(flat-original)))))
        for group,records in enumerate(groups):
            for record in records:
                sid=record['sample_id'];mesh=meshes[record['object_id']].copy()
                cp=ds.resolve_path(record['camera_path'])
                with np.load(cp,allow_pickle=False) as cam:
                    transform=np.diag([-1.,-1.,1.,1.])@cam['T_camera_from_object']
                vertices,frame=camera_oriented_target(mesh.vertices,transform)
                mesh.vertices=vertices
                # Only the same 1e-6 boundary inset as original target generation
                # is allowed; rotations outside the cube were normalized above.
                assert np.abs(vertices).max()<=.5+1e-10
                grid=target_module.voxelize_mesh(mesh)
                mean=vae(grid[None].to(device))['mean'][0].float().cpu().numpy()
                flat=np.ascontiguousarray(mean.transpose(1,2,3,0).reshape(4096,8))
                assert np.isfinite(flat).all()
                new_targets[sid]=torch.from_numpy(flat);physical[sid]=grid[0].numpy().astype(bool)
                frames[sid]=frame
                np.savez_compressed(args.output_dir/f'target_{sid}.npz',latent=flat,
                    physical_occupancy=physical[sid],object_from_output=np.array(frame['object_from_output']))
                np.save(args.output_dir/f'physical_{sid}.npy',physical[sid])
                report['target_frames'].append(dict(sample_id=sid,group=group,**frame,camera_sha256=sha(cp),
                    target_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),latent_mean_square=float(np.mean(flat**2))))
            save();print('Target preparation group',group,'done',flush=True)
    del vae,meshes;gc.collect();torch.cuda.empty_cache()
    # Reset after VAE initialization so the adapted model exactly matches history.
    random.seed(29);np.random.seed(29);torch.manual_seed(29)
    pipeline=build_stage1_pipeline(pipeline_path,device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder=TouchEncoder(output_dim=pipeline.backbone.cond_channels,trainable=False,use_position=False).to(device).eval()
    model=TouchTrainingModel(pipeline.ss_generator,encoder,False,False)
    optimizer,optimized=build_optimizer(encoder,pipeline.backbone,SimpleNamespace(
        learning_rate=1e-4,cross_attention_learning_rate=1e-5,cross_attention_scope='full'))
    report['initial_all_parameters_sha256']=parameter_digest(model.named_parameters())
    assert report['initial_all_parameters_sha256']==multi['initial_all_parameters_sha256']
    gen=pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional==gen.self_consistency_prob==gen.fm_eps_max==0
    assert gen.loss_weights['shape']==1 and all(v==0 for k,v in gen.loss_weights.items() if k!='shape')
    prepared=[]
    for group,records in enumerate(groups):
        ds.records=records;batch=collate_touch_batch([ds[i] for i in range(4)])
        original,ca,kw,xyz,mask=prepare_batch(pipeline,batch,device,'bf16',True,False,False)
        assert len(ca)==1 and not kw and mask.all() and xyz.shape[1]==8192
        with torch.no_grad(),amp(device,'bf16'):
            pp,mm,_,_=encoder.prepare_points(xyz,mask)
            features=encoder.encoder.encode(pp,mm)['x'].detach()
        meta=dict(group=group,split='fit' if group<4 else 'reserved_view',
            sample_ids=[r['sample_id'] for r in records],image_sha256=tensor_digest(batch['image']),
            pointmap_sha256=tensor_digest(batch['pointmap']),target_sha256=tensor_digest(original['shape']),
            features_sha256=tensor_digest(features))
        assert meta==drop['input_batches'][group]==multi['input_batches'][group]
        targets=dict(original);targets['shape']=torch.stack([new_targets[s] for s in meta['sample_ids']]).to(device)
        report['input_batches'].append(meta|dict(shared_target_sha256=tensor_digest(targets['shape'])))
        prepared.append((targets,ca[0],features,original))
    gen.reverse_fn.training=True
    with torch.no_grad(),amp(device,'bf16'):
        _,visual,features,original=prepared[0]
        tokens=encoder.output_projection(features)+encoder.touch_embedding
        for draw in range(8):
            torch.manual_seed(100000+29+draw);random.seed(100000+29+draw)
            loss,_=gen.loss(original,visual,touch_tokens=tokens)
            report['original_loss_replay'].append(float(loss))
    np.testing.assert_array_equal(report['original_loss_replay'],multi['assessments'][0]['rows'][0]['fresh_noise_native_losses'])
    decoder=pipeline.init_ss_decoder(cfg.ss_decoder_config_path,cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    with torch.no_grad(),amp(device,'bf16'):
        original_support=np.stack([occupancy(decoder,prepared[0][3]['shape'][i:i+1]) for i in range(4)])
    assert hashlib.sha256(original_support.tobytes()).hexdigest()==paired['target_support_content_sha256']
    np.save(args.output_dir/'original_target_occupancy.npy',original_support)
    target_supports=[];quality=[]
    with torch.no_grad():
        for group,(targets,_,_,_) in enumerate(prepared):
            supports=[]
            for i,sid in enumerate(report['input_batches'][group]['sample_ids']):
                with amp(device,'bf16'):support=occupancy(decoder,targets['shape'][i:i+1])
                supports.append(support)
                roundtrip=geometry_metrics(support,physical[sid])
                mapped=transform_points(support_points(support),np.array(frames[sid]['object_from_output']))
                common=proximity(mapped,support_points(original_support[i]))
                quality.append(dict(sample_id=sid,roundtrip=roundtrip,common_object_units=common))
            target_supports.append(np.stack(supports))
    report['target_quality']=quality;save()
    # Fail before optimizer updates if the new target representation itself loses
    # material geometry. Return the partial report; do not relax the gate blindly.
    good=all(q['roundtrip']['voxel_iou']>=.95 and min(q['common_object_units']['precision_1v'],
             q['common_object_units']['recall_1v'])>=.95 for q in quality)
    if not good:raise RuntimeError('Shared-target quality gate failed before training; return results.partial.json')
    report['target_quality_gate_passed']=True
    assert parameter_digest(model.named_parameters())==report['initial_all_parameters_sha256']
    print('Original replay, unchanged camera inputs, and all target-quality gates passed. Starting short fit.',flush=True)

    def loss_at(group,visual_drop=False,wrong=False):
        targets,visual,features,_=prepared[group]
        tokens=encoder.output_projection(features)+encoder.touch_embedding
        if wrong:tokens=tokens.roll(1,0)
        visual=torch.zeros_like(visual) if visual_drop else visual
        key='visual_zero' if visual_drop else 'visual_present'
        seen=[]
        def check(module,inputs):
            expected=torch.cat((visual,tokens.to(visual)),1).to(inputs[0])
            torch.testing.assert_close(inputs[0],expected,rtol=0,atol=0);seen.append(True)
        hook=None
        if key not in report['interface_checks']:
            hook=pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check)
        try:loss,_=gen.loss(targets,visual,touch_tokens=tokens)
        finally:
            if hook is not None:hook.remove()
        if hook is not None:
            assert seen;report['interface_checks'][key]=True
        assert torch.isfinite(loss)
        return loss

    def assess(step):
        py=random.getstate();rows=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad(),amp(device,'bf16'):
            gen.reverse_fn.training=True
            for group in range(7):
                for wrong in (False,True):
                    values=[]
                    for draw in range(8):
                        torch.manual_seed(400000+29+draw);random.seed(400000+29+draw)
                        values.append(float(loss_at(group,wrong=wrong)))
                    rows.append(dict(group=group,wrong=wrong,losses=values))
        random.setstate(py);report['assessments'].append(dict(step=step,rows=rows));save()
    assess(0)
    for step in range(1,1001):
        group=(step-1)%4;visual_drop=schedule[step-1]
        torch.manual_seed(29+step);random.seed(29+step)
        optimizer.zero_grad(set_to_none=True)
        with amp(device,'bf16'):loss=loss_at(group,visual_drop=visual_drop)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(optimized,1.,error_if_nonfinite=True)
        optimizer.step()
        report['training'].append(dict(step=step,group=group,visual_dropped=visual_drop,
                                       loss=float(loss),preclip_gradient_norm=float(norm)))
        if step%20==0:print('Shared orientation',step,float(loss),flush=True)
        if step in (300,1000):
            assess(step)
            torch.save(dict(model=trainable_state_dict(model),optimizer=optimizer.state_dict(),step=step,
                settings=report['settings'],conditioning_config=model.conditioning_config,
                target_convention='camera_axes_centered_unit_bbox',source_sha256=report['source_sha256']),
                args.output_dir/f'checkpoint_{step}.pt')
    final_hash=parameter_digest(model.named_parameters());report['final_all_parameters_sha256']=final_hash
    assert final_hash!=report['initial_all_parameters_sha256']
    gen.no_shortcut=True;gen.inference_steps=25;gen.rescale_t=float(cfg.get('ss_rescale_t',3))
    gen.reverse_fn.interval=list(cfg.get('ss_cfg_interval',[0,500]));gen.reverse_fn.strength=0;gen.reverse_fn.training=False
    with torch.no_grad():
        for condition in ('correct_surface','wrong_surface'):
            for group,(targets,visual,features,_) in enumerate(prepared):
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
                        with amp(device,'bf16'):support=occupancy(decoder,pred[i:i+1])
                        supports.append(support)
                        inverse=np.array(frames[sid]['object_from_output'])
                        mapped=transform_points(support_points(support),inverse)
                        report['rows'].append(dict(condition=condition,group=group,draw=draw,sample_id=sid,
                            object_id=sid.rsplit('_',1)[0],split='fit' if group<4 else 'reserved_view',
                            noise_sha256={n:tensor_digest(v[i]) for n,v in noise.items()},
                            latent_mse=float((pred[i].float()-targets['shape'][i].float()).square().mean()),
                            **geometry_metrics(support,target_supports[group][i]),
                            common_object_units=proximity(mapped,support_points(original_support[i]))))
                    name=f'{condition}_g{group}_d{draw}.npz'
                    np.savez_compressed(args.output_dir/name,prediction=pred.float().cpu().numpy(),
                        target=targets['shape'].cpu().numpy(),predicted_occupancy=np.stack(supports),
                        target_occupancy=target_supports[group])
                    report['artifacts'].append(dict(file=name,condition=condition,group=group,draw=draw,
                        sample_ids=report['input_batches'][group]['sample_ids']))
                save();print('Sampling',condition,'group',group,'done',flush=True)
    assert parameter_digest(model.named_parameters())==final_hash
    assert len(report['rows'])==112 and len(report['artifacts'])==28
    report['parameters_unchanged_during_sampling']=True;report['complete']=True
    result=args.output_dir/'results.json';result.write_text(json.dumps(report,indent=2)+'\n')
    arrays={f'shared_orientation/{Path(a["file"]).stem}':args.output_dir/a['file'] for a in report['artifacts']}
    bundle_reports={'shared_orientation/results.json':result,
        'reference/original_target_occupancy.npy':args.output_dir/'original_target_occupancy.npy',
        **{f'physical/{sid}.npy':args.output_dir/f'physical_{sid}.npy' for sid in physical}}
    write_bundle(args.output_dir/'shared_orientation_bundle.zip',bundle_reports,
        arrays,dict(format_version=3,scope=report['scope'],cases=[dict(kind='shared_orientation',
            prefix=f'shared_orientation/{Path(a["file"]).stem}',**a) for a in report['artifacts']]))
    print('Return',args.output_dir/'shared_orientation_bundle.zip',flush=True)


if __name__=='__main__':main()
