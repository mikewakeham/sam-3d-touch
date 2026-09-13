"""Inference-only visual-present/visual-zero factorial on the broader checkpoint.

Same seven views, correct/wrong surfaces, two paired sampling draws. No optimizer
updates, target encoding, regenerated geometry or change to checkpoint parameters.
"""
import argparse
import hashlib
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
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.utils.checkpoint
from omegaconf import OmegaConf
from dataloader import build_dataloader,collate_touch_batch,load_data_config
from train import TouchTrainingModel,amp,build_optimizer,build_stage1_pipeline,prepare_batch
from tiny_fit_gpu import parameter_digest
from probe_tiny_fit_views import choose_views
from experiments.noisy_target.evaluate import tensor_digest
from shared_orientation_scope_protocol import additional_shape_parameter
from shared_orientation_protocol import transform_points
from pose_shape_geometry import points as support_points,metrics as proximity
from rollout_gpu import occupancy,geometry_metrics
from bundle_alignment_geometry import write_bundle


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    fit_path=args.fit_dir/'results.json';fit=json.loads(fit_path.read_text())
    assert fit['complete'] and fit['settings']['arm']=='shape_path'
    assert fit['settings']['base_step']==fit['settings']['additional_steps']==1000
    for name,value in fit['source_sha256'].items():assert sha(REPO/name)==value,name
    checkpoint=args.fit_dir/'checkpoint_2000.pt'
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    for group in range(7):
        for condition in ('correct_surface','wrong_surface'):
            for draw in range(2):
                path=args.fit_dir/f'{condition}_g{group}_d{draw}.npz'
                if not path.is_file():raise FileNotFoundError(path)
    single=json.loads((HERE/'tiny_fit_returned_46083371/camera/results.json').read_text())
    paired=json.loads((HERE/'camera_dropout_sampling_reference.json').read_text())
    pipeline_path=Path(single['settings']['pipeline_config']);data_path=Path(single['settings']['data_config'])
    assert pipeline_path.read_text()==single['pipeline_yaml'] and data_path.read_text()==single['data_yaml']
    cfg=OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent/cfg.ss_generator_config_path).read_text()==single['generator_yaml']
    # Read already validated labels/references from the completed bundle.
    target_supports=[];physical={}
    with zipfile.ZipFile(args.fit_dir/'scope_shape_path_bundle.zip') as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        assert manifest['format_version']==4
        members={r['archive_path']:r for r in manifest['files']}
        assert len(members)==len(manifest['files'])
        def payload(name):
            b=z.read(name);assert hashlib.sha256(b).hexdigest()==members[name]['sha256']
            return b
        assert json.loads(payload('scope_shape_path/results.json'))==fit
        original_support=np.load(io.BytesIO(payload('reference/original_target_occupancy.npy')),allow_pickle=False)
        parent_bytes=payload('reference/parent_results.json')
        assert hashlib.sha256(parent_bytes).hexdigest()==fit['parent_report_sha256']
        for g in range(7):
            target_supports.append(np.load(io.BytesIO(payload(f'scope_shape_path/correct_surface_g{g}_d0/target_occupancy.npy')),allow_pickle=False))
        for frame in fit['target_frames']:
            sid=frame['sample_id'];physical[sid]=payload(f'physical/{sid}.npy')
    assert hashlib.sha256(original_support.tobytes()).hexdigest()==paired['target_support_content_sha256']
    random.seed(29);np.random.seed(29);torch.manual_seed(29)
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    pipeline=build_stage1_pipeline(pipeline_path,device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder=TouchEncoder(output_dim=pipeline.backbone.cond_channels,trainable=False,use_position=False).to(device).eval()
    model=TouchTrainingModel(pipeline.ss_generator,encoder,False,False)
    # Reproduce requires_grad flags for the numerical execution path. The empty
    # optimizer is discarded; all subsequent model calls use torch.no_grad().
    empty_optimizer,_=build_optimizer(encoder,pipeline.backbone,SimpleNamespace(
        learning_rate=1e-4,cross_attention_learning_rate=1e-5,cross_attention_scope='full'))
    del empty_optimizer
    extras=[(n,p) for n,p in pipeline.backbone.named_parameters() if additional_shape_parameter(n)]
    assert [n for n,p in extras]==fit['additional_parameter_names']
    for _,p in extras:p.requires_grad_(True)
    assert len(pipeline.backbone.blocks)==24 and not pipeline.backbone.share_mod
    for b in pipeline.backbone.blocks:
        assert b.self_attn.protect_modality_list==['shape']
        b.use_checkpoint=True
    parameters=dict(model.named_parameters())
    archive=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    assert archive['step']==2000 and archive['settings']==fit['settings']
    assert archive['conditioning_config']==model.conditioning_config
    assert archive['parent_report_sha256']==fit['parent_report_sha256']
    assert set(archive['model'])=={n for n,p in parameters.items() if p.requires_grad}
    with torch.no_grad():
        for name,value in archive['model'].items():parameters[name].copy_(value)
    del archive
    initial=parameter_digest(model.named_parameters());assert initial==fit['final_all_parameters_sha256']
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'parent_results.json').write_bytes(parent_bytes)
    np.save(args.output_dir/'original_target_occupancy.npy',original_support)
    for sid,b in physical.items():(args.output_dir/f'physical_{sid}.npy').write_bytes(b)
    report=dict(settings=dict(model_key='shape_path_2000',fit_dir=str(args.fit_dir),seed=29,precision='bf16',training_updates=0,
        visual_modes=['present','zero'],conditions=['correct_surface','wrong_surface'],
        sampling_steps=25,cfg=0,sampling_bank=200000,sampling_draws=2),
        fit_report_sha256=sha(fit_path),checkpoint_sha256=sha(checkpoint),
        checkpoint_parameters_sha256=initial,
        source_sha256={**fit['source_sha256'],str(Path(__file__).relative_to(REPO)):sha(Path(__file__))},
        target_frames=fit['target_frames'],target_quality=fit['target_quality'],input_batches=[],
        native=[],rows=[],artifacts=[],interface_checks={},historical_sample_replay=[],complete=False,
        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(device)),
        scope='No training. Same shape_path step2000 model, inputs, cached labels and sampling noise. '
        'Whole-visual-zero matches the trained dropout operation; no touch-token removal. '
        'Four fitted identities, four fitted/three reserved views; no new-object claim.')
    def save():(args.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    save();started=time.monotonic()
    data=load_data_config(data_path);data['dataset']['split']='train'
    ds=build_dataloader(data,4,0,shuffle=False,include_touch=True,oracle_point_frame=False).dataset
    groups=choose_views(ds.records,single['sample_ids'],count=6);prepared=[]
    for group,records in enumerate(groups):
        ds.records=records;batch=collate_touch_batch([ds[i] for i in range(4)])
        targets,ca,kw,xyz,mask=prepare_batch(pipeline,batch,device,'bf16',True,False,False)
        assert len(ca)==1 and not kw and mask.all() and xyz.shape[1]==8192
        with torch.no_grad(),amp(device,'bf16'):
            pp,mm,_,_=encoder.prepare_points(xyz,mask)
            features=encoder.encoder.encode(pp,mm)['x'].detach()
        meta=dict(group=group,split='fit' if group<4 else 'reserved_view',sample_ids=[r['sample_id'] for r in records],
            image_sha256=tensor_digest(batch['image']),pointmap_sha256=tensor_digest(batch['pointmap']),
            target_sha256=tensor_digest(targets['shape']),features_sha256=tensor_digest(features))
        assert meta=={k:v for k,v in fit['input_batches'][group].items() if k!='shared_target_sha256'}
        with np.load(args.fit_dir/f'correct_surface_g{group}_d0.npz',allow_pickle=False) as z:
            targets['shape']=torch.from_numpy(z['target'].copy()).to(device)
            np.testing.assert_array_equal(z['target_occupancy'],target_supports[group])
        meta['shared_target_sha256']=tensor_digest(targets['shape']);assert meta==fit['input_batches'][group]
        report['input_batches'].append(meta);prepared.append((targets,ca[0],features))
    gen=pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional==gen.self_consistency_prob==gen.fm_eps_max==0
    assert gen.loss_weights['shape']==1 and all(v==0 for k,v in gen.loss_weights.items() if k!='shape')

    def context_hook(visual,tokens,key):
        def check(module,inputs):
            if key in report['interface_checks']:return
            expected=torch.cat((visual,tokens.to(visual)),1).to(inputs[0])
            torch.testing.assert_close(inputs[0],expected,rtol=0,atol=0)
            assert torch.count_nonzero(tokens).item()>0
            report['interface_checks'][key]=True
        return pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check)

    # Full factorial native comparison; natural inputs replay the completed run.
    with torch.no_grad(),amp(device,'bf16'):
        gen.reverse_fn.training=True
        for visual_mode in ('present','zero'):
            for group,(targets,visual,features) in enumerate(prepared):
                visual=torch.zeros_like(visual) if visual_mode=='zero' else visual
                for wrong in (False,True):
                    values=[]
                    for draw in range(8):
                        torch.manual_seed(400000+29+draw);random.seed(400000+29+draw)
                        tokens=encoder.output_projection(features)+encoder.touch_embedding
                        if wrong:tokens=tokens.roll(1,0)
                        key=f'native/{visual_mode}/{wrong}';hook=context_hook(visual,tokens,key)
                        try:value,_=gen.loss(targets,visual,touch_tokens=tokens)
                        finally:hook.remove()
                        assert torch.isfinite(value);values.append(float(value))
                    report['native'].append(dict(visual_mode=visual_mode,group=group,wrong=wrong,losses=values))
            save()
    actual=np.array([r['losses'] for r in report['native'] if r['visual_mode']=='present'])
    expected=np.array([r['losses'] for r in fit['assessments'][-1]['rows']])
    report['native_replay']=dict(max_absolute_error=float(np.max(np.abs(actual-expected))),
        max_relative_error=float(np.max(np.abs(actual-expected)/np.maximum(np.abs(expected),1e-12))),rtol=1e-3,atol=1e-6)
    save();np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-6)
    print('Checkpoint/input replay passed. Sampling all four visual/surface conditions.',flush=True)
    decoder=pipeline.init_ss_decoder(cfg.ss_decoder_config_path,cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    gen.no_shortcut=True;gen.inference_steps=25;gen.rescale_t=float(cfg.get('ss_rescale_t',3))
    gen.reverse_fn.interval=list(cfg.get('ss_cfg_interval',[0,500]));gen.reverse_fn.strength=0;gen.reverse_fn.training=False
    frames={f['sample_id']:f for f in fit['target_frames']}
    with torch.no_grad():
        for group,(targets,visual,features) in enumerate(prepared):
            for visual_mode in ('present','zero'):
                context=torch.zeros_like(visual) if visual_mode=='zero' else visual
                for condition in ('correct_surface','wrong_surface'):
                    with amp(device,'bf16'):tokens=encoder.output_projection(features)+encoder.touch_embedding
                    if condition=='wrong_surface':tokens=tokens.roll(1,0)
                    for draw in range(2):
                        torch.manual_seed(200000+29+draw);noise=gen._generate_x0(targets)
                        for i,sid in enumerate(report['input_batches'][group]['sample_ids']):
                            assert {n:tensor_digest(v[i]) for n,v in noise.items()}==paired['noise'][sid.rsplit('_',1)[0]+'/'+str(draw)]
                        key=f'sampled/{visual_mode}/{condition}';hook=context_hook(context,tokens,key)
                        try:
                            with amp(device,'bf16'),patch.object(gen,'_generate_noise',side_effect=lambda *a,**k:{n:v.clone() for n,v in noise.items()}):
                                pred=gen({n:tuple(v.shape) for n,v in targets.items()},device,context,touch_tokens=tokens)['shape']
                        finally:hook.remove()
                        assert key in report['interface_checks'] and torch.isfinite(pred).all();supports=[]
                        for i,sid in enumerate(report['input_batches'][group]['sample_ids']):
                            with amp(device,'bf16'):p=occupancy(decoder,pred[i:i+1])
                            supports.append(p)
                            report['rows'].append(dict(visual_mode=visual_mode,condition=condition,group=group,draw=draw,
                                sample_id=sid,object_id=sid.rsplit('_',1)[0],split='fit' if group<4 else 'reserved_view',
                                noise_sha256={n:tensor_digest(v[i]) for n,v in noise.items()},
                                latent_mse=float((pred[i].float()-targets['shape'][i].float()).square().mean()),
                                **geometry_metrics(p,target_supports[group][i]),common_object_units=proximity(
                                    transform_points(support_points(p),np.array(frames[sid]['object_from_output'])),support_points(original_support[i]))))
                        if visual_mode=='present':
                            with np.load(args.fit_dir/f'{condition}_g{group}_d{draw}.npz',allow_pickle=False) as z:
                                old=z['predicted_occupancy'];new=np.stack(supports)
                                report['historical_sample_replay'].append(dict(group=group,draw=draw,condition=condition,
                                    bitwise_support_equal=bool(np.array_equal(new,old)),
                                    per_object_iou_to_previous=[float(np.count_nonzero(p&q)/max(1,np.count_nonzero(p|q))) for p,q in zip(new,old)]))
                        name=f'{visual_mode}_{condition}_g{group}_d{draw}.npz'
                        np.savez_compressed(args.output_dir/name,prediction=pred.float().cpu().numpy(),target=targets['shape'].cpu().numpy(),
                            predicted_occupancy=np.stack(supports),target_occupancy=target_supports[group])
                        report['artifacts'].append(dict(file=name,visual_mode=visual_mode,condition=condition,group=group,draw=draw,
                            sample_ids=report['input_batches'][group]['sample_ids']))
                save();print('Sampled group',group,'visuals',visual_mode,flush=True)
    assert parameter_digest(model.named_parameters())==initial
    assert not any(p.grad is not None for p in model.parameters())
    assert len(report['interface_checks'])==8 and all(report['interface_checks'].values())
    assert len(report['rows'])==224 and len(report['artifacts'])==56
    report['parameters_unchanged']=True;report['elapsed_seconds']=time.monotonic()-started;report['complete']=True
    result=args.output_dir/'results.json';result.write_text(json.dumps(report,indent=2)+'\n')
    arrays={f'visual_probe/{Path(a["file"]).stem}':args.output_dir/a['file'] for a in report['artifacts']}
    refs={'visual_probe/results.json':result,'reference/fit_results.json':fit_path,
        'reference/parent_results.json':args.output_dir/'parent_results.json',
        'reference/original_target_occupancy.npy':args.output_dir/'original_target_occupancy.npy',
        **{f'physical/{sid}.npy':args.output_dir/f'physical_{sid}.npy' for sid in physical}}
    bundle=args.output_dir/'shared_orientation_visuals_bundle.zip'
    write_bundle(bundle,refs,arrays,dict(format_version=5,scope=report['scope'],cases=[dict(kind='visual_probe',
        prefix=f'visual_probe/{Path(a["file"]).stem}',**a) for a in report['artifacts']]))
    print('Return',bundle,flush=True)


if __name__=='__main__':main()
