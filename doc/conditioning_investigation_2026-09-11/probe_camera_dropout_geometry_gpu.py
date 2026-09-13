"""Sample the existing camera/dropout checkpoint under the matched oracle protocol."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from omegaconf import OmegaConf
from bundle_alignment_geometry import write_bundle
from dataloader import build_dataloader,collate_touch_batch,load_data_config
from experiments.noisy_target.evaluate import tensor_digest
from probe_alignment_tolerance_gpu import locate_dropout,sha
from probe_tiny_fit_views import choose_views
from rollout_gpu import occupancy,geometry_metrics
from tiny_fit_gpu import parameter_digest
from train import TouchTrainingModel,amp,build_optimizer,build_stage1_pipeline,prepare_batch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    paths=[HERE/'tiny_fit_returned_46083371/camera/results.json',
           HERE/'multiple_view_returned_46083371/camera.json',
           HERE/'visual_dropout_returned_manual/camera/results.json',
           HERE/'camera_dropout_sampling_reference.json']
    single,multi,drop,paired=[json.loads(p.read_text()) for p in paths]
    assert drop['complete'] and drop['settings']['arm']=='camera' and drop['settings']['steps']==1000
    assert drop['reference_sha256']=={'single':sha(paths[0]),'multiple':sha(paths[1])}
    assert multi['driver_sha256']==sha(HERE/'fit_multiple_views_gpu.py')
    assert multi['view_selector_sha256']==sha(HERE/'probe_tiny_fit_views.py')
    for name,value in drop['source_sha256'].items():assert sha(REPO/name)==value,name
    fit_dir=locate_dropout(args.fit_dir,drop);checkpoint=fit_dir/'checkpoint_1000.pt'
    pipeline_path=Path(single['settings']['pipeline_config']);data_path=Path(single['settings']['data_config'])
    assert pipeline_path.read_text()==single['pipeline_yaml'] and data_path.read_text()==single['data_yaml']
    cfg=OmegaConf.load(pipeline_path)
    assert (pipeline_path.parent/cfg.ss_generator_config_path).read_text()==single['generator_yaml']
    random.seed(29);np.random.seed(29);torch.manual_seed(29)
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    pipeline=build_stage1_pipeline(pipeline_path,device)
    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    encoder=TouchEncoder(output_dim=pipeline.backbone.cond_channels,trainable=False,use_position=False).to(device).eval()
    model=TouchTrainingModel(pipeline.ss_generator,encoder,False,False)
    optimizer,_=build_optimizer(encoder,pipeline.backbone,SimpleNamespace(
        learning_rate=1e-4,cross_attention_learning_rate=1e-5,cross_attention_scope='full'))
    del optimizer
    assert parameter_digest(model.named_parameters())==multi['initial_all_parameters_sha256']
    parameters=dict(model.named_parameters());names={n for n,p in parameters.items() if p.requires_grad}
    archive=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert archive['step']==1000 and archive['settings']==drop['settings']
    assert archive['conditioning_config']=={'no_pointmap':False,'oracle_point_frame':False}
    assert set(archive['model'])==names
    with torch.no_grad():
        for n,v in archive['model'].items():
            assert parameters[n].shape==v.shape;parameters[n].copy_(v)
    del archive
    assert parameter_digest(model.named_parameters())==drop['final_all_parameters_sha256']
    model.requires_grad_(False)
    gen=pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional==gen.self_consistency_prob==gen.fm_eps_max==0
    assert gen.loss_weights['shape']==1 and all(v==0 for k,v in gen.loss_weights.items() if k!='shape')
    data=load_data_config(data_path);data['dataset']['split']='train'
    ds=build_dataloader(data,4,0,shuffle=False,include_touch=True,oracle_point_frame=False).dataset
    groups=choose_views(ds.records,single['sample_ids'],count=6)
    source_files=[Path(__file__),HERE/'probe_alignment_tolerance_gpu.py',HERE/'rollout_gpu.py',
                  HERE/'tiny_fit_gpu.py',HERE/'bundle_alignment_geometry.py']
    report={'settings':{'model_key':'camera_dropout','seed':29,'precision':'bf16','steps':25,'cfg':0,
                        'sampling_bank':200000,'sampling_draws':2,'conditions':['correct_surface','wrong_surface']},
            'reference_sha256':{str(p.relative_to(HERE)):sha(p) for p in paths},
            'source_sha256':{**drop['source_sha256'],**{str(p.relative_to(REPO)):sha(p) for p in source_files}},
            'checkpoint':{'path':str(checkpoint),'sha256':sha(checkpoint),'parameters_sha256':drop['final_all_parameters_sha256']},
            'runtime':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device)},
            'input_batches':[],'replay':[],'rows':[],'artifacts':[],
            'scope':'No training. Same fitted objects and fit/reserved views, CFG0 and sampling noise as the completed oracle/dropout probe. '
                    'Actual camera-frame conditioning, no oracle transform. Wrong surfaces contradict the visual inputs.',
            'complete':False}
    args.output_dir.mkdir(parents=True)
    def save():
        (args.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    prepared=[]
    for group,records in enumerate(groups):
        ds.records=records;batch=collate_touch_batch([ds[i] for i in range(4)])
        targets,ca,kw,points,mask=prepare_batch(pipeline,batch,device,'bf16',True,False,False)
        assert len(ca)==1 and not kw and mask.all() and points.shape[1]==encoder.num_points==8192
        with torch.no_grad(),amp(device,'bf16'):
            pp,mm,_,_=encoder.prepare_points(points,mask)
            features=encoder.encoder.encode(pp,mm)['x']
            tokens=(encoder.output_projection(features)+encoder.touch_embedding).detach()
        entry=dict(group=group,split='fit' if group<4 else 'reserved_view',sample_ids=[r['sample_id'] for r in records],
                   image_sha256=tensor_digest(batch['image']),pointmap_sha256=tensor_digest(batch['pointmap']),
                   target_sha256=tensor_digest(targets['shape']),features_sha256=tensor_digest(features))
        assert entry==drop['input_batches'][group]==multi['input_batches'][group]
        for k,v in entry.items():
            if k!='features_sha256':assert v==paired['input_batches'][group][k]
        report['input_batches'].append(entry);prepared.append((targets,ca[0],tokens))
    gen.reverse_fn.training=True
    with torch.no_grad(),amp(device,'bf16'):
        for group,(targets,visual,tokens) in enumerate(prepared):
            for wrong in (False,True):
                values=[]
                for draw in range(8):
                    torch.manual_seed(400000+29+draw);random.seed(400000+29+draw)
                    loss,_=gen.loss(targets,visual,touch_tokens=tokens.roll(1,0) if wrong else tokens);values.append(float(loss))
                key='swapped_surface_native_losses' if wrong else 'fresh_noise_native_losses'
                np.testing.assert_allclose(values,drop['final_fresh']['visual_present'][group][key],rtol=1e-5,atol=1e-7)
                report['replay'].append(dict(group=group,wrong=wrong,losses=values))
    save();print('Historical checkpoint, inputs and native-loss replay passed.',flush=True)
    decoder=pipeline.init_ss_decoder(cfg.ss_decoder_config_path,cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
    with torch.no_grad(),amp(device,'bf16'):
        target_support=np.stack([occupancy(decoder,prepared[0][0]['shape'][i:i+1]) for i in range(4)])
    assert hashlib.sha256(target_support.tobytes()).hexdigest()==paired['target_support_content_sha256']
    report['target_support_content_sha256']=paired['target_support_content_sha256']
    assert all(torch.equal(p[0]['shape'],prepared[0][0]['shape']) for p in prepared)
    gen.no_shortcut=True;gen.inference_steps=25;gen.rescale_t=float(cfg.get('ss_rescale_t',3))
    gen.reverse_fn.interval=list(cfg.get('ss_cfg_interval',[0,500]));gen.reverse_fn.strength=0;gen.reverse_fn.training=False
    with torch.no_grad():
        for condition in report['settings']['conditions']:
            for group,(targets,visual,correct) in enumerate(prepared):
                tokens=correct.roll(1,0) if condition=='wrong_surface' else correct
                seen=[]
                def check_context(module,inputs):
                    expected=torch.cat((visual,tokens.to(visual)),1).to(inputs[0])
                    torch.testing.assert_close(inputs[0],expected,rtol=0,atol=0);seen.append(True)
                hook=pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(check_context)
                try:
                    for draw in range(2):
                        torch.manual_seed(200000+29+draw);noise=gen._generate_x0(targets)
                        for i,sid in enumerate(report['input_batches'][group]['sample_ids']):
                            hashes={n:tensor_digest(v[i]) for n,v in noise.items()}
                            assert hashes==paired['noise'][sid.rsplit('_',1)[0]+'/'+str(draw)]
                        with amp(device,'bf16'),patch.object(gen,'_generate_noise',side_effect=lambda *a,**k:{n:v.clone() for n,v in noise.items()}):
                            pred=gen({n:tuple(v.shape) for n,v in targets.items()},device,visual,touch_tokens=tokens)['shape']
                        assert torch.isfinite(pred).all();supports=[]
                        for i,sid in enumerate(report['input_batches'][group]['sample_ids']):
                            with amp(device,'bf16'):support=occupancy(decoder,pred[i:i+1])
                            supports.append(support)
                            report['rows'].append(dict(condition=condition,group=group,draw=draw,sample_id=sid,
                                object_id=sid.rsplit('_',1)[0],split='fit' if group<4 else 'reserved_view',
                                noise_sha256={n:tensor_digest(v[i]) for n,v in noise.items()},
                                latent_mse=float((pred[i].float()-targets['shape'][i].float()).square().mean()),
                                **geometry_metrics(support,target_support[i])))
                        name=f'{condition}_g{group}_d{draw}.npz'
                        np.savez_compressed(args.output_dir/name,prediction=pred.float().cpu().numpy(),
                            target=targets['shape'].cpu().numpy(),predicted_occupancy=np.stack(supports),target_occupancy=target_support)
                        report['artifacts'].append(dict(file=name,condition=condition,group=group,draw=draw,
                            sample_ids=report['input_batches'][group]['sample_ids']))
                finally:hook.remove()
                assert seen;save();print(condition,'group',group,'done',flush=True)
    assert parameter_digest(model.named_parameters())==drop['final_all_parameters_sha256']
    assert len(report['rows'])==112 and len(report['artifacts'])==28
    report['parameters_unchanged']=True;report['complete']=True
    result_path=args.output_dir/'results.json';result_path.write_text(json.dumps(report,indent=2)+'\n')
    arrays={f'camera_dropout/{Path(x["file"]).stem}':args.output_dir/x['file'] for x in report['artifacts']}
    metadata={'format_version':2,'scope':report['scope'],'cases':[dict(kind='camera_dropout',
        prefix=f'camera_dropout/{Path(x["file"]).stem}',**x) for x in report['artifacts']]}
    write_bundle(args.output_dir/'camera_dropout_geometry_bundle.zip',{'camera_dropout/results.json':result_path},arrays,metadata)
    print('Complete; return',args.output_dir/'camera_dropout_geometry_bundle.zip',flush=True)


if __name__=='__main__':main()
