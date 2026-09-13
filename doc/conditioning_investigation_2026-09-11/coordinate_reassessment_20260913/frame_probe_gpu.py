"""Finish the full-data camera/oracle comparison; no fitting or target changes."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = HERE.parent / 'oracle_upper_bound'
sys.path[:0] = [str(REPO), str(OLD)]

import numpy as np
import torch
import yaml
from dataloader import TouchDataset, collate_touch_batch
from evaluate import restore_run
from train import amp, build_stage1_pipeline, prepare_batch
from checkpoint_probe_gpu import make_bank, sha, write
from checkpoint_probe_core import condition_tokens, paired_loss, tensor_sha
from checkpoint_rollout_core import sample_from_noise, support_metrics
from checkpoint_rollout_gpu import decode_support
from frame_probe_core import check_observation, check_report_pair, check_loss_pair, normalization_check


def array(t):
    return t.detach().float().cpu().numpy()


def save_npz(folder, report, filename, **arrays):
    np.savez_compressed(folder / filename, **arrays)
    report['files_sha256'][filename] = sha(folder / filename)


def geometry_reference(args, datasets, references, report, device):
    """Mesh-space checks and target regeneration precede loading the generator."""
    import open3d as o3d
    import trimesh
    path = REPO / 'data_generation/objaverse-dexonomy/generate_target_latents.py'
    spec = importlib.util.spec_from_file_location('frame_target_reference', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report['geometry_versions'] = {'trimesh':trimesh.__version__, 'open3d':o3d.__version__}
    report['target_encoder_sha256'] = sha(args.encoder_checkpoint)
    report['target_generation_source_sha256'] = sha(path)
    encoder = module.load_encoder(args.encoder_checkpoint, device).requires_grad_(False)
    grouped = {}
    for b in references:
        ds = datasets[b['split']]
        lookup = {r['sample_id']:r for r in ds.records}
        for sid in b['sample_ids']:
            rec = lookup[sid]
            grouped.setdefault(rec['object_id'], []).append((ds,rec))
    for oi, (oid, examples) in enumerate(grouped.items()):
        ds, first = examples[0]
        mp = ds.root / 'objects' / oid / 'model.obj'
        tp = ds.resolve_path(first['object_transform_path'])
        mesh = module.load_normalized_mesh(mp, tp)
        grid = module.voxelize_mesh(mesh)
        with torch.no_grad():
            regenerated = encoder(grid[None].to(device))['mean'][0].float().cpu().numpy()
        flat = regenerated.transpose(1,2,3,0).reshape(4096,8)
        target = ds.load_target(ds.resolve_path(first['target_path']))
        target_ok = bool(np.allclose(flat,target,rtol=1e-4,atol=1e-5))
        obj = {'object_id':oid, 'mesh_sha256':sha(mp), 'object_transform_sha256':sha(tp),
               'target_file_sha256':sha(ds.resolve_path(first['target_path'])),
               'target_regeneration_max_error':float(np.abs(flat-target).max()),
               'target_regeneration_passed':target_ok, 'target_rtol':1e-4, 'target_atol':1e-5}
        report['geometry_objects'].append(obj)
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        # A known far translation is an independent negative control for distance queries.
        valid_faces=np.asarray(mesh.faces)[np.asarray(mesh.area_faces)>1e-15]
        assert len(valid_faces), 'Mesh has no nondegenerate triangles'
        control = np.asarray(mesh.vertices[np.unique(valid_faces[:128])],dtype=np.float32)
        faithful = scene.compute_distance(o3d.core.Tensor(control)).numpy()
        wrong = scene.compute_distance(o3d.core.Tensor(control+[4.,0.,0.],dtype=o3d.core.Dtype.Float32)).numpy()
        assert float(faithful.max()) < 5e-5 and float(wrong.min()) > 2
        obj['mesh_query_control_max'] = float(faithful.max())
        obj['translated_negative_control_min'] = float(wrong.min())
        with np.load(ds.resolve_path(first['full_surface_path']),allow_pickle=False) as sf:
            sampled, _ = trimesh.sample.sample_surface(mesh,int(sf['requested_point_count']),seed=int(sf['sample_seed']))
        save_npz(args.output,report,f'object_{oid}.npz',
                 mesh_sample_object=sampled, mesh_occupancy=grid[0].numpy().astype(bool),
                 regenerated_target_shape=flat, saved_target_shape=target)
        for ds, rec in examples:
            assert sha(ds.resolve_path(rec['object_transform_path'])) == obj['object_transform_sha256']
            assert np.array_equal(ds.load_target(ds.resolve_path(rec['target_path'])), target)
            with np.load(ds.resolve_path(rec['full_surface_path']),allow_pickle=False) as sf:
                camera = sf['points_camera'].copy(); point_ids = sf['point_ids'].copy()
            with np.load(ds.resolve_path(rec['camera_path']),allow_pickle=False) as cf:
                forward = np.diag([-1.,-1.,1.,1.]) @ cf['T_camera_from_object']
            inv = np.linalg.inv(forward)
            oracle = camera.astype(float) @ inv[:3,:3].T + inv[:3,3]
            closest = scene.compute_closest_points(o3d.core.Tensor(oracle.astype(np.float32)))
            on_mesh = closest['points'].numpy()
            triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[closest['primitive_ids'].numpy()]]
            distance = np.linalg.norm(oracle-on_mesh,axis=1)
            row = {'sample_id':rec['sample_id'], 'object_id':oid,
                   'surface_sha256':sha(ds.resolve_path(rec['full_surface_path'])),
                   'camera_sha256':sha(ds.resolve_path(rec['camera_path'])),
                   'point_to_mesh_max_distance':float(distance.max()),
                   'point_to_mesh_passed':bool(distance.max()<5e-5), 'distance_limit':5e-5,
                   'seed_replay_max_error':float(np.abs(oracle-sampled[point_ids]).max())}
            # Seed replay is descriptive: library/mesh-order changes can alter sampling.
            # Mesh membership plus target regeneration is the independent contract check.
            report['geometry_samples'].append(row)
            save_npz(args.output,report,f"geometry_{rec['sample_id']}.npz",
                     points_camera=camera, point_ids=point_ids, camera_from_object=forward,
                     oracle_points_float64=oracle, closest_points_object=on_mesh,
                     closest_triangles_object=triangles)
        write(args.output/'results.partial.json',report)
        print(f'Geometry/target reference {oi+1}/{len(grouped)} complete',flush=True)
    del encoder
    torch.cuda.empty_cache()
    report['coordinate_reference_passed'] = all(x['target_regeneration_passed'] for x in report['geometry_objects']) and all(x['point_to_mesh_passed'] for x in report['geometry_samples'])
    write(args.output/'results.partial.json',report)
    if not report['coordinate_reference_passed']:
        raise RuntimeError('Independent input/target reference failed; return results.partial.json before interpreting frame effects')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-root',type=Path,required=True)
    parser.add_argument('--rollout-root',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/conditioning_investigation/stage1_full_surface_dropout'))
    parser.add_argument('--encoder-checkpoint',type=Path,default=Path('checkpoints/hf/ss_encoder.ckpt'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=0)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output directory')
    ref_path = args.probe_root/'oracle/results.json'
    ref = json.loads(ref_path.read_text())
    roll_path = args.rollout_root/'oracle/results.json'
    roll = json.loads(roll_path.read_text())
    assert ref['complete'] and roll['complete'] and roll['reference_sha256'] == sha(ref_path)
    assert ref['checkpoint_sha256'] == roll['checkpoint_sha256']
    for name,digest in ref['source_sha256'].items():
        assert sha(REPO/name)==digest, f'Historical source changed: {name}'
    for name,digest in roll['additional_source_sha256'].items():
        assert sha(REPO/name)==digest, f'Historical rollout helper changed: {name}'
    config = yaml.safe_load((args.run_dir/'config.yaml').read_text())
    assert config['data'] == ref['data_config']
    pipeline_path = Path(config['arguments']['pipeline_config'])
    assert sha(pipeline_path) == ref['pipeline_sha256']
    checkpoint_path = args.run_dir/'last.pt'
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['step']==14660 and checkpoint['epoch']==20
    assert checkpoint['conditioning_config']=={'no_pointmap':False,'oracle_point_frame':False}
    assert checkpoint['training_config']=={'visual_dropout':.5,'constant_touch':False}
    assert checkpoint['touch_config']==ref['checkpoint_metadata']['touch_config']==config['touch_config']
    assert checkpoint['cross_attention_scope']=='full' and checkpoint['mode']=='image_touch'
    assert config['arguments']['oracle_point_frame'] is False
    assert config['arguments']['constant_touch'] is False and config['arguments']['no_pointmap'] is False
    oracle_config_path=Path(ref['settings']['run_dir'])/'config.yaml'
    assert sha(oracle_config_path)==ref['config_sha256']
    oracle_config=yaml.safe_load(oracle_config_path.read_text())
    for k in ('batch_size','epochs','learning_rate','cross_attention_learning_rate','precision','no_touch_position','visual_dropout','joint_pointmap'):
        assert config['arguments'][k] == oracle_config['arguments'][k], k
    assert config['global_batch_size']==oracle_config['global_batch_size']==16
    assert config['world_size']==oracle_config['world_size']==4
    torch.cuda.set_device(args.device_index); device=torch.device('cuda',args.device_index)
    datasets={}
    for split in ('train','val'):
        data=copy.deepcopy(config['data']);data['dataset']['split']=split
        # Load the oracle transform for auditing; training/sampling still uses camera mode.
        datasets[split]=TouchDataset(data,include_touch=True,oracle_point_frame=True)
    dataset_hashes={k:sha(datasets['train'].resolve_path(config['data']['dataset'][k])) for k in ('manifest','split_file')}
    assert dataset_hashes==ref['dataset_sha256']
    report={'complete':False,'settings':{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'arm':'camera'},
            'checkpoint_metadata':{k:checkpoint[k] for k in ref['checkpoint_metadata']},
            'checkpoint_sha256':sha(checkpoint_path),'config_sha256':sha(args.run_dir/'config.yaml'),
            'reference_sha256':sha(ref_path),'rollout_reference_sha256':sha(roll_path),
            'dataset_sha256':dataset_hashes,'pipeline_sha256':ref['pipeline_sha256'],
            'source_sha256':ref['source_sha256'],
            'additional_source_sha256':{str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),HERE/'frame_probe_core.py')},
            'inputs':[],'banks':[],'samples':[],'loss_banks':[],'rows':[], 'geometry_objects':[],
            'geometry_samples':[],'normalization_checks':[],'files_sha256':{},
            'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device)},
            'scope':'Existing full-data camera checkpoint, paired to F18/F19; no training or target changes. Geometry is scoring/audit only, never conditioning.'}
    args.output.mkdir(parents=True)
    write(args.output/'results.partial.json',report)
    geometry_reference(args,datasets,ref['inputs'],report,device)
    random.seed(29);np.random.seed(29);torch.manual_seed(29)
    pipeline=build_stage1_pipeline(pipeline_path,device)
    model=restore_run(pipeline,checkpoint,device)
    expected_weights={k:tensor_sha(v) for k,v in checkpoint['model'].items()}
    parameters=dict(model.named_parameters())
    assert all(tensor_sha(parameters[k])==v for k,v in expected_weights.items())
    del checkpoint
    model.requires_grad_(False);model.touch_encoder.eval()
    cfg=yaml.safe_load(pipeline_path.read_text())
    decoder=pipeline.init_ss_decoder(cfg['ss_decoder_config_path'],cfg['ss_decoder_ckpt_path']).eval().requires_grad_(False)
    paths=[Path(cfg[k]) for k in ('ss_decoder_config_path','ss_decoder_ckpt_path')]
    paths=[p if p.is_absolute() else pipeline_path.parent/p for p in paths]
    report['decoder_sha256']={str(p):sha(p) for p in paths}
    gen=pipeline.ss_generator
    assert gen.reverse_fn.p_unconditional==gen.self_consistency_prob==gen.fm_eps_max==0
    gen.no_shortcut=True;gen.inference_steps=25;gen.rescale_t=float(cfg.get('ss_rescale_t',3))
    gen.reverse_fn.interval=list(cfg.get('ss_cfg_interval',[0,500]));gen.reverse_fn.strength=0
    t_seq,d=gen._prepare_t_and_d();assert d==0
    report['sampler']={'steps':25,'cfg_strength':0,'no_shortcut':True,'rescale_t':gen.rescale_t,
        'reversed_timestamp':gen.reversed_timestamp,'time_sequence':t_seq.tolist(),
        'solver':gen._solver_method,'precision':'bf16','draws':2}
    assert report['sampler']==roll['sampler'] and report['decoder_sha256']==roll['decoder_sha256']
    with torch.no_grad():
        for previous in ref['inputs']:
            split,gi=previous['split'],previous['group']
            ds=copy.copy(datasets[split]);lookup={r['sample_id']:r for r in ds.records}
            records=[lookup[sid] for sid in previous['sample_ids']];ds.records=records
            batch=collate_touch_batch([ds[i] for i in range(4)])
            targets,ca,kw,points,mask=prepare_batch(pipeline,batch,device,'bf16',True,False,False)
            assert len(ca)==1 and not kw and mask.all() and points.shape[1]==8192
            transform=batch['object_from_camera'].to(device)
            oracle=batch['touch_xyz'].to(device)@transform[:,:3,:3].transpose(1,2)+transform[:,None,:3,3]
            assert tensor_sha(oracle)==previous['points_sha256']
            prepared=[];original_prepare=model.touch_encoder.prepare_points
            def capture(*a,**kw):
                result=original_prepare(*a,**kw);prepared.append(result);return result
            with patch.object(model.touch_encoder,'prepare_points',side_effect=capture),amp(device,'bf16'):
                tokens=model.get_touch_tokens(points,mask)
            assert len(prepared)==1 and torch.isfinite(tokens).all()
            qp,mp,cp,sp=prepared[0]
            qo,mo,co,so=original_prepare(oracle,mask)
            assert torch.equal(mp,mask) and torch.equal(mo,mask)
            actual={'split':split,'group':gi,'sample_ids':batch['sample_id'],'object_ids':[r['object_id'] for r in records],
                'image_sha256':tensor_sha(batch['image']),'pointmap_sha256':tensor_sha(batch['pointmap']),
                'visual_sha256':tensor_sha(ca[0]),'points_sha256':tensor_sha(points),'oracle_points_sha256':tensor_sha(oracle),
                'mask_sha256':tensor_sha(mask),'tokens_sha256':tensor_sha(tokens),
                'target_sha256':{k:tensor_sha(v) for k,v in targets.items()}}
            check_observation(actual,previous)
            for i,rec in enumerate(records):
                with np.load(args.output/f"geometry_{rec['sample_id']}.npz",allow_pickle=False) as geo:
                    delta=float(np.abs(array(oracle[i])-geo['oracle_points_float64']).max())
                assert delta<5e-6
                checks={'sample_id':rec['sample_id'],'gpu_vs_float64_oracle_max_error':delta,
                    'camera':normalization_check(array(points[i]),array(qp[i]),array(cp[i]),float(sp[i,0])),
                    'oracle':normalization_check(array(oracle[i]),array(qo[i]),array(co[i]),float(so[i,0]))}
                report['normalization_checks'].append(checks)
            save_npz(args.output,report,f'{split}_g{gi}_coordinate_inputs.npz',
                sample_ids=np.array(batch['sample_id']),raw_camera_points=array(batch['touch_xyz']),
                camera_pre_encoder_points=array(points),oracle_pre_encoder_points=array(oracle),
                camera_vecsetx_points=array(qp),oracle_vecsetx_points=array(qo),
                camera_center=array(cp),camera_inverse_radius=array(sp),
                oracle_center=array(co),oracle_inverse_radius=array(so),object_from_camera=array(transform),mask=mask.cpu().numpy())
            with amp(device,'bf16'):
                target_support=np.stack([decode_support(decoder,t[None]) for t in targets['shape']])
            actual['target_support_sha256']=tensor_sha(torch.from_numpy(target_support));report['inputs'].append(actual)
            old_input=next(b for b in roll['inputs'] if (b['split'],b['group'])==(split,gi))
            assert actual['target_support_sha256']==old_input['target_support_sha256']
            target_file=f'{split}_g{gi}_targets.npz'
            save_npz(args.output,report,target_file,target_occupancy=target_support,target_shape=array(targets['shape']),sample_ids=np.array(batch['sample_id']))
            gen.reverse_fn.training=True
            loss_bank=make_bank(gen,targets,2300000+(0 if split=='train' else 10000)+gi*1000+29)
            new_banks=[{'split':split,'group':gi,'time_kind':kind,'draw':draw,'times':times.cpu().tolist(),
                'time_sha256':tensor_sha(times),'noise_sha256':{k:tensor_sha(v) for k,v in noise.items()}}
                for kind,draw,times,noise in loss_bank]
            assert new_banks==[b for b in ref['banks'] if (b['split'],b['group'])==(split,gi)]
            report['loss_banks'].extend(new_banks)
            for visual_state in ('present','zero'):
                visual=ca[0] if visual_state=='present' else torch.zeros_like(ca[0])
                for shift in range(4):
                    conditioned=condition_tokens(tokens,shift)
                    for kind,draw,times,noise in loss_bank:
                        with amp(device,'bf16'): losses=paired_loss(gen,targets,visual,conditioned,times,noise)
                        report['rows'].append({'split':split,'group':gi,'visual':visual_state,'surface_shift':shift,
                            'time_kind':kind,'draw':draw,'losses':losses})
            gen.reverse_fn.training=False
            banks=[]
            for draw in range(2):
                seed=2400029+(0 if split=='train' else 10000)+gi*100+draw
                with torch.random.fork_rng(devices=[device.index]):
                    torch.manual_seed(seed);noise=gen._generate_noise({k:tuple(v.shape) for k,v in targets.items()},device)
                banks.append(noise)
                item={'split':split,'group':gi,'draw':draw,'seed':seed,'noise_sha256':{k:tensor_sha(v) for k,v in noise.items()}}
                assert item==next(b for b in roll['banks'] if (b['split'],b['group'],b['draw'])==(split,gi,draw))
                report['banks'].append(item)
            for visual_state in ('present','zero'):
                visual=ca[0] if visual_state=='present' else torch.zeros_like(ca[0])
                for shift in (0,1):
                    conditioned=condition_tokens(tokens,shift)
                    expected_context=torch.cat((visual,conditioned.to(visual)),1);seen=[]
                    def context_check(module,inputs):
                        torch.testing.assert_close(inputs[0],expected_context.to(inputs[0]),rtol=0,atol=0);seen.append(True)
                    hook=pipeline.backbone.blocks[0].cross_attn['shape'].to_kv.register_forward_pre_hook(context_check)
                    try:
                        for draw,noise in enumerate(banks):
                            with amp(device,'bf16'):
                                predicted=sample_from_noise(gen,noise,visual,conditioned)
                                supports=np.stack([decode_support(decoder,p[None]) for p in predicted])
                            filename=f'{split}_g{gi}_{visual_state}_s{shift}_d{draw}.npz'
                            save_npz(args.output,report,filename,predicted_occupancy=supports,predicted_shape=array(predicted),sample_ids=np.array(batch['sample_id']))
                            for i,rec in enumerate(records):
                                report['samples'].append({'split':split,'group':gi,'visual':visual_state,'surface_shift':shift,
                                    'draw':draw,'sample_id':rec['sample_id'],'object_id':rec['object_id'],
                                    'prediction_file':filename,'array_index':i,'target_file':target_file,
                                    **support_metrics(supports[i],target_support[i])})
                        assert seen
                    finally:hook.remove()
            write(args.output/'results.partial.json',report)
            print('Camera loss/rollout',split,f'group {gi+1}/8 complete',flush=True)
    assert all(tensor_sha(parameters[k])==v for k,v in expected_weights.items())
    report['adapted_parameters_unchanged']=True
    report['coordinate_contract_passed']=report['coordinate_reference_passed'] and len(report['normalization_checks'])==64
    report['complete']=True
    check_report_pair(report,roll)
    check_loss_pair(report,ref)
    write(args.output/'results.json',report)
    print('Complete:',args.output/'results.json',flush=True)


if __name__=='__main__':main()
