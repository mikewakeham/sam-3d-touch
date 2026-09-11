"""Tiny fitting diagnostic. Fresh pretrained SAM3D, frozen VecSetX, native loss.

One arm per invocation. Camera/oracle arms share initialization, observations,
targets and noise seeds. This is memorization, not held-out evaluation.
"""
import argparse,hashlib,json,random,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from dataloader import build_dataloader,load_data_config
from train import build_stage1_pipeline,prepare_batch,build_optimizer,amp,TouchTrainingModel,trainable_state_dict,component_gradient_norms
from experiments.noisy_target.evaluate import tensor_digest
from rollout_gpu import occupancy,geometry_metrics

IDS=['479bbf9062de4b3196f08e52d1c12821_000','e846574df9334973955c80286af47ba7_007','fe200ce00a5d4c298eee89de0fc15f01_011','146b6c1f10bd445da977424c64dbb051_002']

def parameter_digest(parameters):
 h=hashlib.sha256()
 for name,p in parameters:
  h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--arm',choices=['image','camera','oracle'],required=True)
 p.add_argument('--pipeline-config',type=Path,default=Path('checkpoints/hf/pipeline.yaml'))
 p.add_argument('--data-config',type=Path,default=Path('configs/data_full_surface.yaml'))
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--steps',type=int,default=1000)
 p.add_argument('--objects',type=int,choices=[1,4],default=4)
 p.add_argument('--seed',type=int,default=29)
 p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
 a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 if a.steps<1:p.error('steps must be positive')
 ids=IDS[:a.objects];device=torch.device('cuda');cfg=OmegaConf.load(a.pipeline_config)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 pipeline=build_stage1_pipeline(a.pipeline_config,device)
 encoder=None
 if a.arm!='image':
  from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
  encoder=TouchEncoder(output_dim=pipeline.backbone.cond_channels,trainable=False,use_position=False).to(device).eval()
 model=TouchTrainingModel(pipeline.ss_generator,encoder,False,a.arm=='oracle')
 optimizer,parameters=build_optimizer(encoder,pipeline.backbone,SimpleNamespace(learning_rate=1e-4,cross_attention_learning_rate=1e-5,cross_attention_scope='full'))
 initial=parameter_digest(model.named_parameters())
 ca_initial=parameter_digest((n,p) for n,p in pipeline.backbone.named_parameters() if p.requires_grad)
 data=load_data_config(a.data_config);data['dataset']['split']='train'
 if data.get('touch',{}).get('source')!='full_surface':raise ValueError('Require full_surface data config')
 loader=build_dataloader(data,a.objects,0,shuffle=False,include_touch=encoder is not None,oracle_point_frame=a.arm=='oracle')
 lookup={r['sample_id']:r for r in loader.dataset.records};loader.dataset.records=[lookup[s] for s in ids]
 batch=next(iter(loader))
 targets,ca,kw,points,mask=prepare_batch(pipeline,batch,device,a.precision,encoder is not None,False,a.arm=='oracle')
 gen=pipeline.ss_generator
 assert gen.self_consistency_prob==0 and gen.fm_eps_max==0 and gen.reverse_fn.p_unconditional==0
 # Cache only frozen encoder outputs. The trainable projector remains in every loss graph.
 features=None
 if encoder is not None:
  with torch.no_grad(),amp(device,a.precision):
   pp,mm,_,_=encoder.prepare_points(points,mask)
   features=encoder.encoder.encode(pp,mm)['x'].detach()
   direct=encoder(points,mask)
   cached=encoder.output_projection(features)+encoder.touch_embedding
   torch.testing.assert_close(cached,direct,rtol=.002,atol=.00002)
 def tokens():return None if encoder is None else encoder.output_projection(features)+encoder.touch_embedding
 def kwargs(tok):return dict(kw) if tok is None else {**kw,'touch_tokens':tok}
 decoder=pipeline.init_ss_decoder(cfg.ss_decoder_config_path,cfg.ss_decoder_ckpt_path).eval().requires_grad_(False)
 # Targets are evaluated one at a time: occupancy helper is intentionally batch-one.
 with torch.no_grad(),amp(device,a.precision):target_support=[occupancy(decoder,targets['shape'][i:i+1]) for i in range(a.objects)]
 a.output_dir.mkdir(parents=True)
 report={'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'sample_ids':ids,
  'initial_all_parameters_sha256':initial,'initial_trainable_ca_sha256':ca_initial,
  'image_sha256':tensor_digest(batch['image']),'pointmap_sha256':tensor_digest(batch['pointmap']),
  'target_sha256':tensor_digest(targets['shape']),'point_features_sha256':tensor_digest(features) if features is not None else None,
  'pipeline_yaml':a.pipeline_config.read_text(),'generator_yaml':(a.pipeline_config.parent/cfg.ss_generator_config_path).read_text(),
  'runtime':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name()},
  'training':[],'assessments':[], 'purpose':'Fitting these exact training inputs; no held-out generalization claim.',
  'trainable_parameters':{g['name']:sum(p.numel() for p in g['params']) for g in optimizer.param_groups},
  'data_yaml':a.data_config.read_text()}
 repo=Path(__file__).resolve().parents[2]
 report['source_sha256']={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in ['train.py','dataloader.py','sam3d_objects/model/backbone/dit/embedder/touch.py']}
 def assess(step):
  py=random.getstate();nr=np.random.get_state()
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
   # Conditional wrapper training flag only; keep frozen backbone and encoder eval.
   gen.reverse_fn.training=True
   losses=[];swapped_losses=[];noise_digests=[]
   for draw in range(8):
    torch.manual_seed(100000+a.seed+draw);random.seed(100000+a.seed+draw)
    with amp(device,a.precision):loss,_=gen.loss(targets,*ca,**kwargs(tokens()))
    losses.append(float(loss))
    if encoder is not None and a.objects>1:
     torch.manual_seed(100000+a.seed+draw);random.seed(100000+a.seed+draw)
     with amp(device,a.precision):wrong_loss,_=gen.loss(targets,*ca,**kwargs(tokens().roll(1,0)))
     swapped_losses.append(float(wrong_loss))
   row={'step':step,'fresh_noise_native_losses':losses,'swapped_surface_native_losses':swapped_losses,'sampled':[]}
   gen.no_shortcut=True;gen.inference_steps=25;gen.rescale_t=float(cfg.get('ss_rescale_t',3));gen.reverse_fn.interval=list(cfg.get('ss_cfg_interval',[0,500]));gen.reverse_fn.training=False
   with amp(device,a.precision):tok=tokens()
   # Paired native noise draws, shared across every assessment and arm.
   for draw in [0,1]:
    torch.manual_seed(200000+a.seed+draw);noise=gen._generate_x0(targets)
    noise_digests.append({k:tensor_digest(v) for k,v in noise.items()})
    shapes={k:tuple(v.shape) for k,v in targets.items()}
    for strength in [0,7]:
     gen.reverse_fn.strength=strength
     with patch.object(gen,'_generate_noise',side_effect=lambda *args,**kw:{k:v.clone() for k,v in noise.items()}),amp(device,a.precision):pred=gen(shapes,device,*ca,**kwargs(tok))['shape']
     for i,sid in enumerate(ids):
      item={'sample_id':sid,'noise_draw':draw,'cfg':strength,'latent_mse':float((pred[i].float()-targets['shape'][i].float()).square().mean())}
      if not torch.isfinite(pred[i]).all():item['failure']='Nonfinite prediction'
      else:
       with amp(device,a.precision):support=occupancy(decoder,pred[i:i+1])
       item.update(geometry_metrics(support,target_support[i]))
       filename=f'step{step}_{sid}_draw{draw}_cfg{strength}.npz'
       np.savez_compressed(a.output_dir/filename,prediction=pred[i].float().cpu().numpy(),target=targets['shape'][i].cpu().numpy(),predicted_occupancy=support,target_occupancy=target_support[i])
       item['artifact']=filename
      row['sampled'].append(item)
   row['sample_noise_sha256']=noise_digests
   report['assessments'].append(row)
   gen.reverse_fn.training=True
  random.setstate(py);np.random.set_state(nr)
  (a.output_dir/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
  print(a.arm,'step',step,'fresh-noise loss',np.mean(losses),flush=True)
 assess(0)
 marks={min(100,a.steps),min(300,a.steps),a.steps}
 for step in range(1,a.steps+1):
  # Explicitly matched stochastic draws independent of initialization/assessment RNG.
  torch.manual_seed(a.seed+step);random.seed(a.seed+step)
  optimizer.zero_grad(set_to_none=True)
  with amp(device,a.precision):loss,_=gen.loss(targets,*ca,**kwargs(tokens()))
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
  loss.backward()
  if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise AssertionError('Frozen parameter has gradient')
  gradients=component_gradient_norms(model) if step==1 or step%20==0 or step in marks else {}
  norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
  optimizer.step()
  report['training'].append({'step':step,'loss':float(loss),'preclip_gradient_norm':float(norm),**gradients})
  if step%20==0:print(a.arm,'step',step,'loss',float(loss),flush=True)
  if step in marks:assess(step)
 report['final_trainable_ca_sha256']=parameter_digest((n,p) for n,p in pipeline.backbone.named_parameters() if p.requires_grad)
 report['final_all_parameters_sha256']=parameter_digest(model.named_parameters())
 assert report['initial_all_parameters_sha256']!=report['final_all_parameters_sha256']
 torch.save({'model':trainable_state_dict(model),'step':a.steps,'settings':report['settings'],'conditioning_config':model.conditioning_config},a.output_dir/'fitted_parameters.pt')
 (a.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
