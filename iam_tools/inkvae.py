"""CPU forward check / opt-in bounded single-GPU InkVAE mechanics runner."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
import yaml
from .check_batch import load_module


def setup(config_path, repo, data_root=None):
    repo=Path(repo).resolve();sys.path.insert(0,str(repo))
    from model.vae import VAE
    from utils.utils import ModelConfig
    cfg=yaml.safe_load(Path(config_path).read_text())
    root=Path(data_root or cfg['data_root'])
    ds=load_module(repo/'dataset/vae_dataset.py','inkvae_dataset')
    # Explicit cache reset when switching loader-test vs overfit vocabularies.
    ds.TrainDataset.text_cache=None;ds.TrainDataset.writer_cache=None
    train=ds.TrainDataset(str(root/cfg['train_file']),str(root/cfg['text_file']),str(root/cfg['writer_file']),transform=None)
    val=ds.TrainDataset(str(root/cfg['val_file']),str(root/cfg['text_file']),str(root/cfg['writer_file']),transform=None)
    cfg['num_text_embedding']=len(train.text_cache)+1;cfg['num_writer']=len(train.writer_cache)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('split_policy')!='same-writers-line-disjoint-overfit':
        train.hf.close();val.hf.close();raise ValueError('use the dedicated repeated-writer overfit dataset')
    if not 0<cfg['model_input_scale']<=1:raise ValueError('invalid model input scale')
    if cfg['aug_prob']!=0 or not cfg['gmm_logspace']:raise ValueError('overfit config must disable augmentation and use stable GMM')
    if not len(train) or not len(val):raise ValueError('empty dataset')
    torch.manual_seed(cfg['seed'])
    return VAE(ModelConfig(cfg)),train,val,cfg,root


def logspace_gmm_nll(pi, mu1, mu2, s1, s2, rho, x, y):
    """Evaluate the existing correlated bivariate mixture without PDF underflow."""
    s1=s1.clamp_min(1e-6);s2=s2.clamp_min(1e-6)
    rho=rho.clamp(-1+1e-5,1-1e-5)
    one_minus=(1-rho.square()).clamp_min(1e-8)
    dx=(x-mu1)/s1;dy=(y-mu2)/s2
    quadratic=(dx-rho*dy).square()/one_minus+dy.square()
    log_pdf=-math.log(2*math.pi)-s1.log()-s2.log()-0.5*one_minus.log()-0.5*quadratic
    return -torch.logsumexp(pi.clamp_min(torch.finfo(pi.dtype).tiny).log()+log_pdf,dim=1)


def batch_losses(model,batch,cfg,device):
    from model.gmm import get_mixture_coef_max,get_loss
    from utils.mask import downsample_mask
    data,mask,text,_,writers=batch
    data,mask,text,writers=[x.to(device) for x in (data,mask,text,writers)]
    data=data.clone()
    data[:,:,:2]*=cfg['model_input_scale']
    channel=data.transpose(1,2)
    output,ctc,kl,style=model(channel,downsample_mask(mask,8),text,writers)
    pi,mx,my,sx,sy,rho,pen,logits=get_mixture_coef_max(output,20)
    legacy,pen_element=get_loss(pi,mx,my,sx,sy,rho,pen,logits,
                              channel[:,:1],channel[:,1:2],channel[:,2:],focal_loss_reduction='none')
    stable=logspace_gmm_nll(pi,mx,my,sx,sy,rho,channel[:,:1],channel[:,1:2])
    terms={'gmm':(stable*mask).sum()/mask.sum(),
           'pen':(pen_element*mask).sum()/mask.sum(),'ctc':ctc,'kl':kl,'style':style}
    total=sum(value*cfg[name+'_weight'] for name,value in terms.items())
    floor_fraction=float(((legacy>=-math.log(1e-8)-1e-4)&mask).sum().detach()/mask.sum())
    if not all(torch.isfinite(v).all() for v in [output,total,*terms.values()]):
        raise FloatingPointError('nonfinite InkVAE forward/loss')
    return total,terms,output,mask,floor_fraction


@contextmanager
def fixed_evaluation(model, seed, device):
    """Hold latent noise fixed without changing training RNG or model mode."""
    device=torch.device(device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    was_training=model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            yield
    finally:
        model.train(was_training)


def greedy_ctc(ids, characters):
    result=[];previous=None
    for idx in ids:
        if idx!=previous and idx!=0:result.append(characters[idx-1])
        previous=idx
    return ''.join(result)


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def module_grad_norm(module):
    norms=[p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def smoke(config_path,repo,data_root=None,batch_size=1):
    torch.set_num_threads(2)
    model,train,val,cfg,root=setup(config_path,repo,data_root)
    try:
        model.eval()
        batch=train.collate_fn([train[i] for i in range(min(batch_size,len(train)))])
        with torch.no_grad():total,terms,output,mask,floor=batch_losses(model,batch,cfg,'cpu')
        if float(terms['ctc'])<=0:raise AssertionError('real VAE.forward produced zero CTC')
        report={'training':False,'optimizer_steps':0,'gpu':False,'device':'cpu',
                'parameter_count':sum(p.numel() for p in model.parameters()),
                'config_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                'ctc_path':'VAE.forward -> VAE.get_ocr_loss -> shared OCR helper',
                'input_shape':list(batch[0].shape),'output_shape':list(output.shape),
                'num_writer':cfg['num_writer'],'num_text_embedding':cfg['num_text_embedding'],
                'losses':{k:float(v) for k,v in terms.items()},'total_loss':float(total),
                'legacy_gmm_pdf_floor_fraction':floor,'runner_gmm':'logspace, masked reconstruction losses','model_input_scale':cfg['model_input_scale'],
                'normalization':'experimental','character_ends':'experimental'}
        (root/'vae_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:train.hf.close();val.hf.close()


def train_opt_in(config_path,repo,data_root=None,allow_experimental=False):
    if not allow_experimental:raise ValueError('experimental normalization/end-state acknowledgement required')
    if not torch.cuda.is_available():raise RuntimeError('single-GPU training requires CUDA')
    model,train,val,cfg,root=setup(config_path,repo,data_root)
    if not 1<=cfg['max_steps']<=200 or not 1<=cfg['max_wall_seconds']<=600:
        train.hf.close();val.hf.close();raise ValueError('smoke training budget exceeds hard cap')
    previous=json.loads((root/'vae_smoke.json').read_text())
    expected_config=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    expected_manifest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    if previous.get('config_sha256')!=expected_config or previous.get('manifest_sha256')!=expected_manifest:
        train.hf.close();val.hf.close();raise ValueError('rerun CPU smoke check for this exact config and dataset')
    from torch.utils.data import DataLoader
    from .preview import render
    from .preprocess import sequence_strokes
    from model.gmm import get_mixture_coef_max
    output_dir=Path(cfg['output_base'])/time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    output_dir.mkdir(parents=True,exist_ok=False)
    optimizer=None
    def checkpoint(step):
        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'step':step,'config':cfg,'experimental':True},output_dir/'checkpoint.pt')
    evaluated_steps=set()
    def reconstructions(step):
        if step in evaluated_steps:return
        evaluated_steps.add(step)
        # Both metrics and reconstructions share exactly the same forward/noise.
        for name,dataset in [('train',train),('val',val)]:
            captured={}
            hooks=[model.ocr_model.register_forward_hook(lambda m,a,o:captured.update(ocr=o)),
                   model.style_classifier.register_forward_hook(lambda m,a,o:captured.update(style=o))]
            sample=dataset[0]
            writer,points,text,_=sample
            batch=dataset.collate_fn([sample])
            try:
                with fixed_evaluation(model,cfg['seed']+1000,'cuda'):
                    total,terms,prediction,mask,floor=batch_losses(model,batch,cfg,'cuda')
                    pi,mx,my,*rest=get_mixture_coef_max(prediction,20)
                    choice=pi.argmax(dim=1,keepdim=True)
                    xy=torch.stack([mx.gather(1,choice).squeeze(1),my.gather(1,choice).squeeze(1)],dim=-1)
                    states=torch.nn.functional.one_hot(rest[-1].argmax(dim=1),3)
                    seq=torch.cat([xy,states],dim=-1)[0,:len(points)].cpu().numpy()
                    ocr_steps=(len(points)+7)//8
                    decoded=greedy_ctc(captured['ocr'][:ocr_steps,0].argmax(-1).tolist(),list(dataset.text_cache))
                    row={'step':step,'split':name,'sample_id':dataset.keys[dataset.sorted_indices[0]],
                         'writer_id':writer,'text':text,'total':float(total),
                         **{k:float(v) for k,v in terms.items()},
                         'greedy_ocr':decoded,'cer':edit_distance(text,decoded)/len(text),
                         'style_correct':bool(captured['style'].argmax(-1).item()==batch[-1].item()),
                         'xy_rmse_model_units':float(((xy[0,:len(points)]-points[:,:2].to('cuda')*cfg['model_input_scale']).square().mean()).sqrt()),
                         'pen_accuracy':float((states[0,:len(points)].argmax(-1)==points[:,2:].to('cuda').argmax(-1)).float().mean()),
                         'latent_noise_seed':cfg['seed']+1000}
            finally:
                for hook in hooks:hook.remove()
            with (output_dir/'fixed_metrics.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            print({'fixed':row},flush=True)
            import numpy as np
            np.save(output_dir/f'{name}-step-{step}-prediction-model-units.npy',seq)
            seq[:,:2]/=cfg['model_input_scale']
            # Force only the display endpoint; unmodified prediction saved above.
            seq[-1,2:]=[0,0,1]
            for suffix,trajectory in [('prediction',seq),('input',points.numpy())]:
                strokes=sequence_strokes(trajectory.copy())
                for stroke in strokes:stroke[:,1]*=-1
                render({'id':f'{name}-step-{step}-{suffix}','writer_id':writer,'text':text,
                        'strokes':[s.tolist() for s in strokes]},output_dir/f'{name}-step-{step}-{suffix}')
    try:
        model.to('cuda').train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['base_lr'],betas=cfg['betas'],weight_decay=cfg['weight_decay'])
        generator=torch.Generator().manual_seed(cfg['seed'])
        loader=DataLoader(train,batch_size=cfg['train_batch_size'],shuffle=True,generator=generator,num_workers=0,collate_fn=train.collate_fn)
        started=time.monotonic();step=0
        reconstructions(0)
        with (output_dir/'metrics.jsonl').open('w') as log:
            while step<cfg['max_steps'] and time.monotonic()-started<cfg['max_wall_seconds']:
                for batch in loader:
                    if step>=cfg['max_steps'] or time.monotonic()-started>=cfg['max_wall_seconds']:break
                    optimizer.zero_grad(set_to_none=True)
                    total,terms,_,_,floor=batch_losses(model,batch,cfg,'cuda')
                    total.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
                    optimizer.step();step+=1
                    row={'step':step,'total':float(total.detach()),'gradient_norm':float(norm),
                         **{k:float(v.detach()) for k,v in terms.items()},
                         'ocr_gradient_norm':module_grad_norm(model.ocr_model),
                         'style_gradient_norm':module_grad_norm(model.style_classifier)}
                    log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True)
                    if step%cfg['save_every']==0:checkpoint(step);reconstructions(step)
        checkpoint(step);reconstructions(step)
        result={'training':True,'steps':step,'output':str(output_dir),'experimental':True,
                'elapsed_seconds':time.monotonic()-started,'gpu_name':torch.cuda.get_device_name(),
                'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
                'config_sha256':expected_config,'manifest_sha256':expected_manifest,
                'stop_reason':'max_steps' if step==cfg['max_steps'] else 'wall_time',
                'fixed_evaluation_steps':sorted(evaluated_steps)}
        (output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:train.hf.close();val.hf.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    repo='third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    p.add_argument('--repo',default=repo);p.add_argument('--config',default=repo+'/configs/vae_iam_overfit.yaml')
    p.add_argument('--data-root');p.add_argument('--train',action='store_true');p.add_argument('--allow-experimental',action='store_true')
    args=p.parse_args()
    if args.train:
        result=train_opt_in(args.config,args.repo,args.data_root,args.allow_experimental)
    else:result=smoke(args.config,args.repo,args.data_root)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
