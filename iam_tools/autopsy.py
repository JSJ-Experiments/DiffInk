"""Guarded, exactly-one-line geometry-only InkVAE autopsy; other stages are not automatic."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .inkvae import setup,logspace_gmm_nll,fixed_evaluation
from .trajectory_diagnostics import diagnostics,save_views
from .autopsy_resume import load_resume,restore_optimizer,clipping_summary


def check_geometry_config(cfg):
    if cfg.get('stage')!='geometry-only' or cfg.get('pen_policy')!='off':raise ValueError('only geometry stage is authorized')
    if any(cfg[name+'_weight']!=0 for name in ['pen','ctc','style','kl']):raise ValueError('all non-coordinate objectives must be off')
    if cfg['gmm_weight']!=1 or cfg['train_batch_size']!=1:raise ValueError('use one line and unweighted coordinate NLL')
    if not 1<=cfg['max_steps']<=1000 or not 1<=cfg['max_wall_seconds']<=600:raise ValueError('autopsy budget exceeds 1000 steps / 600 seconds')
    if cfg['seed']!=42:raise ValueError('keep seed 42')
    if cfg['initialization'] not in ['fresh-seed-42','resume-model-and-optimizer']:raise ValueError('unsupported initialization')
    if cfg['initialization']=='resume-model-and-optimizer':
        if cfg.get('resume_step')!=1000 or cfg['base_lr']!=1e-5:raise ValueError('resume step 1000 at LR 1e-5 only')
        if not cfg.get('resume_checkpoint_sha256') or not cfg.get('resume_checkpoint'):raise ValueError('pinned checkpoint required')


def selected_sample(dataset,sample_id):
    if sample_id not in dataset.keys:raise ValueError('selected sample not in training split')
    position=dataset.keys.index(sample_id)
    index=next(i for i,x in enumerate(dataset.sorted_indices) if int(x)==position)
    return dataset[index]


def sample_sha(sample):
    writer,points,text,_=sample
    h=hashlib.sha256(json.dumps([writer,text],ensure_ascii=False).encode())
    h.update(points.contiguous().numpy().tobytes());return h.hexdigest()


def freeze_auxiliaries(model):
    for module in [model.ocr_model,model.style_classifier]:
        module.requires_grad_(False)


def coordinate_forward(model,batch,cfg,device):
    from utils.mask import downsample_mask
    from model.gmm import get_mixture_coef_max
    data,mask,text,_,writers=batch
    data=data.to(device).clone();data[:,:,:2]*=cfg['model_input_scale']
    mask=mask.to(device)
    # Flags actually suppress auxiliary forwards; zero loss weights alone are insufficient.
    output,ctc,kl,style=model(data.transpose(1,2),downsample_mask(mask,8),text.to(device),writers.to(device),
                           get_ctc_loss=False,get_style_loss=False,input_is_model_space=True)
    pi,mx,my,sx,sy,rho,pen,logits=get_mixture_coef_max(output,20)
    element=logspace_gmm_nll(pi,mx,my,sx,sy,rho,data[:,:,:1].transpose(1,2),data[:,:,1:2].transpose(1,2))
    loss=element[mask].mean() # only this term participates in backward
    if not all(torch.isfinite(v).all() for v in [output,loss]):raise FloatingPointError('nonfinite geometry loss/forward')
    return loss,output,kl


def evaluate(model,batch,sample,cfg,device,step,output_dir=None,gradient_info=None):
    from model.gmm import get_mixture_coef_max
    with fixed_evaluation(model,cfg['seed']+1000,device):
        loss,output,kl=coordinate_forward(model,batch,cfg,device)
        pi,mx,my,sx,sy,*rest=get_mixture_coef_max(output,20)
        choice=pi.argmax(1,keepdim=True)
        xy=torch.stack([mx.gather(1,choice).squeeze(1),my.gather(1,choice).squeeze(1)],-1)
        states=torch.nn.functional.one_hot(rest[-1].argmax(1),3)
        seq=torch.cat([xy,states],-1)[0,:len(sample[1])].cpu().numpy()
        sigma_x=sx.gather(1,choice)[0,0,:len(sample[1])]
        sigma_y=sy.gather(1,choice)[0,0,:len(sample[1])]
        row={'step':step,'sample_id':cfg['sample_id'],'gmm_nll':float(loss),
             'kl_report_only':float(kl),'latent_noise_seed':cfg['seed']+1000,
             'selected_sigma_x_median':float(sigma_x.median()),'selected_sigma_y_median':float(sigma_y.median()),
             'diagnostics':diagnostics(seq,sample[1].numpy(),cfg['model_input_scale'])}
        if gradient_info is not None:row.update(gradient_info)
    if output_dir is not None:
        sample_meta={'writer_id':sample[0],'text':sample[2]}
        save_views(output_dir,f'train-step-{step}',seq,sample[1].numpy(),cfg['model_input_scale'],sample_meta)
        # Neither nearest-target component nor ground-truth pen metrics are generation claims.
        with (output_dir/'fixed_metrics.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
        print({'fixed_geometry':row},flush=True)
    return row


def identity(config_path,root,sample):
    return {'config_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
            'sample_sha256':sample_sha(sample)}


def preflight(config_path,repo,data_root=None,resume_checkpoint=None):
    torch.set_num_threads(2)
    model,train,val,cfg,root=setup(config_path,repo,data_root)
    try:
        check_geometry_config(cfg);freeze_auxiliaries(model)
        sample=selected_sample(train,cfg['sample_id']);batch=train.collate_fn([sample])
        parent=load_resume(cfg,sample_sha(sample),resume_checkpoint)
        restored=None
        if parent is not None:
            model.load_state_dict(parent['model_state_dict'],strict=True)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['base_lr'],betas=cfg['betas'],weight_decay=cfg['weight_decay'])
            restored=restore_optimizer(optimizer,parent,cfg['base_lr'])
        row=evaluate(model,batch,sample,cfg,'cpu',parent['step'] if parent else 0)
        report={'training':False,'optimizer_steps':0,'gpu':False,'stage':cfg['stage'],
                'sample_id':cfg['sample_id'],'writer_id':sample[0],'unique_training_lines':1,
                'objectives':{'coordinates':True,'pen':False,'ctc':False,'style':False,'kl':False},
                **identity(config_path,root,sample),'initial':row,'restored_optimizer':restored,
                'resume_checkpoint_sha256':cfg.get('resume_checkpoint_sha256'),
                'source_rng_available':bool(parent and 'rng_state_cpu' in parent and 'rng_state_cuda' in parent)}
        (root/cfg.get('preflight_file','autopsy_preflight.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    finally:train.hf.close();val.hf.close()


def train_geometry(config_path,repo,data_root=None,allow_experimental=False,resume_checkpoint=None):
    if not allow_experimental:raise ValueError('explicit experimental acknowledgement required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    model,train,val,cfg,root=setup(config_path,repo,data_root)
    try:
        check_geometry_config(cfg);freeze_auxiliaries(model)
        sample=selected_sample(train,cfg['sample_id']);batch=train.collate_fn([sample])
        hashes=identity(config_path,root,sample)
        previous=json.loads((root/cfg.get('preflight_file','autopsy_preflight.json')).read_text())
        if any(previous.get(k)!=v for k,v in hashes.items()):raise ValueError('CPU preflight must match exact config/dataset/sample')
        parent=load_resume(cfg,hashes['sample_sha256'],resume_checkpoint)
        if parent is not None:
            if previous.get('resume_checkpoint_sha256')!=cfg['resume_checkpoint_sha256']:raise ValueError('CPU resume gate mismatch')
            model.load_state_dict(parent['model_state_dict'],strict=True)
        aux_before={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith(('ocr_model.','style_classifier.'))}
        output_dir=Path(cfg['output_base'])/time.strftime('%Y%m%d-%H%M%S',time.gmtime());output_dir.mkdir(parents=True,exist_ok=False)
        model.to('cuda').train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['base_lr'],betas=cfg['betas'],weight_decay=cfg['weight_decay'])
        restored=None
        rng_policy='fresh-seed-42'
        if parent is not None:
            restored=restore_optimizer(optimizer,parent,cfg['base_lr'])
            if 'rng_state_cpu' in parent and 'rng_state_cuda' in parent:
                torch.set_rng_state(parent['rng_state_cpu']);torch.cuda.set_rng_state_all(parent['rng_state_cuda'])
                rng_policy='restored-source-RNG'
            else:
                # The original checkpoint omitted RNG states; do not claim bitwise continuation.
                torch.manual_seed(cfg['seed']);rng_policy='seed-42-restart; source RNG states absent'
        # Collate exactly once: no shuffled loader or other lines can enter this job.
        batch=tuple(x.to('cuda') if isinstance(x,torch.Tensor) else x for x in batch)
        evaluated=[];history=[];started=time.monotonic();start_step=parent['step'] if parent else 0;step=start_step;updates=0
        def checkpoint():
            torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                        'step':step,'additional_steps':updates,'config':cfg,'sample_sha256':hashes['sample_sha256'],'experimental':True,
                        'rng_state_cpu':torch.get_rng_state(),'rng_state_cuda':torch.cuda.get_rng_state_all()},output_dir/'checkpoint.pt')
        def record():
            if step not in evaluated:
                info={'additional_steps':updates,'raw_grad_norm':history[-1]['gradient_norm'] if history else None,
                      'was_gradient_clipped':history[-1]['was_gradient_clipped'] if history else None,
                      'clipping_last_100':clipping_summary(history[-100:],cfg['grad_clip']),
                      'clipping_this_run':clipping_summary(history,cfg['grad_clip'])}
                evaluate(model,batch,sample,cfg,'cuda',step,output_dir,info);evaluated.append(step)
        record()
        with (output_dir/'metrics.jsonl').open('w') as log:
            while updates<cfg['max_steps'] and time.monotonic()-started<cfg['max_wall_seconds']:
                optimizer.zero_grad(set_to_none=True)
                loss,_,kl=coordinate_forward(model,batch,cfg,'cuda');loss.backward()
                if any(p.grad is not None for module in [model.ocr_model,model.style_classifier] for p in module.parameters()):
                    raise AssertionError('auxiliary gradient present in coordinate-only experiment')
                pen_grad=model.transformer_decoder.fc.weight.grad[:3]
                if pen_grad.count_nonzero():raise AssertionError('pen output has loss gradient in geometry stage')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
                optimizer.step();step+=1;updates+=1
                row={'step':step,'additional_steps':updates,'gmm_nll':float(loss.detach()),'gradient_norm':float(norm),
                     'was_gradient_clipped':float(norm)>cfg['grad_clip'],'learning_rate':optimizer.param_groups[0]['lr'],
                     'kl_report_only':float(kl.detach()),'auxiliary_gradients_absent':True,'pen_output_loss_gradient_zero':True}
                history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if step%50==0:print(row,flush=True)
                if step%cfg['save_every']==0:checkpoint();record()
        checkpoint();record()
        unchanged=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in aux_before.items())
        if not unchanged:raise AssertionError('frozen auxiliary state changed')
        result={'training':True,'stage':cfg['stage'],'steps':step,'start_step':start_step,'additional_steps':updates,'sample_id':cfg['sample_id'],
                'unique_training_lines':1,'repeated_same_sample_every_update':True,'auxiliary_state_unchanged':unchanged,
                'objectives':{'coordinates':True,'pen':False,'ctc':False,'style':False,'kl':False},
                'output':str(output_dir),'elapsed_seconds':time.monotonic()-started,
                'gpu_name':torch.cuda.get_device_name(),'torch_version':torch.__version__,
                'model_input_scale':cfg['model_input_scale'],'initialization':cfg['initialization'],
                'stop_reason':'max_steps' if updates==cfg['max_steps'] else 'wall_time',
                'fixed_evaluation_steps':evaluated,'experimental':True,**hashes,
                'resume_checkpoint':cfg.get('resume_checkpoint'),'resume_checkpoint_sha256':cfg.get('resume_checkpoint_sha256'),
                'restored_optimizer':restored,'rng_policy':rng_policy,
                'learning_rate':cfg['base_lr'],'clipping_this_run':clipping_summary(history,cfg['grad_clip'])}
        (output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        page=['<!doctype html><meta charset="utf-8"><h1>One-line geometry autopsy</h1>',
              '<p>Pen/CTC/style/KL loss OFF. True-pen renders diagnose XY only, not end-to-end reconstruction.</p>']
        for s in evaluated:page.append(f'<h2>Step {s}</h2><img style="max-width:100%" src="train-step-{s}-comparison.png">')
        (output_dir/'index.html').write_text('\n'.join(page))
        return result
    finally:train.hf.close();val.hf.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    repo='third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    parser.add_argument('--repo',default=repo);parser.add_argument('--config',default=repo+'/configs/vae_iam_autopsy_geometry.yaml')
    parser.add_argument('--resume-checkpoint');parser.add_argument('--data-root');parser.add_argument('--train',action='store_true');parser.add_argument('--allow-experimental',action='store_true')
    args=parser.parse_args()
    result=train_geometry(args.config,args.repo,args.data_root,args.allow_experimental,args.resume_checkpoint) if args.train else preflight(args.config,args.repo,args.data_root,args.resume_checkpoint)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
