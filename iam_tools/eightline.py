"""Guarded 8-line/one-writer engineering mechanics run through the REAL trainer."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .inkvae import setup,fixed_evaluation
from .autopsy import selected_sample,sample_sha,freeze_auxiliaries
from .pen_ab import file_sha,boundary_metrics,render_snapshot,MANIFEST_SHA
from .reconstruction import point_report

REFIT_SHA='c7427d8e0f6d1bf5dcbb6b6c98a08126a91ec275bac21b5b6ffe422222b246bf'


def check_config(cfg):
    required={'profile':'engineering-reconstruction','model_input_scale':.01,'trans_dropout':0,
              'use_decoder_padding_mask':True,'train_batch_size':1,'gradient_accumulation_steps':8,
              'rotation_degrees':0,'aug_prob':0,'gmm_weight':1,'pen_policy':'bounded_three_state',
              'kl_weight':0,'ctc_weight':0,'style_weight':0,'max_optimizer_updates':200,
              'sampled_z_evaluations':20,'max_wall_seconds':600,'eval_every':50,'source_sha256':REFIT_SHA}
    for key,value in required.items():
        if cfg.get(key)!=value:raise ValueError('unsupported eight-line setting: '+key)
    if len(cfg['sample_ids'])!=8 or len(set(cfg['sample_ids']))!=8:raise ValueError('eight distinct IDs required')


def load(config,repo,data_root=None,checkpoint=None):
    model,train,val,cfg,root=setup(config,repo,data_root)
    try:
        check_config(cfg)
        source=Path(checkpoint or cfg['source_checkpoint'])
        if file_sha(source)!=REFIT_SHA:raise ValueError('pinned refit checkpoint required')
        parent=torch.load(source,map_location='cpu',weights_only=True)
        model.load_state_dict(parent['model_state_dict'],strict=True);freeze_auxiliaries(model)
        samples=[selected_sample(train,sid) for sid in cfg['sample_ids']]
        if {s[0] for s in samples}!={'10174'}:raise ValueError('one writer only')
        if file_sha(root/'manifest.json')!=MANIFEST_SHA:raise ValueError('pinned manifest required')
        batches=[train.collate_fn([s]) for s in samples]
        hashes={'config_sha256':file_sha(config),'source_sha256':REFIT_SHA,'manifest_sha256':MANIFEST_SHA,
                'sample_sha256':[sample_sha(s) for s in samples]}
        return model,samples,batches,cfg,root,source,hashes
    finally:train.hf.close();val.hf.close()


def one_output(model,batch,z,point_mask):
    return model.decode(z,padding_mask=~point_mask.bool())


def evaluate(model,samples,batches,cfg,device,step,directory=None,sampled_count=None):
    from model.losses import mixture_expectation
    counts=sampled_count if sampled_count is not None else cfg['sampled_z_evaluations']
    rows=[]
    for j,(sample,batch) in enumerate(zip(samples,batches)):
        data,mask,*_=batch;data=data.to(device).transpose(1,2);mask=mask.to(device)
        length=len(sample[1]);truth=sample[1].numpy();labels=truth[:,2:].argmax(1)
        with fixed_evaluation(model,1042+j*100,device):
            features=model.encoder(model.to_model_space(data))
            mu=model.conv_mu(features);logvar=model.conv_logvar(features);std=(.5*logvar).exp()
            evaluated=[];arrays=[]
            for k in range(-1,counts):
                z=mu if k<0 else mu+torch.randn_like(std)*std
                output=one_output(model,batch,z,mask)
                xy=mixture_expectation(output)[0,:length].cpu().numpy()
                states=output[:,:3].argmax(1)[0,:length].cpu().numpy()
                geo=point_report(xy,states,truth,.01);pen=boundary_metrics(states,labels)
                evaluated.append({'kind':'mu' if k<0 else 'sampled_z','sample_index':k,'geometry':geo,'pen':pen})
                arrays.append((xy,states))
            sampled=evaluated[1:]
            report={'step':step,'sample_id':cfg['sample_ids'][j],'writer_id':sample[0],'text':sample[2],
                    'real_points':length,'physical_length':data.shape[-1],'minimal_padding_points':data.shape[-1]-length,
                    'latent_std_median':float(std.median()),'latent_std_mean':float(std.mean()),
                    'mu':evaluated[0],'sampled_z':sampled,'sampled_count':counts,
                    'seed':1042+j*100,'pen_f1_sampled':None,'xy_sampled_summary':{}}
            if counts:
                vals=[v['pen']['pen_up_f1'] for v in sampled]
                report['pen_f1_sampled']={'min':min(vals),'median':float(np.median(vals)),'max':max(vals)}
                for axis in ('x','y'):
                    vals=[v['geometry']['axes'][axis]['rmse_model_units'] for v in sampled]
                    report['xy_sampled_summary'][axis]={'min':min(vals),'median':float(np.median(vals)),'max':max(vals)}
            if directory is not None:
                line_dir=directory/f'step-{step}'/cfg['sample_ids'][j];line_dir.mkdir(parents=True,exist_ok=True)
                for k,(xy,states) in enumerate(arrays):
                    np.save(line_dir/('mu.npy' if k==0 else f'z-{k-1}.npy'),np.column_stack([xy,np.eye(3)[states]]))
                selections=[(0,'mu')]
                if counts:
                    scores=[sum(r['geometry']['axes'][a]['rmse_model_units']**2 for a in ('x','y')) for r in sampled]
                    ordered=np.argsort(scores)
                    selections.extend([(int(ordered[len(ordered)//2])+1,'sampled-median'),(int(ordered[-1])+1,'sampled-worst')])
                for k,label in selections:
                    xy,states=arrays[k];render_snapshot(line_dir,label,xy,states,truth,.01,sample[2],evaluated[k]['pen'])
            rows.append(report)
    result={'step':step,'training_samples_only':True,'lines':rows,
            'mu_mean_x_rmse':float(np.mean([r['mu']['geometry']['axes']['x']['rmse_model_units'] for r in rows])),
            'mu_mean_y_rmse':float(np.mean([r['mu']['geometry']['axes']['y']['rmse_model_units'] for r in rows])),
            'mu_macro_pen_f1':float(np.mean([r['mu']['pen']['pen_up_f1'] for r in rows]))}
    if directory is not None:
        with (directory/'fixed_metrics.jsonl').open('a') as log:log.write(json.dumps(result,allow_nan=False)+'\n')
        print({k:v for k,v in result.items() if k!='lines'},flush=True)
    return result


def preflight(config,repo,data_root=None,checkpoint=None):
    torch.set_num_threads(2)
    model,samples,batches,cfg,root,source,hashes=load(config,repo,data_root,checkpoint)
    from model.losses import calibrate_anchor
    model.train();torch.manual_seed(42)
    calibration=calibrate_anchor(model,batches,model.config,'cpu',cfg['anchor_gradient_fraction'])
    initial=evaluate(model,samples,batches,cfg,'cpu',0,sampled_count=0)
    result={'training':False,'optimizer_steps':0,'gpu':False,'physical_batch':1,'accumulation':8,
            'one_writer':'10174','sample_ids':cfg['sample_ids'],'calibration_cpu_only':calibration,'initial_mu':initial,**hashes}
    (root/cfg['preflight_file']).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def train(config,repo,allow_experimental=False):
    if not allow_experimental:raise ValueError('experimental acknowledgement required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    model,samples,batches,cfg,root,source,hashes=load(config,repo)
    from model.losses import calibrate_anchor
    from trainer.vae_trainer import train_vae_one_epoch
    previous=json.loads((root/cfg['preflight_file']).read_text())
    if any(previous.get(k)!=v for k,v in hashes.items()):raise ValueError('matching CPU preflight required')
    model.to('cuda').train();before={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith(('ocr_model.','style_classifier.'))}
    directory=Path(cfg['output_base'])/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42);calibration=calibrate_anchor(model,batches,model.config,'cuda',cfg['anchor_gradient_fraction'])
    cfg['expected_xy_weight']=calibration['weight'];model.config.expected_xy_weight=calibration['weight']
    (directory/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['base_lr'],betas=cfg['betas'],weight_decay=cfg['weight_decay'])
    torch.manual_seed(42);started=time.monotonic();fixed=[];stop_reason='max_updates'
    fixed.append(evaluate(model,samples,batches,cfg,'cuda',0,directory))
    # Each optimizer update sees each of the eight lines ONCE, each physically
    # collated separately to its OWN minimal multiple-of-8 length.
    class Loader:
        def __len__(self):return 8*cfg['max_optimizer_updates']
        def __iter__(self):
            generator=np.random.default_rng(42)
            for _ in range(cfg['max_optimizer_updates']):
                for j in generator.permutation(8):yield batches[int(j)]
    log=(directory/'metrics.jsonl').open('w')
    class TimeLimit(Exception):pass
    def save(step):
        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'config':cfg,'optimizer_updates':step,'source_sha256':REFIT_SHA,'sample_ids':cfg['sample_ids'],
                    'rng_state_cpu':torch.get_rng_state(),'rng_state_cuda':torch.cuda.get_rng_state_all()},directory/'checkpoint.pt')
    def callback(row):
        log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
        step=row['optimizer_step']
        if step%cfg['eval_every']==0:
            save(step);fixed.append(evaluate(model,samples,batches,cfg,'cuda',step,directory))
        if time.monotonic()-started>cfg['max_wall_seconds']:raise TimeLimit()
    try:
        history=train_vae_one_epoch(model,model.config,Loader(),optimizer,None,0,1,'cuda',on_optimizer_step=callback,max_optimizer_updates=200)
        updates=len(history)
    except TimeLimit:
        stop_reason='wall_time';updates=sum(1 for _ in (directory/'metrics.jsonl').open())
    finally:log.close()
    save(updates)
    if fixed[-1]['step']!=updates:fixed.append(evaluate(model,samples,batches,cfg,'cuda',updates,directory))
    if not all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items()):raise AssertionError('auxiliary state changed')
    if file_sha(source)!=REFIT_SHA:raise AssertionError('source changed')
    result={'gpu':torch.cuda.get_device_name(),'training':True,'optimizer_updates':updates,'physical_microbatches':updates*8,
            'physical_batch':1,'gradient_accumulation':8,'ctc_style_kl_off':True,'decoder_dropout':0,
            'rotation_off':True,'model_owned_scale':.01,'anchor_calibration':calibration,
            'auxiliary_state_unchanged':True,'source_unchanged':True,'output':str(directory),
            'elapsed_seconds':time.monotonic()-started,'stop_reason':stop_reason,'initial':fixed[0],
            'final':fixed[-1],'training_samples_only':True,'reconstruction_pen_gate':'requires per-line visual/sample review; no CTC promotion',**hashes}
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    page=['<!doctype html><meta charset="utf-8"><h1>Eight training lines / one writer</h1><p>Dropout0, minimal-padding physical batch1, accumulation8. GMM + calibrated expected-XY + bounded pen. KL/CTC/style off. No held-out/generalization claim.</p>']
    for row in fixed:
        page.append(f'<h2>Optimizer update {row["step"]}</h2>')
        for line in row['lines']:
            sid=line['sample_id'];page.append(f'<h3>{sid}</h3>')
            for label in ('mu','sampled-median','sampled-worst'):page.append(f'<img style="max-width:100%" src="step-{row["step"]}/{sid}/{label}-comparison.png">')
    (directory/'index.html').write_text('\n'.join(page));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='third_party/DiffInk');p.add_argument('--config',default='third_party/DiffInk/configs/engineering_english.yaml')
    p.add_argument('--data-root',default='data/diffink/iam_overfit');p.add_argument('--checkpoint',default='data/checkpoints/iam_autopsy/pen_refit/20261005-100653/bounded_three_state/checkpoint.pt')
    a=p.parse_args();result=preflight(a.config,a.repo,a.data_root,a.checkpoint)
    print(json.dumps({k:v for k,v in result.items() if k!='initial_mu'},indent=2))
if __name__=='__main__':main()
