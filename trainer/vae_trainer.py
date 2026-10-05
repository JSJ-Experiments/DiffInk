"""Real VAE training path: model-owned scaling, masked English losses, accumulation."""
import os
from contextlib import nullcontext
import torch
from model.losses import loss_terms, weighted_loss
from utils.mask import downsample_mask
from utils.ddp import reduce_loss


def train_vae_one_epoch(vae,config,train_loader,optimizer,scheduler,epoch,num_epochs,device,ddp=False,
                        on_optimizer_step=None,max_optimizer_updates=None):
    vae.train();base=vae.module if hasattr(vae,'module') else vae
    accumulation=int(getattr(config,'gradient_accumulation_steps',1))
    if accumulation<1:raise ValueError('positive gradient accumulation required')
    if getattr(config,'padding_strategy',None)=='microbatch' and getattr(config,'train_batch_size',1)!=1:
        raise ValueError('microbatch padding mitigation requires physical batch 1')
    if getattr(config,'post_end_eoc_window',0):
        raise ValueError('fixed post-end window not implemented; do not silently use random batch padding')
    if getattr(config,'anchor_gradient_fraction',0)>0 and getattr(config,'expected_xy_weight',0)<=0:
        raise ValueError('calibrate expected-XY anchor before training; do not silently train with anchor weight zero')
    optimizer.zero_grad(set_to_none=True);rows=[];failed=False;last_terms=None
    for i,batch in enumerate(train_loader):
        group_start=(i//accumulation)*accumulation
        group_size=min(accumulation,len(train_loader)-group_start)
        if i==group_start:group_terms={};group_loss=0.0
        boundary=(i-group_start+1)==group_size
        data,mask,text,_,writer=batch
        data=data.to(device).transpose(1,2);mask=mask.to(device)
        sync=vae.no_sync() if ddp and not boundary else nullcontext()
        with sync:
            output,ctc,kl,style=vae(data,downsample_mask(mask,8),text.to(device),writer.to(device),
                                  getattr(config,'ctc_weight',0)!=0,getattr(config,'style_weight',0)!=0,point_mask=mask)
            terms=loss_terms(output,base.to_model_space(data),mask,config,ctc,kl,style)
            total=weighted_loss(terms,config)
            finite=torch.isfinite(total).float()
            if ddp:torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                failed=True;optimizer.zero_grad(set_to_none=True)
            elif not failed:(total/group_size).backward()
        last_terms={k:float(v.detach()) for k,v in terms.items()}
        for key,value in last_terms.items():group_terms[key]=group_terms.get(key,0.0)+value
        group_loss+=float(total.detach())
        if not boundary:continue
        if failed:
            optimizer.zero_grad(set_to_none=True);failed=False;continue
        norm=torch.nn.utils.clip_grad_norm_(vae.parameters(),getattr(config,'grad_clip',10),error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        row={'optimizer_step':len(rows)+1,'microbatches':group_size,'loss':group_loss/group_size,
             'loss_log_scope':'effective_batch_mean','last_microbatch_loss':float(total.detach()),
             **{k:v/group_size for k,v in group_terms.items()},'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr']}
        rows.append(row)
        if on_optimizer_step:on_optimizer_step(row)
        elif not ddp or torch.distributed.get_rank()==0:print(row,flush=True)
        if max_optimizer_updates is not None and len(rows)>=max_optimizer_updates:break
    return rows


def val_vae_one_batch(model,val_loader,epoch,device,save_path):
    from model.gmm import get_mixture_coef,sample_from_params
    from utils.visual import plot_line_cv2
    model.eval();model=model.module if hasattr(model,'module') else model
    os.makedirs(save_path,exist_ok=True)
    with torch.no_grad():
        for p,batch in enumerate(val_loader):
            data,mask,*_=batch;data=data.to(device);mask=mask.to(device)
            output=model.val(data.transpose(1,2),point_mask=mask,latent_mean=True)
            coefficients=get_mixture_coef(output,20)
            for i in range(len(data)):
                params=[v[i].cpu() for v in coefficients[:7]]
                sequence=sample_from_params(params,max_seq_len=int(mask[i].sum()),mode='expectation')
                recon=model.to_data_space(sequence)
                plot_line_cv2(data[i].transpose(0,1),save_path=f'{save_path}/gt_{p}_{i}.png',canvas_height=256,padding=20,line_thickness=2)
                plot_line_cv2(recon,save_path=f'{save_path}/recon_{p}_{i}.png',canvas_height=256,padding=20,line_thickness=2)
            break
