"""CPU-only saved-checkpoint autopsy: latent-mean/noise and component-selection controls."""
import argparse
import json
import math
from pathlib import Path
import torch
from .autopsy import selected_sample,coordinate_forward
from .inkvae import setup,fixed_evaluation
from .trajectory_diagnostics import diagnostics,save_views


def examine(directory,config_path,repo,data_root):
    directory=Path(directory);torch.set_num_threads(2)
    checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    model,train,val,cfg,root=setup(config_path,repo,data_root)
    try:
        if checkpoint['config']!=cfg:raise ValueError('checkpoint config mismatch')
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        sample=selected_sample(train,cfg['sample_id']);batch=train.collate_fn([sample])
        truth=sample[1].numpy();length=len(truth);target=sample[1][:,:2]*cfg['model_input_scale']
        from model.gmm import get_mixture_coef_max
        rows=[]
        def describe(output,mode,seed):
            pi,mx,my,sx,sy,rho,pen,logits=get_mixture_coef_max(output,20)
            sx=sx.clamp_min(1e-6);sy=sy.clamp_min(1e-6);rho=rho.clamp(-1+1e-5,1-1e-5)
            dx=(target[:,0][None,None,:]-mx[:,:,:length])/sx[:,:,:length]
            dy=(target[:,1][None,None,:]-my[:,:,:length])/sy[:,:,:length]
            determinant=(1-rho[:,:,:length].square()).clamp_min(1e-8)
            quadratic=(dx-rho[:,:,:length]*dy).square()/determinant+dy.square()
            scores=pi[:,:,:length].clamp_min(torch.finfo(pi.dtype).tiny).log()-math.log(2*math.pi)-sx[:,:,:length].log()-sy[:,:,:length].log()-.5*determinant.log()-.5*quadratic
            choice=pi[:,:,:length].argmax(1,keepdim=True);oracle=scores.argmax(1,keepdim=True)
            selections={'highest_weight':torch.stack([mx[:,:,:length].gather(1,choice).squeeze(1),my[:,:,:length].gather(1,choice).squeeze(1)],-1),
                        'mixture_mean':torch.stack([(pi*mx).sum(1)[:,:length],(pi*my).sum(1)[:,:length]],-1),
                        'target_posterior_oracle':torch.stack([mx[:,:,:length].gather(1,oracle).squeeze(1),my[:,:,:length].gather(1,oracle).squeeze(1)],-1)}
            states=torch.nn.functional.one_hot(logits[:,:,:length].argmax(1),3)
            for name,xy in selections.items():
                seq=torch.cat([xy,states],-1)[0].numpy()
                row={'mode':mode,'seed':seed,'selection':name,'uses_target_for_component_selection':name=='target_posterior_oracle',
                     'diagnostics':diagnostics(seq,truth,cfg['model_input_scale'])};rows.append(row)
                if mode=='latent_mean' or (seed==1042 and name=='highest_weight'):
                    save_views(directory,f'cpu-{mode}-{name}',seq,truth,cfg['model_input_scale'],{'writer_id':sample[0],'text':sample[2]})
        for seed in [1042,1043,1044,1045]:
            with fixed_evaluation(model,seed,'cpu'):
                _,output,_=coordinate_forward(model,batch,cfg,'cpu');describe(output,'sampled_latent',seed)
        with fixed_evaluation(model,1042,'cpu'):
            data=batch[0].clone();data[:,:,:2]*=cfg['model_input_scale']
            features=model.encoder(data.transpose(1,2));mu=model.conv_mu(features);logvar=model.conv_logvar(features)
            output=model.decode(mu);describe(output,'latent_mean',None)
            std=(.5*logvar).exp();latent_std_median=float(std.median())
        report={'gpu':False,'training':False,'new_optimizer_steps':0,'checkpoint_step':checkpoint['step'],
                'strict_load':True,'all_state_finite':all(torch.isfinite(v).all() for v in model.state_dict().values()),
                'latent_mean_is_inference_control_only':True,'oracle_not_usable_for_generation':True,
                'latent_std_median':latent_std_median,'samples':rows}
        (directory/'cpu_checkpoint_diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    finally:train.hf.close();val.hf.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    repo='third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    parser.add_argument('--repo',default=repo);parser.add_argument('--config',default=repo+'/configs/vae_iam_autopsy_geometry.yaml')
    parser.add_argument('--data-root',default='data/diffink/iam_overfit');args=parser.parse_args()
    print(json.dumps(examine(args.directory,args.config,args.repo,args.data_root),indent=2))
if __name__=='__main__':main()
