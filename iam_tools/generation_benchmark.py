"""Bounded disposable throughput study. Does NOT select/train a production model.

Same parent model/data; CPU thread count, input caching, digest synchronization,
and physical batch size examined separately. Quality not comparable for batch32.
"""
import copy
import json
import time
from pathlib import Path
import h5py
import torch
from .generation_study import collate
from .latent_diffusion import TextLatentDenoiser, cosine_schedule, forward_noise, masked_mse
from .resource_monitor import ResourceMonitor
from .pen_ab import file_sha

PARENT='checkpoints/iam_generation_gate/20261007-153404'
CASES=[('live_threads4_batch8',4,8,False,True),
       ('cached_threads4_batch8',4,8,True,True),
       ('cached_nohash_threads4_batch8',4,8,True,False),
       ('cached_nohash_threads1_batch8',1,8,True,False),
       ('cached_nohash_threads1_batch32',1,32,True,False)]


def run(root='/data', seconds=45):
    if not torch.cuda.is_available() or not 35<=seconds<=90:raise ValueError('T4 and bounded35–90s/case required')
    root=Path(root);parent=root/PARENT;settings=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text())
    latents={}
    with h5py.File(parent/'source.h5') as f:
        for sid in data['splits']['train']:latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda')
    stats=torch.load(parent/'whitening.pt',weights_only=False);stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()}
    checkpoint=torch.load(parent/'text/checkpoint-last.pt',map_location='cuda',weights_only=False)
    ids=data['splits']['train'];alpha=cosine_schedule(device='cuda');out=root/'checkpoints/iam_resource_benchmark'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__);(out/source.name).write_bytes(source.read_bytes());(out/'resource_monitor.py').write_bytes(source.with_name('resource_monitor.py').read_bytes())
    config=dict(parent=PARENT,parent_sha256=file_sha(parent/'text/checkpoint-last.pt'),cases=CASES,seconds_per_case=seconds,cpu_request=4,cpu_limit='Modal default (request is not limit)',memory_request_mib=16384,
                warmed=True,device='T4',dtype='FP32',optimizer='AdamW restored parent moments,LR1e-4; disposable models only',same_parent=True,benchmark_not_quality_experiment=True,
                synchronization='float(clip_grad_norm) and loss.item each step, plus original epsilon/t/drop digest copy in hash cases; no artificial per-step CUDA synchronize',
                latency='CPU wall per update through scalar sync; GPU boundary synchronize for complete case elapsed')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');results={}
    with ResourceMonitor(out,cpu_request=4,memory_request_mib=16384,interval=2,sustained_seconds=30) as monitor:
        for name,threads,batch,cached,hash_noise in CASES:
            monitor.set_phase('prepare/'+name);torch.set_num_threads(threads);torch.manual_seed(4142)
            model=TextLatentDenoiser(**settings['model']).cuda();model.load_state_dict(checkpoint['model_state_dict']);model.train()
            optimizer=torch.optim.AdamW(model.parameters());optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for group in optimizer.param_groups:group['lr']=1e-4
            groups=[ids[i:i+8] for i in range(0,32,8)] if batch==8 else [ids]
            prepared=[collate(latents,data['records'],settings['vocab'],group,stats,'cuda') for group in groups]
            rng=torch.Generator(device='cuda').manual_seed(6142)
            def update(step):
                clean,mask,text=prepared[step%len(groups)] if cached else collate(latents,data['records'],settings['vocab'],groups[step%len(groups)],stats,'cuda')
                noise=torch.randn(clean.shape,device='cuda',generator=rng);t=torch.randint(0,1000,(batch,),device='cuda',generator=rng);drop=torch.rand(batch,device='cuda',generator=rng)<.1
                if hash_noise:
                    import hashlib
                    hashlib.sha256(noise.cpu().numpy().tobytes()+t.cpu().numpy().tobytes()+drop.cpu().numpy().tobytes()).hexdigest()
                pred=model(forward_noise(clean,noise,t,alpha,mask),t.float()/999,text,mask,drop);loss=masked_mse(pred,clean,mask)
                optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
                value=float(loss)
                if not torch.isfinite(torch.tensor(value)):raise FloatingPointError('nonfinite disposable benchmark')
                return value,norm
            for i in range(40):update(i)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();monitor.set_phase('train/'+name);start=time.monotonic();step=0
            while time.monotonic()-start<seconds and step<6000:
                began=time.monotonic();loss,norm=update(step);monitor.step(time.monotonic()-began,batch);step+=1
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            result=dict(steps=step,examples=step*batch,seconds=elapsed,examples_per_second=step*batch/elapsed,updates_per_second=step/elapsed,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,last_loss=loss)
            results[name]=result;print(dict(case=name,**result),flush=True);monitor.set_phase('cleanup');del model,optimizer;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),results=results)
