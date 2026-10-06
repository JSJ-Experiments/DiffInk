"""Bounded T4 utilization measurement. No optimizer or checkpoint writes."""
"""Bounded existing-data geometry pilot; no OCR/KL/style or full-IAM training."""
from pathlib import Path
import modal

volume=modal.Volume.from_name('diffink-data')
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
       .workdir('/app')
       .add_local_dir(str(repo/'model'),'/app/model')
       .add_local_dir(str(repo/'dataset'),'/app/dataset')
       .add_local_dir(str(repo/'utils'),'/app/utils')
       .add_local_dir(str(repo/'trainer'),'/app/trainer')
       .add_local_dir('iam_tools','/app/iam_tools')
       .add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
SOURCE='checkpoints/iam_fullset_joint/20261006-140945/checkpoint-best.pt'
SHA='89ec459de6496275a3712c08629daea10d8f4f03311712c77f49e652bca915a3'

app = modal.App('diffink-english-posterior-batching-validation')

@app.function(image=image, volumes={'/data': volume}, gpu='T4', cpu=4,
              memory=16384, timeout=600, retries=0, max_containers=1)
def benchmark():
    import json,time,threading,subprocess
    import numpy as np
    import torch
    from iam_tools.writer_expansion import load,device_batches,evaluate
    from iam_tools.latent_integration import terms
    from iam_tools.fast_geometry import fast_terms,GeometryGraphs
    from iam_tools.metric_workers import metric_pool,geometry_job
    from iam_tools.pen_ab import file_sha
    torch.set_num_threads(4);torch.manual_seed(4042)
    model,samples,raw,cfg,vocab,prov=load('/app/configs/engineering_english.yaml','/app','/data',SOURCE,SHA,writer_id=None)
    model.cuda().eval();model.ocr_model.requires_grad_(False);model.style_classifier.requires_grad_(False)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    batches=device_batches(raw,'cuda');ids=prov['splits']['train'][:8]
    selected=[batches[i] for i in ids];pen=.02099049935353879
    directory=Path('/data/checkpoints/iam_throughput_validation')/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True)
    graphs=GeometryGraphs(model,pen)
    begin=time.perf_counter();graphs.prepare(selected);torch.cuda.synchronize()
    capture_seconds=time.perf_counter()-begin
    print(dict(capture_seconds=capture_seconds,graphs=len(graphs.cache),memory_gb=torch.cuda.memory_reserved()/2**30),flush=True)
    torch.manual_seed(991);noise=[torch.randn(1,model.config.latent_dim,b[0].shape[-1]//8,device='cuda') for b in selected]
    weights=torch.tensor([1.,.1,pen],device='cuda')
    def reference_loss(batch,eps):
        t=terms(model,batch,epsilon=eps,pen_on_mean=True)
        return torch.stack([t[k] for k in ('mean_geometry','sampled_geometry','pen')])
    def check_inputs(bs,es):
        model.zero_grad(set_to_none=False);eager=[]
        for b,e in zip(bs,es):
            t=reference_loss(b,e);(t*weights).sum().div(len(bs)).backward();eager.append(t.detach())
        grads=[p.grad.clone() for p in graphs.parameters]
        model.zero_grad(set_to_none=False);actual=[graphs.replay(b,divisor=len(bs),epsilon=e) for b,e in zip(bs,es)]
        numerator=torch.stack([(p.grad-g).square().sum() for p,g in zip(graphs.parameters,grads)]).sum().sqrt()
        denominator=torch.stack([g.square().sum() for g in grads]).sum().sqrt().clamp_min(1e-12)
        relative=float(numerator/denominator);difference=float((torch.stack(eager)-torch.stack(actual)).abs().max())
        if relative>5e-5 or difference>2e-5:raise AssertionError(dict(gradient_relative_l2=relative,max_loss_difference=difference))
        return dict(gradient_relative_l2=relative,max_loss_difference=difference,microbatches=len(bs))
    checks=[check_inputs(selected,noise)]
    shifted=[(b[0].clone(),b[1],b[2]) for b in selected[:2]]
    for raw,mask,_ in shifted:raw[:,:2]+=mask[:,None]*.01
    checks.append(check_inputs(shifted,noise[:2]))
    # Captures must read live parameter storage, not a frozen copy.
    with torch.no_grad():model.conv_mu.bias[0].add_(1e-4)
    checks.append(check_inputs(selected[:2],noise[:2]))
    with torch.no_grad():model.conv_mu.bias[0].copy_(original['conv_mu.bias'][0])
    stage=['warmup'];stopping=threading.Event();gpu=[]
    def sampler():
        while not stopping.is_set():
            try:
                u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip())
                gpu.append(dict(phase=stage[0],time=time.time(),utilization=u))
            except Exception:pass
            stopping.wait(.2)
    thread=threading.Thread(target=sampler,daemon=True);thread.start();results={}
    def timed(name,fn):
        torch.cuda.synchronize();stage[0]=name;w=time.perf_counter();c=time.process_time();value=fn();torch.cuda.synchronize()
        results[name]=dict(wall_seconds=time.perf_counter()-w,process_cpu_seconds=time.process_time()-c,result=value)
        stage[0]='between';print(name,results[name]['wall_seconds'],flush=True)
        return value
    def train(mode):
        torch.manual_seed(4042)
        for _ in range(32):
            model.zero_grad(set_to_none=False);total=torch.zeros(3,device='cuda')
            for b in selected:
                eps=torch.randn(1,model.config.latent_dim,b[0].shape[-1]//8,device='cuda')
                if mode=='graphs':t=graphs.replay(b,divisor=8,epsilon=eps)
                else:
                    t=reference_loss(b,eps) if mode=='reference' else fast_terms(model,b[0],b[1],eps)
                    (t*weights).sum().div(8).backward()
                    if mode=='reference':
                        for v in t:float(v.detach())
                total+=t.detach()
            total.cpu().tolist()
            float(torch.nn.utils.clip_grad_norm_(graphs.parameters,5,error_if_nonfinite=True))
        return dict(updates=32,microbatches=256,optimizer_steps=0)
    try:
        for mode in ['reference','fast_eager','graphs','graphs','reference']:
            timed('training_'+mode+('_repeat' if 'training_'+mode in results else ''),lambda mode=mode:train(mode))
        eval_ids=prov['splits']['train'][:24]
        sub_samples={i:samples[i] for i in eval_ids};sub_batches={i:batches[i] for i in eval_ids};splits={'train':eval_ids}
        with metric_pool(3) as pool:
            # Warm spawned workers separately; reuse them across all evaluations.
            xy=np.array([[0.,0.],[1.,0.],[1.,1.],[2.,1.]])
            warm=time.perf_counter();[f.result() for f in [pool.submit(geometry_job,xy,xy,np.array([0,0,1,2])) for _ in range(6)]]
            results['worker_startup_seconds']=time.perf_counter()-warm
            serial=timed('evaluation_serial',lambda:evaluate(model,sub_samples,sub_batches,splits,vocab,directory,1,draws=20))
            parallel=timed('evaluation_parallel',lambda:evaluate(model,sub_samples,sub_batches,splits,vocab,directory,2,draws=20,metric_pool=pool))
            if serial['lines']!=parallel['lines'] or serial['groups']!=parallel['groups']:raise AssertionError('parallel metrics or RNG differ')
            timed('evaluation_parallel_repeat',lambda:evaluate(model,sub_samples,sub_batches,splits,vocab,directory,3,draws=20,metric_pool=pool))
            batched=timed('evaluation_posterior_batches',lambda:evaluate(model,sub_samples,sub_batches,splits,vocab,directory,4,draws=20,metric_pool=pool,draw_batch_size=21))
            max_xy=0.;pen_mismatches=0
            for sid in eval_ids:
                for kind in ['mu']+[f'z-{k}' for k in range(20)]:
                    a=np.load(directory/f'step-1/{sid}/{kind}.npy');b=np.load(directory/f'step-4/{sid}/{kind}.npy')
                    max_xy=max(max_xy,float(np.abs(a[:,:2]-b[:,:2]).max()));pen_mismatches+=int((a[:,2:]!=b[:,2:]).any(1).sum())
            ocr_equal=all(a['decoded']==b['decoded'] for la,lb in zip(serial['lines'],batched['lines']) for a,b in zip([la['mu']]+la['sampled'],[lb['mu']]+lb['sampled']))
            if max_xy>5e-5 or pen_mismatches or not ocr_equal:raise AssertionError(dict(max_xy=max_xy,pen_mismatches=pen_mismatches,ocr_equal=ocr_equal))
            results['evaluation_posterior_batches']['result']=dict(lines=24,draws=20,max_xy_difference=max_xy,pen_mismatches=pen_mismatches,ocr_equal=ocr_equal)
            for key in ['evaluation_serial' ,'evaluation_parallel','evaluation_parallel_repeat']:results[key]['result']={'lines':24,'draws':20,'equal':True}
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in original.items())
        assert file_sha(Path('/data')/SOURCE)==SHA
        for name,row in results.items():
            if not isinstance(row,dict):continue
            values=[s['utilization'] for s in gpu if s['phase']==name]
            row['gpu_samples']=len(values);row['mean_gpu_utilization_percent']=float(np.mean(values)) if values else None
        output=dict(source_rel=SOURCE,source_sha256=SHA,checks=checks,results=results,capture_seconds=capture_seconds,
                    captured_lengths=list(graphs.cache),peak_memory_gb=torch.cuda.max_memory_reserved()/2**30,
                    gpu_samples=gpu,weights_unchanged=True,optimizer_updates=0,
                    caveat='GPU busy-time sampling is not SM occupancy; process CPU time excludes spawned workers.')
        (directory/'benchmark.json').write_text(json.dumps(output,indent=2)+'\n')
        return dict(output=str(directory),checks=checks,results=results,weights_unchanged=True)
    finally:
        stopping.set();thread.join(5);volume.commit()

@app.local_entrypoint()
def main(run:bool=False):
    if not run:print('No GPU allocated; --run required.');return
    print(benchmark.remote())
