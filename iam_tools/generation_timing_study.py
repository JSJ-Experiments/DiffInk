"""CPU one-block PE-vs-context autopsy on ALL256 TRAIN lines, no training."""
import json,time,tarfile
from pathlib import Path
import h5py,numpy as np,torch
from .generation_timing import TimingProbe,intervention_lengths
from .generation_alignment_study import ARMS
from .generation_alignment_review import verify_sources
from .generation_capacity import DATA
from .generation_duration_eval import inputs
from .generation_coverage_study import writer_tensor
from .generation_study import decode_sample
from .latent_diffusion import transform
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha

PARENT='checkpoints/iam_generation_alignment_continuation/20261008-084328'
PARENT_SHA={'global':'c09bf46fc27aa9e2f8fa7ddf3a2c957e7fcc9ff2a3c2550f0293824845b4f93a','soft_gaussian':'6f3890d5a7d4b985025d75eedeb288191a06a44d04132d44a5f76754efe957ed'}


@torch.no_grad()
def run(repo,root='/data'):
    torch.set_num_threads(4);root=Path(root);p=root/PARENT;cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());results=json.loads((p/'result.json').read_text());guards,_=verify_sources(p,repo,root,cfg,data,results)
    if any(file_sha(p/a/'checkpoint-last.pt')!=PARENT_SHA[a] for a in ARMS):raise ValueError('pinned completed48000-update sources required')
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);reader.eval().requires_grad_(False);stats=torch.load(root/DATA/'whitening.pt',weights_only=True)
    cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict());out=root/'checkpoints/iam_generation_timing'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as tar:
        for directory in ['iam_tools','model','dataset','utils','trainer','configs']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for q in sorted(base.rglob('*')):
                if q.is_file() and '__pycache__' not in q.parts and q.suffix in ('.py','.yaml','.json'):tar.add(q,arcname=directory+'/'+str(q.relative_to(base)))
    ids=data['training_ids']['global'];records=data['records'];vocab=cfg['vocab'];writers=cfg['writers'];targets={}
    with h5py.File(root/DATA/'source.h5') as f:
        for sid in ids:targets[sid]=f[sid]['target'][:]
    protocol=dict(parent=PARENT,source_checkpoints=PARENT_SHA,train_ids=ids,modes=['baseline','pe_only','mask_only','both'],delta_blocks=[-1,1],batch=8,no_training=True,no_confirmation_used=True,oracle_diagnostic_only=True,definition='Same frozen checkpoint/text/writer. ±1block independent changes: PE-only denominator; mask-only context/output length; both; native baseline. Common-prefix drift compared against baseline (not a new index-aligned target when lengths differ). No forced EOC or smoothing.',source_archive_sha256=file_sha(out/'as-run-source.tar.gz'),config=cfg)
    (out/'config.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[];parity=[]
    with h5py.File(out/'evaluation.h5','w') as dest:
        for a in ARMS:
            model=TimingProbe(**cfg['models'][a]);saved=torch.load(p/a/'checkpoint-last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict']);model.eval();digest=tensor_digest(model.state_dict());baselines={}
            for delta,mode in [(0,'baseline')]+[(d,m) for d in (-1,1) for m in ('pe_only','mask_only','both')]:
                policy=mode if delta==0 else mode+f'_{delta:+d}'
                for start in range(0,len(ids),8):
                    batch=ids[start:start+8];oracle=[(records[s]['points']+7)//8 for s in batch];pairs=[(o,o) if mode=='baseline' else intervention_lengths(o,delta,mode) for o in oracle];context=[c for c,_ in pairs];position=[v for _,v in pairs]
                    x,mask,text=inputs([records[s]['text'] for s in batch],context,vocab,'cpu');wi=writer_tensor(batch,records,writers,'cpu');lengths=torch.tensor(position);pred=model(x,torch.ones(len(batch)),text,mask,writer_ids=wi,position_lengths=lengths);z=transform(pred,stats,True)
                    if mode=='baseline':
                        native=model(x,torch.ones(len(batch)),text,mask,writer_ids=wi)
                        if float((native-pred).abs().max())>2e-5:raise ValueError('diagnostic override differs from native forward')
                    for j,sid in enumerate(batch):
                        points,m=decode_sample(codec,reader,z[j,:context[j]],records[sid],vocab);n=min(len(points),len(targets[sid]));old=points if mode=='baseline' else baselines[sid];common=min(len(old),len(points),records[sid]['points']);diff=points[:common,:2]-old[:common,:2]
                        row=dict(sample_id=sid,arm=a,policy=policy,oracle_blocks=oracle[j],context_blocks=context[j],position_blocks=position[j],common_points=common,common_x_drift_rmse=float(np.sqrt((diff[:,0]**2).mean())),common_y_drift_rmse=float(np.sqrt((diff[:,1]**2).mean())),common_pen_changes=int((points[:common,2:].argmax(1)!=old[:common,2:].argmax(1)).sum()),**m)
                        rows.append(row);g=dest.create_group(a+'/'+policy+'/'+sid);g.create_dataset('points',data=points,compression='gzip');g.attrs['row']=json.dumps(row)
                        if mode=='baseline':
                            baselines[sid]=points
                            with h5py.File(p/a/'evaluation-24000.h5') as f:q=f['correct/'+sid];gpu=q['points'][:];oldrow=json.loads(q.attrs['row'])
                            parity.append(dict(arm=a,sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),reader_equal=m['free_decoded']==oldrow['free_decoded']))
                print(dict(arm=a,policy=policy,completed_lines=len(ids)),flush=True)
            if tensor_digest(model.state_dict())!=digest:raise ValueError('frozen mapper drift')
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd:raise ValueError('frozen evaluators drift')
    groups={}
    for a in ARMS:
        groups[a]={}
        for policy in sorted({r['policy'] for r in rows}):
            q=[r for r in rows if r['arm']==a and r['policy']==policy];groups[a][policy]=dict(lines=len(q),cer=sum(r['free_errors'] for r in q)/sum(r['characters'] for r in q),exact=sum(r['free_errors']==0 for r in q),common_x_drift_mean=float(np.mean([r['common_x_drift_rmse'] for r in q])),common_y_drift_mean=float(np.mean([r['common_y_drift_rmse'] for r in q])),common_pen_changes=sum(r['common_pen_changes'] for r in q),missing_eoc=sum(r['first_eoc_point'] is None for r in q))
    summary=dict(protocol=protocol,guards=guards,lines=rows,aggregate=groups,full256_native_cpu_gpu_parity=parity,packed_h5_sha256=file_sha(out/'evaluation.h5'),limitations='TRAIN-only frozen-model intervention; original oracle lengths intentionally used to disentangle mechanisms, NOT available free-generation method. One-block context shortening can truncate final strokes; report PE-only separately. No evidence from sealed confirmation informs weights or selection.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return dict(output=str(out),aggregate=groups)
