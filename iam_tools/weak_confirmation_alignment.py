"""Post-opening frozen weak timing probe; no updates or candidate selection.

Source-reader CTC positions are weak reference timing, NOT exact IAM borders.
Different legitimate generated trajectories may have different timing. Cannot
call source-index agreement independent proof of correct glyph generation.
"""
import json
from pathlib import Path
import h5py,numpy as np,torch
from .ctc_alignment import forced_ctc
from .weak_alignment import packed_labels,capture_last_alignment,alignment_summary
from .generation_prefix_contract import PrefixContractWriter
from .generation_cache import CachedLatentPool
from .generation_duration_eval import inputs
from .generation_coverage_study import writer_tensor
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .inkvae import greedy_ctc,edit_distance
from .pen_ab import file_sha
from .ocr_context_study import tensor_digest


def require_opened(directory):
    p=Path(directory)/'confirmation/summary.json'
    if not p.is_file():raise ValueError('posthoc timing probe requires completed one-shot confirmation first')
    s=json.loads(p.read_text())
    if set(s['arms'])!={'control','weak_alignment'} or len(s['preflight'])!=16:raise ValueError('complete paired confirmation required')
    return s


@torch.no_grad()
def probe(directory,repo,root='data'):
    p=Path(directory);root=Path(root);opened=require_opened(p);cfg=json.loads((p/'config.json').read_text());records=opened['reservation']['paired']['records'];ids=opened['reservation']['paired']['ids']
    source=p/'confirmation/source.h5'
    if file_sha(source)!=opened['source_h5_sha256']:raise ValueError('immutable opened source guard')
    torch.set_num_threads(2);out=p/'confirmation-timing-probe';out.mkdir(exist_ok=False)
    (out/'probe-source.py').write_bytes(Path(__file__).read_bytes())
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);reader,_=load_reader(root,cc);rd=tensor_digest(reader.state_dict())
    from .generation_capacity import DATA
    stats=torch.load(root/DATA/'whitening.pt',weights_only=True);latents={}
    with h5py.File(source) as f:
        for s in ids:latents[s]=torch.tensor(f[s]['latent_mean'][:])
    pool=CachedLatentPool(latents,records,cfg['vocab'],ids,stats,device='cpu',max_lines=16);paths={};teacher_rows=[]
    for start in range(0,16,8):
        batch=ids[start:start+8];clean,mask,text,pm=pool.select(batch);raw=torch.zeros_like(clean)
        for j,s in enumerate(batch):raw[j,:len(latents[s])]=latents[s]
        logits=reader(raw.transpose(1,2),padding_mask=~mask,point_mask=pm)
        for j,s in enumerate(batch):
            r=records[s];frames=(r['points']+3)//4;logp=logits[:frames,j].log_softmax(-1).numpy();labels=[cfg['vocab'].index(c)+1 for c in r['text']];path=forced_ctc(logp,labels)
            paths[s]=packed_labels(path['token_indices'],len(r['text']),r['points']);decoded=greedy_ctc(logp.argmax(-1).tolist(),cfg['vocab']);teacher_rows.append(dict(sample_id=s,reader_text=decoded,reader_errors=edit_distance(r['text'],decoded),characters=len(r['text'])))
    results={}
    with h5py.File(out/'attention.h5','w') as dest:
        for arm in ['control','weak_alignment']:
            cp=p/arm/'checkpoint-best.pt'
            if file_sha(cp)!=opened['arms'][arm]['checkpoint_sha256']:raise ValueError('unchanged frozen one-shot candidate required')
            m=PrefixContractWriter(**cfg['models'][arm]).eval();m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model_state_dict']);digest=tensor_digest(m.state_dict());rng=torch.get_rng_state().clone();rows=[]
            for start in range(0,16,8):
                batch=ids[start:start+8];x,mask,text=inputs([records[s]['text'] for s in batch],[256]*len(batch),cfg['vocab'],'cpu');wi=writer_tensor(batch,records,cfg['writers'],'cpu')
                with capture_last_alignment(m) as c:m(x,torch.ones(len(batch)),text,mask,writer_ids=wi)
                labels=torch.full((len(batch),256,2),-1,dtype=torch.long)
                for j,s in enumerate(batch):labels[j,:len(paths[s])]=torch.tensor(paths[s])
                for j,s in enumerate(batch):
                    row=dict(sample_id=s,**alignment_summary(c['log_probs'][j:j+1],labels[j:j+1],mask[j:j+1],text[j:j+1]));rows.append(row)
                    g=dest.create_group(arm+'/'+s);n=len(paths[s]);count=len(records[s]['text']);g.create_dataset('reference_forced_token_indices',data=paths[s]);g.create_dataset('attention_probability',data=c['log_probs'][j,:,:n,1:count+1].exp().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
            if tensor_digest(m.state_dict())!=digest or not torch.equal(rng,torch.get_rng_state()):raise ValueError('frozen timing probe state/RNG drift')
            total=sum(r['supervised_queries'] for r in rows);results[arm]=dict(lines=rows,aggregate=dict(lines=16,supervised_queries=total,cross_entropy=sum(r['cross_entropy']*r['supervised_queries'] for r in rows)/total,target_mass=sum(r['target_mass']*r['supervised_queries'] for r in rows)/total,mean_line_index_error={k:float(np.mean([r['mean_index_error'][k] for r in rows])) for k in ['median','p90','p99']}))
    if tensor_digest(reader.state_dict())!=rd:raise ValueError('frozen reader drift')
    s=dict(arms=results,source_reader=teacher_rows,confirmation_summary_sha256=file_sha(p/'confirmation/summary.json'),source_sha256=file_sha(source),packed_h5_sha256=file_sha(out/'attention.h5'),definitions='Source forced-reader timing every4 processed indices aggregated into packed8 queries. Both frozen candidates use SAME generous256query budget, no teacher/masks/source durations at inference. Compare last3local heads to reference source timing ONLY after one-shot confirmation opening.',limitations='Not true IAM glyph borders, not generated-trajectory alignment labels, bidirectional corpus-familiar reader, legitimate generated timing can differ from source. Posthoc mechanism diagnosis, NEVER new model selection or composition proof; all16retained. No training/selection/held exclusions.')
    (out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');render(out,records,ids,s);print({a:r['aggregate'] for a,r in results.items()},flush=True);return str(out/'index.html')


def render(out,records,ids,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pages=[]
    with h5py.File(out/'attention.h5') as f:
        for start in range(0,16,4):
            group=ids[start:start+4];fig,axs=plt.subplots(len(group),2,figsize=(18,4*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                for ax,arm in zip(axs[j],['control','weak_alignment']):
                    g=f[arm+'/'+sid];p=g['attention_probability'][:3].mean(0);p=p/p.sum(-1,keepdims=True).clip(1e-30);labels=g['reference_forced_token_indices'][:]
                    ax.imshow(p,aspect='auto',origin='upper',vmin=0,vmax=1,extent=(-.5,p.shape[1]-.5,p.shape[0]-.5,-.5),cmap='Blues')
                    for q in [0,1]:
                        good=labels[:,q]>=0;ax.scatter(labels[good,q],np.arange(len(labels))[good],s=8,c='orange')
                    ax.set_xticks(range(len(records[sid]['text'])));ax.set_xticklabels(list(records[sid]['text']),fontsize=6);ax.set_title(arm+' / '+sid+'\n'+records[sid]['text'],fontsize=8);ax.set_ylabel('processed query index, NOT physical time')
            fig.tight_layout();name=f'timing-{start//4+1}.png';fig.savefig(out/name,dpi=110);plt.close(fig);pages.append(name)
    body='<!doctype html><meta charset="utf-8"><title>Frozen new-source weak timing diagnosis</title><style>body{font:17px system-ui;max-width:1800px;margin:30px auto}img{width:100%}</style><h1>Post-opening weak timing, NOT exact glyph alignment</h1><p>'+s['definitions']+'</p><p>'+s['limitations']+'</p><p>Blue: final-local-head conditional-character attention; orange: source forced-reader label emissions. Orange is a weak reference, not generated glyph truth.</p><a href="summary.json">All16 metrics/source-reader guards</a>'
    for name in pages:body+=f'<img loading="lazy" src="{name}">'
    (out/'index.html').write_text(body)
