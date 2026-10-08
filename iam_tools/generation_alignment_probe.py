"""CPU-only TRAIN weak-alignment probe on a frozen absolute-query candidate.

CTC path is forced through the requested transcript: NOT independent correctness
or true IAM character boundaries. Reader is corpus-familiar and bidirectional.
This probe selects no model/checkpoint and generates no new blind prompts.
"""
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from .ctc_alignment import forced_ctc
from .generation_position_contract import PositionContractWriter
from .generation_capacity import DATA
from .generation_cache import CachedLatentPool
from .generation_coverage_study import writer_tensor
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .inkvae import greedy_ctc,edit_distance
from .pen_ab import file_sha
from .ocr_context_study import tensor_digest

SOURCE='checkpoints/iam_generation_position_contract/20261008-101747'


def quantiles(x):
    a=np.abs(np.asarray(x))
    return dict(median=float(np.median(a)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)))


def probe(repo,root='data'):
    root=Path(root);p=root/SOURCE;cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text())
    result=json.loads((p/'absolute/result.json').read_text());path=p/'absolute/checkpoint-best.pt'
    if file_sha(path)!=result['selected_sha256']:raise ValueError('frozen TRAIN-selected candidate guard')
    saved=torch.load(path,map_location='cpu',weights_only=False);model=PositionContractWriter(**cfg['models']['absolute']).eval();model.load_state_dict(saved['model_state_dict'])
    torch.set_num_threads(2);codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    reader,_=load_reader(root,cc);rd=tensor_digest(reader.state_dict());md=tensor_digest(model.state_dict())
    out=root/'checkpoints/iam_generation_alignment_probe'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'probe-source.py').write_bytes(Path(__file__).read_bytes());(out/'alignment-source.py').write_bytes(Path(__file__).with_name('ctc_alignment.py').read_bytes())
    stats=torch.load(root/DATA/'whitening.pt',weights_only=True);ids=data['splits']['all_train256'];records=data['records'];latents={}
    with h5py.File(root/DATA/'source.h5') as f:
        for sid in ids:latents[sid]=torch.tensor(f[sid]['latent_mean'][:])
    pool=CachedLatentPool(latents,records,cfg['vocab'],ids,stats,device='cpu',max_lines=256);attention={};rows=[];prior_errors=[];learned_errors=[]
    def capture(module,args,kwargs):
        # Call .forward directly, NOT module(...): no recursive pre-hook. Compute
        # descriptive weights separately; normal SDPA model forward is unchanged.
        with torch.no_grad():
            options=dict(kwargs,need_weights=True,average_attn_weights=False)
            attention['last']=module.forward(*args,**options)[1].detach()
    hook=model.blocks[-1].cross_attention.register_forward_pre_hook(capture,with_kwargs=True)
    try:
        with torch.no_grad(),h5py.File(out/'alignment.h5','w') as dest:
            for start in range(0,len(ids),8):
                batch=ids[start:start+8];clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,cfg['writers'],'cpu')
                # Reader receives actual frozen-codec latent means + original masks.
                raw=torch.zeros_like(clean)
                for j,sid in enumerate(batch):raw[j,:len(latents[sid])]=latents[sid]
                logits=reader(raw.transpose(1,2),padding_mask=~mask,point_mask=pm)
                model(torch.zeros_like(clean),torch.ones(len(batch)),text,mask,writer_ids=wi)
                for j,sid in enumerate(batch):
                    r=records[sid];n=r['points'];frames=(n+3)//4;l=len(latents[sid]);labels=[cfg['vocab'].index(c)+1 for c in r['text']]
                    logp=logits[:frames,j].log_softmax(-1).numpy();path=forced_ctc(logp,labels);token=path['token_indices'];valid=token>=0
                    frame=np.arange(frames);block=frame//2;center=np.minimum(block/cfg['duration_model']['blocks_per_character'],len(labels)-1)
                    weights=attention['last'][j,:,:l,1:len(labels)+1].numpy();char=np.arange(len(labels))
                    expected=(weights*char).sum(-1)/weights.sum(-1).clip(1e-12)
                    head_error=np.abs(expected[:,block][:,valid]-token[valid][None])
                    # Three local heads versus the untouched global head, separately.
                    pe=center[valid]-token[valid];le=expected[:3,block][:,valid].mean(0)-token[valid]
                    prior_errors.extend(pe.tolist());learned_errors.extend(le.tolist())
                    decoded=greedy_ctc(logp.argmax(1).tolist(),cfg['vocab'])
                    row=dict(sample_id=sid,text=r['text'],writer_id=r['writer_id'],frames=frames,forced_nonblank_frames=int(valid.sum()),reader_text=decoded,reader_errors=edit_distance(r['text'],decoded),characters=len(r['text']),
                        prior_token_index_error=quantiles(pe),prior_outside_sigma_fraction=float((np.abs(pe)>2.5).mean()),
                        final_local_attention_mean_token_index_error=quantiles(le),heads=[quantiles(e) for e in head_error],
                        forced_path_log_probability=path['log_probability'],unconstrained_best_frame_log_probability=float(logp.max(1).sum()))
                    rows.append(row);g=dest.create_group(sid);g.create_dataset('forced_states',data=path['states']);g.create_dataset('forced_token_indices',data=token)
                    g.create_dataset('prior_center',data=center);g.create_dataset('last_attention_expected_char_index',data=expected,compression='gzip');g.create_dataset('last_attention',data=weights,compression='gzip');g.attrs['row']=json.dumps(row)
    finally:hook.remove()
    if tensor_digest(reader.state_dict())!=rd or tensor_digest(model.state_dict())!=md:raise ValueError('frozen diagnostic state drift')
    summary=dict(parent=SOURCE,checkpoint_sha256=result['selected_sha256'],reader_sha256=cfg['reader_sha256'],source_h5_sha256=file_sha(root/DATA/'source.h5'),alignment_h5_sha256=file_sha(out/'alignment.h5'),lines=rows,
        aggregate=dict(lines=len(rows),reader_cer=sum(r['reader_errors'] for r in rows)/sum(r['characters'] for r in rows),reader_exact=sum(r['reader_errors']==0 for r in rows),prior_index_error=quantiles(prior_errors),prior_outside_sigma_fraction=float((np.abs(prior_errors)>2.5).mean()),last_local_attention_mean_index_error=quantiles(learned_errors)),
        definitions='CTC emissions every4 processed indices, compare forced nonblank token INDEX against static prior and final cross-attention conditional-character mean at floor(frame/2) packed8 queries. Not physical time, exact character boundaries or demonstrated causal character routing. Blank frames omitted from index-error metric, not sources from reader CER. Attention has no character-identity ground truth.',
        caveats='TRAIN256 only; corpus-familiar bidirectional reader; forced labels can fit even reader errors. Delayed marks and context lookahead mean this is WEAK alignment, not IAM segmentation. No selection, updates, held source opening or new composition claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(dict(output=str(out),aggregate=summary['aggregate']),flush=True);return str(out)
