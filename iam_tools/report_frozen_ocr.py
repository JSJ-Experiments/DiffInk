"""CPU reload, complete transcripts and marker-free frozen-geometry gallery."""
import html
import json
from pathlib import Path
import numpy as np
import torch
from .writer_expansion import load,device_batches
from .frozen_ocr_study import cache_latents,ocr_evaluate,ablate_inactive_projection
from .pen_ab import file_sha


def verify(directory,repo,root='/data'):
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    result=json.loads((directory/'result.json').read_text());sha=result['selected_sha256']
    model,samples,raw,cfg,vocab,prov=load(Path(repo)/'configs/engineering_english.yaml',repo,root,
        str((directory/'checkpoint-best.pt').relative_to(root)),sha,writer_id=None)
    source=torch.load(root/result['source_rel'],map_location='cpu',weights_only=True)
    saved=torch.load(directory/'checkpoint-best.pt',map_location='cpu',weights_only=True)
    unchanged=all(torch.equal(v.cpu(),source['model_state_dict'][k]) for k,v in model.state_dict().items() if not k.startswith('ocr_model.'))
    original=json.loads((directory/'provenance.json').read_text())
    if prov['splits']!=original['splits'] or prov['samples']!=original['samples'] or not unchanged or file_sha(root/result['source_rel'])!=result['source_sha256']:
        raise AssertionError('CPU reload provenance/frozen codec violation')
    model.eval();model.ocr_model.ctc.zero_infinity=False
    cache=cache_latents(model,device_batches(raw,'cpu'),{i:s[2] for i,s in samples.items()})
    cpu=ocr_evaluate(model,cache,{i:s[2] for i,s in samples.items()},prov['splits'],vocab,directory,'cpu-reload')
    gpu=json.loads((directory/f'ocr-{result["best_step"]}.json').read_text());differences=[]
    for a,b in zip(cpu['lines'],gpu['lines']):
        if a['sample_id']!=b['sample_id']:raise AssertionError('evaluation order changed')
        for kind,items in [('mu',[(a['mu'],b['mu'])]),('sampled',zip(a['sampled'],b['sampled']))]:
            for j,(x,y) in enumerate(items):
                if x['decoded']!=y['decoded']:differences.append(dict(sample_id=a['sample_id'],kind=kind,draw=j,cpu=x['decoded'],gpu=y['decoded']))
    # CPU/CUDA randn streams differ: posterior decoding equality is NOT expected
    # cross-device. Mean predictions should be stable; sampled aggregate reported.
    mean_differences=[r for r in differences if r['kind']=='mu']
    check=dict(selected_sha256=sha,selected_updates=saved['ctc_head_updates'],all_non_ocr_state_bitwise_unchanged=unchanged,
        source_unchanged=True,all_mean_transcripts_equal_cpu_gpu=not mean_differences,mean_transcript_differences=mean_differences,
        posterior_cpu_gpu_transcript_differences=len(differences)-len(mean_differences),posterior_noise_is_not_cross_device_paired=True,
        cpu_groups=cpu['groups'],gpu_groups=gpu['groups'])
    (directory/'cpu-reload-check.json').write_text(json.dumps(check,indent=2)+'\n')
    # SAME CPU noise before/after. This head-only ablation never changes the codec.
    original_head={k:v.detach().clone() for k,v in model.ocr_model.state_dict().items()}
    training=[cache[i] for i in prov['splits']['train']]
    stats=dict(train_max_abs_unused_mu=max(float(c['mu'][:,40:].abs().max()) for c in training),
        train_mean_unused_mu_rms=float(np.mean([float(c['mu'][:,40:].square().mean().sqrt()) for c in training])),
        train_mean_unused_posterior_std=float(np.mean([float((.5*c['lv'][:,40:]).exp().mean()) for c in training])))
    # Body learning has made unused means tiny but NOT analytically zero. This
    # exploratory ablation must measure changes, not assume a safe exact pruning.
    info=ablate_inactive_projection(model,require_exact=False)
    info['latent_statistics']=stats
    ablated=ocr_evaluate(model,cache,{i:s[2] for i,s in samples.items()},prov['splits'],vocab,directory,'inactive-projection')
    identical=all(a['mu']['decoded']==b['mu']['decoded'] for a,b in zip(cpu['lines'],ablated['lines']))
    changes=[dict(sample_id=a['sample_id'],before=a['mu']['decoded'],after=b['mu']['decoded']) for a,b in zip(cpu['lines'],ablated['lines']) if a['mu']['decoded']!=b['mu']['decoded']]
    info.update(all_mean_transcripts_unchanged=identical,mean_transcript_changes=changes,before_cpu=cpu['groups'],after_cpu=ablated['groups'],paired_cpu_noise=True)
    diagnostic=dict(saved);diagnostic['model_state_dict']=model.state_dict()
    diagnostic['config']=dict(saved['config'],ocr_inactive_projection_ablation=info['inactive_channels'],diagnostic_only=True)
    diagnostic.pop('ocr_optimizer_state_dict',None)
    torch.save(diagnostic,directory/'checkpoint-inactive-projection.pt')
    info['diagnostic_checkpoint_sha256']=file_sha(directory/'checkpoint-inactive-projection.pt')
    (directory/'inactive-projection-ablation.json').write_text(json.dumps(info,indent=2)+'\n')
    check['inactive_projection_ablation']=info
    model.ocr_model.load_state_dict(original_head)
    for r in mean_differences:
        c=cache[r['sample_id']]
        with torch.no_grad():
            logits=model.ocr_model(c['mu'],padding_mask=~c['mask'])[:int(c['mask'].sum()),0]
            top=logits.topk(2,dim=-1).values
            r['cpu_min_top_two_logit_margin']=float((top[:,0]-top[:,1]).min())
        r['split']='train' if r['sample_id'] in prov['splits']['train'] else 'held_out'
    check['all_training_mean_transcripts_equal_cpu_gpu']=all(r['split']=='held_out' for r in mean_differences)
    (directory/'cpu-reload-check.json').write_text(json.dumps(check,indent=2)+'\n')
    if any(r['split']=='train' or r['cpu_min_top_two_logit_margin']>2e-4 for r in mean_differences):
        raise AssertionError('unexpected mean OCR CPU/CUDA discrepancy; investigate before reporting')
    return check


def report(directory,repo,root='/data'):
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    directory=Path(directory);root=Path(root);result=json.loads((directory/'result.json').read_text())
    settings=json.loads((directory/'config.json').read_text())
    source_updates=settings.get('source_ocr_updates',0)
    metadata=dict(recorded_schedule=settings.get('training_schedule'),
        actual_schedule=f'seed42 bucketed complete OCR epochs; skip{source_updates} previous OCR updates',
        recorded_wall_seconds=settings.get('max_wall_seconds'),actual_training_wall_seconds=900,recorded_optimizer=settings.get('optimizer'),actual_optimizer='restored head AdamW moments; constant LR1e-4' if source_updates else 'fresh head AdamW',
        note='As-run JSON inherited prior-codec schedule/optimizer text in the first version; actual source-code snapshots and update/sample logs define execution. Original JSON preserved; clarification does not alter checkpoints or metrics.')
    (directory/'metadata-clarification.json').write_text(json.dumps(metadata,indent=2)+'\n')
    settings['optimizer']=metadata['actual_optimizer']
    check=verify(directory,repo,root);prov=json.loads((directory/'provenance.json').read_text());splits=prov['splits'];step=result['best_step']
    before=json.loads((directory/'ocr-0.json').read_text());after=json.loads((directory/f'ocr-{step}.json').read_text())
    out=directory/'report';out.mkdir(exist_ok=True);targets={}
    for kind,ids in [('train',splits['train']),('val',splits['held_out'])]:
        with h5py.File(root/f'diffink/iam_overfit/tiny_{kind}.h5') as hf:
            for sid in ids:
                a=hf[sid]['point_seq'][:].copy();a[:,:2]*=.01;targets[sid]=a
    def arrays(sid):
        a=np.load(directory/f'geometry-source/step-0/{sid}/mu.npy')
        b=np.load(directory/f'geometry-selected/step-{step}/{sid}/mu.npy')
        if not np.array_equal(a,b):raise AssertionError('gallery geometry drift')
        return [('IAM/RDP target',targets[sid]),('before frozen OCR',a),('after frozen OCR',b)]
    pages=[]
    for group in ('old','new','held_out'):
        for page,ids in enumerate(paginate_ids(splits[group]),1):
            fig,axs=plt.subplots(len(ids),3,figsize=(15,2*len(ids)),squeeze=False)
            for j,sid in enumerate(ids):
                lo=targets[sid][:,:2].min(0);hi=targets[sid][:,:2].max(0)
                for ax,(label,a) in zip(axs[j],arrays(sid)):
                    draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' — '+label,fontsize=8)
            fig.tight_layout();name=f'{group}-{page}.png';fig.savefig(out/name,dpi=130);plt.close(fig);pages.append((group,page,name))
    fig,axs=plt.subplots(2,3,figsize=(12,6),squeeze=False)
    for j,(sid,xlim,ylim) in enumerate([('p08-936z-05',(4.94,5.34),(.15,.57)),('a07-421z-02',(8.28,8.75),(.40,.97))]):
        for ax,(label,a) in zip(axs[j],arrays(sid)):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(sid+' — '+label,fontsize=8)
    fig.tight_layout();fig.savefig(out/'user-regions.png',dpi=170);plt.close(fig)
    data=dict(result=result,cpu_reload=check,before=before['groups'],after=after['groups']);(out/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    chunks=['<meta charset="utf-8"><title>Frozen English codec OCR refit</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto}img{max-width:100%}td{padding:.3em}pre{white-space:pre-wrap}</style>',
        '<h1>Frozen codec OCR refit — geometry is EXACTLY unchanged</h1>',
        '<p>Only OCR parameters train. Head sees cached minimally padded individual-line latents, right-padded OCR batches16 with attention masks. AdamW, betas.9/.99, weight decay1e-4, clip5, OCR dropout.1. No raw trajectory batching, no geometry/readout/pen/posterior/style updates. This is head fitting, NOT useful generative-latent proof or joint regularization.</p>',
        '<p>Training latent: '+html.escape(settings['ocr_training_latent'])+'; LR '+str(settings['base_lr'])+' → '+str(settings['final_lr'])+'; initial blank bias '+str(settings['initial_blank_bias'])+'. '+html.escape(settings.get('optimizer','fresh head optimizer'))+'</p>',
        f'<p>Selected continuation update{step} (total OCR updates{check["selected_updates"]}) using TRAIN CER/CTC only. Held-out32 lines from the same8 writers; forms overlap, not a writer/form-independent benchmark. 20 fixed GPU posterior draws/line per evaluation. CPU reload uses different device RNG, so sampled CPU/CUDA draws are not paired.</p>',
        '<p>All non-OCR state tensors are bitwise source-identical. All4704 saved mean/posterior trajectories before/after are bitwise identical. Their preserved IAM/RDP corners are not smoothed. Frozen source: '+html.escape(result['source_rel'])+' SHA '+result['source_sha256']+'</p>',
        '<p><a href="summary.json">Metrics/provenance/reload check</a> · <a href="../provenance.json">All224 line IDs and hashes</a> · <a href="../config.json">Exact configuration</a></p>',
        '<table border="1"><tr><th>stage/group</th><th>mean CER / exact</th><th>sampled CER / exact draws</th><th>repeat/nonrepeat mean CER</th><th>CTC</th></tr>']
    for label,row in [('before',before),('selected',after)]:
        for group in ('train','held_out','old'):
            s=row['groups'][group];m,z=s['mu'],s['sampled']
            chunks.append(f'<tr><td>{label}/{group}</td><td>{m["cer"]:.4%} / {m["exact_lines"]}/{m["evaluations"]}</td><td>{z["cer"]:.4%} / {z["exact_lines"]}/{z["evaluations"]}</td><td>{m["repeats"]["cer"]} / {m["no_repeats"]["cer"]}</td><td>{s["mean_ctc_loss"]:.6f}</td></tr>')
    a=check['inactive_projection_ablation'];chunks.append('</table><h2>Paired no-training inactive-channel OCR ablation</h2><p>The384-D codec initially reserved344 unused mean channels with posterior std≈1. Body learning made their means tiny but NOT analytically zero. Mean-only fitting gives little supervision to reject their much larger posterior noise. This exploratory ablation zeros ONLY those OCR projection columns and measures mean/sample changes; it does not assume exact invariance. The codec/latent distribution is untouched. This is specific to the initialized transport, not a general latent-pruning recommendation. Diagnostic weights saved separately; no claim of validation generalization.</p><table border="1"><tr><th>group</th><th>before sampled CER</th><th>after sampled CER</th><th>mean CER after ablation</th></tr>')
    chunks.append('<caption>Training unused mean maxabs='+str(a['latent_statistics']['train_max_abs_unused_mu'])+'; mean unused std='+str(a['latent_statistics']['train_mean_unused_posterior_std'])+'</caption>')
    for group in ('train','held_out'):
        b,c=a['before_cpu'][group],a['after_cpu'][group]
        chunks.append(f'<tr><td>{group}</td><td>{b["sampled"]["cer"]:.4%}</td><td>{c["sampled"]["cer"]:.4%}</td><td>{c["mu"]["cer"]:.4%}</td></tr>')
    chunks.append('</table><h2>Named curve regions — no change expected or observed</h2><img src="user-regions.png"><h2>Complete transcript evaluation</h2>')
    bmap={r['sample_id']:r for r in before['lines']}
    for group in ('train','held_out'):
        chunks.append('<h3>'+group+'</h3><table border="1"><tr><th>ID</th><th>truth</th><th>before</th><th>selected mean</th><th>sampled CER</th></tr>')
        for r in after['lines']:
            sid=r['sample_id']
            if sid not in splits[group]:continue
            cer=sum(v['errors'] for v in r['sampled'])/sum(v['characters'] for v in r['sampled'])
            chunks.append('<tr><td>'+html.escape(sid)+'</td><td>'+html.escape(r['text'])+'</td><td>'+html.escape(bmap[sid]['mu']['decoded'])+'</td><td>'+html.escape(r['mu']['decoded'])+f'</td><td>{cer:.3%}</td></tr>')
        chunks.append('</table>')
    for group,page,name in pages:chunks.append(f'<h2>{group} page{page}: target / before / after</h2><img loading="lazy" src="{name}">')
    (out/'index.html').write_text('\n'.join(chunks))
    from .report_pointer import publish_pointer
    publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),best_step=step,selected_sha256=result['selected_sha256'],groups={g:after['groups'][g] for g in ('train','held_out')})


def research_summary(family,root='/data'):
    """Explicit completed studies only; compare paired schedules and source hashes."""
    family=Path(family);root=Path(root)
    stages={'first_mu':'20261007-014901','paired_mu':'20261007-015617','paired_mu_z':'20261007-015548'}
    results={k:json.loads((family/v/'result.json').read_text()) for k,v in stages.items()}
    configs={k:json.loads((family/v/'config.json').read_text()) for k,v in stages.items()}
    for k,date in stages.items():
        settings=configs[k];previous=settings.get('source_ocr_updates',0)
        note=dict(recorded_schedule=settings.get('training_schedule'),actual_schedule=f'seed42 bucketed complete OCR epochs; skip{previous} source OCR updates',
            recorded_wall_seconds=settings.get('max_wall_seconds'),actual_training_wall_seconds=900,recorded_optimizer=settings.get('optimizer'),actual_optimizer='restored head AdamW moments; constant LR1e-4' if previous else 'fresh head AdamW',
            note='Prior-stage textual metadata was inherited in the first runner version; source snapshots and per-update sample logs define actual execution. Original config/checkpoints preserved.')
        (family/date/'metadata-clarification.json').write_text(json.dumps(note,indent=2)+'\n')
    def readlog(stage):return [json.loads(l) for l in (family/stages[stage]/'metrics.jsonl').read_text().splitlines()]
    a,b=readlog('paired_mu'),readlog('paired_mu_z')
    paired=dict(source_sha256=results['paired_mu']['source_sha256'],same_source=results['paired_mu']['source_sha256']==results['paired_mu_z']['source_sha256'],
        same_initial_ocr=json.loads((family/stages['paired_mu']/'ocr-0.json').read_text())==json.loads((family/stages['paired_mu_z']/'ocr-0.json').read_text()),
        same_training_sample_schedule=[r['sample_ids'] for r in a]==[r['sample_ids'] for r in b],same_lrs=[r['lr'] for r in a]==[r['lr'] for r in b],updates=len(a))
    if not all(paired[k] for k in ('same_source','same_initial_ocr','same_training_sample_schedule','same_lrs')) or len(a)!=len(b):raise AssertionError(paired)
    out=family/'research-summary';out.mkdir(exist_ok=True)
    ablations={k:json.loads((family/v/'inactive-projection-ablation.json').read_text()) for k,v in stages.items()}
    checks={k:json.loads((family/v/'cpu-reload-check.json').read_text()) for k,v in stages.items()}
    data=dict(stages=stages,results=results,configs=configs,paired_continuation_audit=paired,ablations=ablations,cpu_reload=checks)
    (out/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    chunks=['<meta charset="utf-8"><title>Faithful codec: frozen OCR research summary</title><style>body{font-family:sans-serif;max-width:1300px;margin:auto}td{padding:.4em}img{max-width:100%}</style>',
        '<h1>Frozen English codec OCR — memorization versus generalization</h1>',
        '<p>Geometry is not altered: all non-OCR parameters/buffers unchanged and all4704 saved before/after mean/posterior trajectories bitwise identical in EVERY study. No new jaggedness or pen/EOC failures. Readout, encoder, posterior, KL/style objectives are not optimized here. Frozen source geometry remains the protected codec9c53f68.</p>',
        '<p>First fit:1000 updates, head AdamW5e-4 →1e-4 at750, dropout.1, blank bias0, cached mean inputs. Paired continuations:500 more updates from the exact same step1000 weights/Adam/RNG, LR1e-4. One is two mean forwards; the other is half mean+half sampled-z CTC. Both consume identical noise RNG and dropout forwards. Batch16 only in the mask-aware OCR head; each raw line was encoded separately with minimal padding. Held-out32 same-writer lines/forms overlap; no held-out optimization/selection.</p>',
        '<p>The head now memorizes all192 mean transcripts, but held-out CER≈83–84% remains BAD. This does not demonstrate generalization, useful semantic latents, successful joint training, or readiness for InkDiT. Frozen transport coordinates preserve geometry, not automatically English semantics.</p>',
        '<p><a href="summary.json">All metrics, hashes, exact configurations and pairing checks</a></p><table border="1"><tr><th>study</th><th>train mean CER / exact</th><th>train sampled CER</th><th>held-out mean/sample CER</th><th>train CTC</th><th>mean unchanged after projection ablation?</th></tr>']
    for key,date in stages.items():
        r=results[key];row=json.loads((family/date/f'ocr-{r["best_step"]}.json').read_text());t,h=row['groups']['train'],row['groups']['held_out']
        chunks.append(f'<tr><td><a href="../{date}/report/index.html">{key}</a></td><td>{t["mu"]["cer"]:.4%} / {t["mu"]["exact_lines"]}/192</td><td>{t["sampled"]["cer"]:.4%}</td><td>{h["mu"]["cer"]:.4%}/{h["sampled"]["cer"]:.4%}</td><td>{t["mean_ctc_loss"]:.6f}</td><td>{ablations[key]["all_mean_transcripts_unchanged"]}</td></tr>')
    chunks.append('</table><h2>Why sampled OCR failed while geometry was faithful</h2><p>344 unused channels have tiny but NONZERO means (training max1.38e-6/RMS1.74e-7) and posterior std≈0.9999. Mean fitting offers almost no signal to suppress their inherited projection weights. A paired no-training ablation zeros ONLY these OCR input-projection columns: codec, latent distribution and geometry are unmodified. Effects measured, not assumed; every prepared mean transcript stayed unchanged. This is codec-specific evidence, NOT a general recipe for pruning learned latents.</p><table border="1"><tr><th>study</th><th>CPU paired train sampled CER before/after</th><th>held-out sampled CER before/after</th><th>diagnostic checkpoint SHA</th></tr>')
    for k,a in ablations.items():
        t,h=a['before_cpu']['train'],a['before_cpu']['held_out'];u,v=a['after_cpu']['train'],a['after_cpu']['held_out']
        chunks.append(f'<tr><td>{k}</td><td>{t["sampled"]["cer"]:.4%} →{u["sampled"]["cer"]:.4%}</td><td>{h["sampled"]["cer"]:.4%} →{v["sampled"]["cer"]:.4%}</td><td>{a["diagnostic_checkpoint_sha256"]}</td></tr>')
    chunks.append('</table><p>Mean-only control + this projection ablation:192/192 training means and3840/3840 CPU posterior draws have CER0. The independent sampled-aware continuation also learns robustness:59.9% →3.75% training sampled CER versus61.24% for its matched mean-only control. Neither intervention solves held-out reading.</p>')
    chunks.append('<h2>Frozen target fidelity: named c/h regions</h2><p>These images intentionally do NOT change across head fitting. Remaining target polygonality at high zoom is IAM/RDP itself; no generic smoothing applied.</p><img src="../20261007-015617/report/user-regions.png"><h2>CPU/CUDA and provenance caveats</h2><ul>')
    for k,c in checks.items():
        chunks.append(f'<li>{k}: CPU/CUDA mean differences {len(c["mean_transcript_differences"])}; all non-OCR weights source-identical. Posterior random draws differ across device RNGs; only within-device ablations are paired.</li>')
    chunks.append('</ul><p>Sampled-aware head has one held-out CPU/CUDA mean transcript difference on an ambiguous frame; training mean CER unchanged. Exact text differences/margin are in its CPU reload audit, not silently hidden. Initial as-run configs inherited outdated codec schedule/optimizer text: separate metadata-clarification.json explains actual execution from source snapshots and sample logs; originals remain preserved.</p><h2>Next useful experiment</h2><p>Do NOT spend more updates memorizing these192 lines. Test whether controlled local-context/translation invariance or a larger TRAIN-only supervision pool improves unseen-transcript reading while keeping the codec frozen. Diagnose line-level memorization before enabling unconstrained encoder/readout or joint OCR/KL/style. The geometry gate remains passed.</p>')
    (out/'index.html').write_text('\n'.join(chunks))
    from .report_pointer import publish_pointer
    publish_pointer(family,out)
    return dict(output=str(out/'index.html'),paired_continuation_audit=paired)
