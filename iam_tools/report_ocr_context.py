"""Reload standalone research heads and report all context-control arms fairly."""
import html,json
from pathlib import Path
import torch
from .ocr_context_study import ARMS,SOURCE,SHA,head_model,tensor_digest
from .ocr_context_features import make_head
from .frozen_ocr_study import cache_latents,ocr_evaluate
from .writer_expansion import load,device_batches
from .pen_ab import file_sha


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    from .report_pointer import publish_pointer
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    results=json.loads((directory/'result.json').read_text())
    if set(results)!=set(ARMS):raise AssertionError('require all four completed arms')
    base,samples,raw,cfg,vocab,prov=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    expected=json.loads((directory/'provenance.json').read_text())
    if prov['splits']!=expected['splits'] or prov['samples']!=expected['samples']:raise AssertionError('dataset provenance drift')
    base.eval().requires_grad_(False);initial=tensor_digest(base.state_dict());texts={i:s[2] for i,s in samples.items()}
    cache=cache_latents(base,device_batches(raw,'cpu'),texts);checks={};evaluations={};gpu={};configs={};schedules={}
    for name in ARMS:
        arm=directory/name;settings=json.loads((arm/'config.json').read_text());configs[name]=settings;r=results[name]
        if settings['source_rel']!=SOURCE or settings['source_sha256']!=SHA or file_sha(arm/'head-best.pt')!=r['selected_sha256']:
            raise AssertionError('head/source checkpoint mismatch')
        saved=torch.load(arm/'head-best.pt',map_location='cpu',weights_only=True)
        if saved['config']!=settings or saved['updates']!=r['best_step']:raise AssertionError('head feature contract mismatch')
        head=make_head(cfg,len(vocab)+1,settings['mode'],settings['feature_stats'],settings['attention_radius'],seed=42)
        cpu_initial_digest=tensor_digest(head.state_dict())
        buffer_differences={}
        # The sinusoidal positional table is a registered CPU-generated buffer.
        # Different host SIMD math can change its last bits despite identical RNG.
        # Compare RANDOM PARAMETERS separately, using the as-run saved buffers.
        with torch.no_grad():
            for key,buffer in head.named_buffers():
                original=saved['ocr_state_dict'][key]
                buffer_differences[key]=float((buffer-original).abs().max())
                buffer.copy_(original)
        if tensor_digest(head.state_dict())!=r['initial_head_tensor_sha256']:
            raise AssertionError('random parameter initialization differs after restoring nonrandom buffers')
        head.load_state_dict(saved['ocr_state_dict'],strict=True);head.eval()
        cpu=ocr_evaluate(head_model(head),cache,texts,prov['splits'],vocab,arm,'cpu-reload')
        selected=json.loads((arm/f'ocr-{r["best_step"]}.json').read_text());gpu[name]=selected;evaluations[name]=cpu
        differences=[]
        for a,b in zip(cpu['lines'],selected['lines']):
            if a['sample_id']!=b['sample_id']:raise AssertionError('evaluation order drift')
            if a['mu']['decoded']!=b['mu']['decoded']:
                c=cache[a['sample_id']]
                with torch.no_grad():
                    logits=head(c['mu'],padding_mask=~c['mask'])[:int(c['mask'].sum()),0];top=logits.topk(2,dim=-1).values
                differences.append(dict(sample_id=a['sample_id'],cpu=a['mu']['decoded'],gpu=b['mu']['decoded'],
                    cpu_min_top_two_logit_gap=float((top[:,0]-top[:,1]).min()),split='train' if a['sample_id'] in prov['splits']['train'] else 'held_out'))
        check=dict(checkpoint_sha256=r['selected_sha256'],updates=saved['updates'],mean_cpu_gpu_differences=differences,
            cpu_groups=cpu['groups'],gpu_groups=selected['groups'],posterior_noise_not_paired_across_devices=True,
            cpu_fresh_initial_digest=cpu_initial_digest,initial_parameters_match_gpu_with_saved_buffers=True,
            regenerated_cpu_buffer_max_differences=buffer_differences,
            entire_source_codec_unchanged=tensor_digest(base.state_dict())==initial)
        if not check['entire_source_codec_unchanged'] or file_sha(root/SOURCE)!=SHA:raise AssertionError('codec mutation')
        (arm/'cpu-reload-check.json').write_text(json.dumps(check,indent=2)+'\n');checks[name]=check
        log=[json.loads(s) for s in (arm/'metrics.jsonl').read_text().splitlines()]
        schedules[name]=[(r['sample_ids'],r['lr']) for r in log]
    first=next(iter(results));paired=dict(same_initial_head_tensors=len({r['initial_head_tensor_sha256'] for r in results.values()})==1,
        same_update_sample_and_lr_schedule=all(v==schedules[first] for v in schedules.values()),
        same_codec_and_dataset=True,dropout_caveat='Same seeds and forward counts; different attention kernels need not produce bit-identical dropout masks.')
    if not paired['same_initial_head_tensors'] or not paired['same_update_sample_and_lr_schedule']:raise AssertionError(paired)
    out=directory/'report';out.mkdir(exist_ok=True)
    summary=dict(results=results,configs=configs,cpu_reload=checks,pairing=paired,
        transport_preflight=json.loads((directory/'transport-preflight.json').read_text()),
        pretrained_ablation=json.loads((directory/'pretrained-ablation-summary.json').read_text()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,r in results.items():
        history=[dict(step=0,groups=r['initial'])]+r['history']
        for ax,group in zip(axes,('train','held_out')):
            ax.plot([p['step'] for p in history],[100*p['groups'][group]['mu']['cer'] for p in history],label=name,marker='.')
            ax.set_title(group+' mean CER');ax.set_xlabel('optimizer updates');ax.set_ylabel('CER (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    pages=[]
    for page,ids in enumerate(paginate_ids(prov['splits']['held_out'],limit=8),1):
        fig,axes=plt.subplots(len(ids),1,figsize=(13,2.7*len(ids)),squeeze=False)
        for ax,sid in zip(axes[:,0],ids):
            target=raw[sid][0][0,:int(raw[sid][1].sum())].numpy().copy();target[:,:2]*=.01
            draw(ax,split_xy(target[:,:2],target[:,2:].argmax(1)))
            lines=['truth: '+texts[sid]]
            for name in ARMS:
                row=next(r for r in gpu[name]['lines'] if r['sample_id']==sid)
                lines.append(name+': '+row['mu']['decoded'])
            ax.set_title(sid+'\n'+'\n'.join(lines),fontsize=8,loc='left')
        fig.tight_layout();fname=f'held-out-{page}.png';fig.savefig(out/fname,dpi=130);plt.close(fig);pages.append(fname)
    esc=html.escape
    chunks=['<meta charset="utf-8"><title>Frozen English OCR context controls</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto}img{max-width:100%}td{padding:.4em}pre{white-space:pre-wrap}</style>',
        '<h1>Frozen English OCR: coordinate/context controls</h1>',
        '<p>Four FRESH identically initialized OCR heads, each1000 updates: AdamW5e-4→1e-4 at750, betas.9/.99, weight decay1e-4, clip5, dropout.1, physical OCR batch16, seed42 complete bucketed train epochs. Same source codec, samples, schedule and train-only calibration. Selected by TRAIN mean CER then CTC, never held-out. See exact configs and update logs. Different attention kernels may consume dropout RNG differently.</p>',
        '<p>192 training and32 held-out lines from the same8 writers; overlapping forms. This is a small exploratory diagnostic, NOT an independent IAM writer/form benchmark. No encoder, geometry, pen, KL, style, readout, posterior or original OCR parameter is trained. Source codec, including old OCR, is bitwise unchanged. No new geometry reconstructions are claimed here; prior visual fidelity audit remains valid.</p>',
        '<p>The input contract is specific to this initialized polyphase40 transport: real phases up to first EOC retained; synthetic tail phases and344 unused channels zeroed ONLY at OCR input in EVERY arm. This is not safe for arbitrary learned latents. All224 payload mappings/pen phases checked against raw trajectory before training. Standalone head checkpoints require their config feature contract and cannot be loaded as ordinary VAE checkpoints.</p>',
        '<p>global_raw: absolute XY. global_scaled: train-only XY-axis standardization. relative_scaled: phase offsets relative to block first X plus first-X displacement from previous block; translation invariant, invertible up to X origin. Y absolute, standardized. Index differences are NOT physical velocity. relative_local4 additionally limits each layer to±4 blocks;3 layers imply±12 feature blocks, plus one prior block for relative X. Padded queries attend one safe key; real queries have NO global-key escape.</p>',
        '<p><a href="summary.json">Complete metrics, pairing, hashes and CPU reload audit</a> · <a href="../provenance.json">Line IDs/provenance</a> · <a href="../../../iam_frozen_ocr_study/research-summary/index.html">Previously established marker-free codec fidelity</a></p>',
        '<table border="1"><tr><th>arm/selected step</th><th>train mean CER/exact</th><th>train sampled CER</th><th>held-out mean CER/exact</th><th>held-out sampled CER</th><th>train CTC</th><th>CPU/CUDA mean differences</th></tr>']
    for name,r in results.items():
        t,h=r['selected']['train'],r['selected']['held_out']
        chunks.append(f'<tr><td>{name}/{r["best_step"]}</td><td>{t["mu"]["cer"]:.4%}/{t["mu"]["exact_lines"]}/192</td><td>{t["sampled"]["cer"]:.4%}</td><td>{h["mu"]["cer"]:.4%}/{h["mu"]["exact_lines"]}/32</td><td>{h["sampled"]["cer"]:.4%}</td><td>{t["mean_ctc_loss"]:.6f}</td><td>{len(checks[name]["mean_cpu_gpu_differences"])}</td></tr>')
    chunks.append('</table><img src="learning.png"><h2>Pretrained-head dependence ablations (NO training)</h2><p>These perturb the previously memorized head8738dc9, not these new heads. They are out-of-distribution dependence checks, NOT evidence of a sole causal mechanism. Mean-only; no posterior draws. The synthetic width/length input removes handwriting shape: interpret any survival as shortcut dependence, not OCR ability.</p><table border="1"><tr><th>perturbation</th><th>train CER</th><th>held-out CER</th></tr>')
    for kind,g in summary['pretrained_ablation']['groups'].items():
        chunks.append(f'<tr><td>{kind}</td><td>{g["train"]["mu"]["cer"]:.4%}</td><td>{g["held_out"]["mu"]["cer"]:.4%}</td></tr>')
    chunks.append('</table><h2>Every transcript: GPU selected checkpoint</h2><p>CPU reload metrics/differences are retained separately, not hidden. CPU/CUDA posterior noise streams differ; comparisons are paired only within a device.</p>')
    maps={k:{r['sample_id']:r for r in v['lines']} for k,v in gpu.items()}
    for group in ('train','held_out'):
        chunks.append('<h3>'+group+'</h3><table border="1"><tr><th>ID</th><th>target</th>'+''.join('<th>'+k+'</th>' for k in ARMS)+'</tr>')
        for sid in prov['splits'][group]:
            chunks.append('<tr><td>'+esc(sid)+'</td><td>'+esc(texts[sid])+'</td>'+''.join('<td>'+esc(maps[k][sid]['mu']['decoded'])+'</td>' for k in ARMS)+'</tr>')
        chunks.append('</table>')
    chunks.append('<h2>Held-out handwriting and predictions</h2><p>Marker-free IAM/RDP targets, not purported new generated handwriting. Geometry itself was untouched. Inspect recognition mistakes on the same observed line in each arm.</p>')
    for fname in pages:chunks.append('<img loading="lazy" src="'+fname+'">')
    (out/'index.html').write_text('\n'.join(chunks));annotate(directory);publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),arms={n:dict(train_cer=r['selected']['train']['mu']['cer'],held_out_cer=r['selected']['held_out']['mu']['cer']) for n,r in results.items()},pairing=paired)


def text_coverage(texts,splits):
    """Count train vocabulary coverage; no held-out-derived training statistics."""
    from collections import Counter
    train=Counter(''.join(texts[i] for i in splits['train']))
    held=Counter(''.join(texts[i] for i in splits['held_out']))
    known={texts[i] for i in splits['train']}
    return dict(train_characters=dict(train),held_out_characters=dict(held),
        held_out_unseen_characters={c:n for c,n in held.items() if c not in train},
        held_out_total_characters=sum(held.values()),
        held_out_exact_transcripts_in_train=sum(texts[i] in known for i in splits['held_out']))


def annotate(directory):
    """Cheap report-only interpretation pass; no model load or optimization."""
    directory=Path(directory);out=directory/'report';summary=json.loads((out/'summary.json').read_text())
    prov=json.loads((directory/'provenance.json').read_text())
    coverage=text_coverage({k:r['text'] for k,r in prov['samples'].items()},prov['splits'])
    (out/'text-coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    r=summary['results'];raw=r['global_raw']['selected']['held_out']['mu']['cer'];rel=r['relative_scaled']['selected']['held_out']['mu']['cer']
    memorized=sum(v['selected']['train']['mu']['cer']==0 and v['selected']['train']['sampled']['cer']==0 for v in r.values())
    exact_held={k:v['selected']['held_out']['mu']['exact_lines'] for k,v in r.items()}
    text=f'''<section id="interpretation"><h2>What was established / next gate</h2>
<p>Translation-invariant relative features reduce selected held-out CER from {raw:.4%} to {rel:.4%} ({100*(raw-rel):.2f} percentage points) in this matched small-pool diagnostic. Attention restriction did NOT improve further. These are exploratory single-seed comparisons; coordinate transformation changes conditioning as well as absolute-X dependence, so this is NOT a proof of a single causal mechanism.</p>
<p>{memorized} arms have zero train mean/posterior CER. Exact held-out line counts: {html.escape(str(exact_held))}. Shape-free width/length perturbation loses most training recognition; whole-line attention removal strongly hurts the pretrained memorizer. Thus dependence on global context exists, but pure width-only memorization is NOT sufficient, and OOD perturbations do not establish the sole cause.</p>
<p>Only {sum(coverage['held_out_unseen_characters'].values())}/{coverage['held_out_total_characters']} held-out target characters are unseen in training ({html.escape(str(coverage['held_out_unseen_characters']))}); there are {coverage['held_out_exact_transcripts_in_train']} identical train/held-out transcripts. Missing labels cannot explain the roughly72–83% error. Training has only{sum(coverage['train_characters'].values())} character occurrences, so a larger TRAIN-only supervision pool is the next test, not more memorization updates or releasing the geometry guard. Require form/writer/split provenance, train-only calibration, and independent selection before claiming generalization.</p>
<p>The CPU initializer's sinusoidal positional buffer differs across builds/hosts by small roundoff (recorded per arm). All random parameter tensors match the as-run initialization when the saved buffer is restored. CPU reload uses the ACTUAL saved positional buffer and head weights; it does not silently regenerate them. Any CPU/CUDA transcript differences are retained in the reload audit. Codec9c53 remains the selected geometry checkpoint; none of these research heads is promoted to joint VAE/DiT training.</p>
<p><a href="text-coverage.json">Train/held-out text coverage</a>.</p></section>'''
    page=out/'index.html';old=page.read_text();old=old.split('<section id="interpretation">')[0];page.write_text(old+'\n'+text)
    return dict(output=str(page),text_coverage=coverage)
