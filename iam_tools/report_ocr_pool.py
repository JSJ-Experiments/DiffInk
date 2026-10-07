"""CPU reload and marker-free provenance report for larger frozen-codec OCR."""
import html,json
from pathlib import Path
import h5py,numpy as np,torch
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_context_features import make_head
from .ocr_context_study import tensor_digest
from .writer_expansion import load
from .pen_ab import file_sha
from .report_pointer import publish_pointer


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    results=json.loads((directory/'result.json').read_text());configs={k:json.loads((directory/k/'config.json').read_text()) for k in results}
    if set(results)!={'small192','large2048'}:raise AssertionError('both complete arms required')
    first=configs['small192'];pool,m,vocab=load_pool(root,first['pool_manifest_sha256'])
    if (directory/'pool-manifest.json').read_bytes()!=(pool/'manifest.json').read_bytes():raise AssertionError('dataset drift')
    base,_,_,cfg,original_vocab,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    if vocab!=original_vocab:raise AssertionError('vocab drift')
    base.eval().requires_grad_(False);digest=tensor_digest(base.state_dict());cache,texts,codec_audit=cache_corpus(base,pool,m,vocab,directory,device='cpu',save_geometry=False)
    checks={};gpu={};cpu={};initials=[]
    for name,r in results.items():
        folder=directory/name;c=configs[name];saved=torch.load(folder/'head-best.pt',weights_only=True,map_location='cpu')
        if file_sha(folder/'head-best.pt')!=r['selected_sha256'] or saved['config']!=c or saved['updates']!=r['best_step']:raise AssertionError('selected checkpoint/contract integrity failure')
        if c['feature_stats']!=first['feature_stats'] or c['feature_calibration_ids']!=first['feature_calibration_ids']:raise AssertionError('unpaired feature calibration')
        head=make_head(cfg,len(vocab)+1,c['feature_mode'],c['feature_stats'],c['attention_radius'],seed=42)
        buffers={}
        with torch.no_grad():
            for key,buf in head.named_buffers():
                value=saved['ocr_state_dict'][key];buffers[key]=float((buf-value).abs().max());buf.copy_(value)
        if tensor_digest(head.state_dict())!=r['initial_head_tensor_sha256']:raise AssertionError('fresh random weights mismatch after restoring saved nonrandom buffers')
        initials.append(r['initial_head_tensor_sha256']);head.load_state_dict(saved['ocr_state_dict']);head.eval()
        splits=dict(train=c['train_ids'],dev=c['dev_ids'],held_out=c['held_out_ids'],common_train_probe=c['common_train_probe'])
        row=evaluate(head,cache,texts,splits,vocab,folder,'cpu-reload',c['posterior_evaluation_ids']);cpu[name]=row
        selected=json.loads((folder/f'ocr-{r["best_step"]}.json').read_text());gpu[name]=selected
        by_id={x['sample_id']:x for x in selected['lines']};differences=[]
        for x in row['lines']:
            before=by_id[x['sample_id']]
            if x['mu']['decoded']!=before['mu']['decoded']:
                z=cache[x['sample_id']]
                with torch.no_grad():
                    logits=head(z['mu'],padding_mask=~z['mask'])[:int(z['mask'].sum()),0];top=logits.topk(2,dim=-1).values
                differences.append(dict(sample_id=x['sample_id'],cpu=x['mu']['decoded'],gpu=before['mu']['decoded'],min_cpu_top_two_logit_gap=float((top[:,0]-top[:,1]).min())))
        check=dict(selected_sha256=r['selected_sha256'],random_parameter_initialization_matches_as_run=True,regenerated_positional_buffer_differences=buffers,
            mean_cpu_gpu_transcript_differences=differences,cpu_groups=row['groups'],gpu_groups=selected['groups'],posterior_device_rng_not_paired=True,
            entire_codec_state_source_identical=tensor_digest(base.state_dict())==digest)
        if not check['entire_codec_state_source_identical'] or file_sha(root/SOURCE)!=SHA:raise AssertionError('codec mutation on CPU')
        (folder/'cpu-reload-check.json').write_text(json.dumps(check,indent=2)+'\n');checks[name]=check
    if len(set(initials))!=1:raise AssertionError('unpaired arms')
    logs={k:[json.loads(s) for s in (directory/k/'metrics.jsonl').read_text().splitlines()] for k in results}
    paired=dict(same_initial_head_tensors=True,same_feature_calibration=True,same_writer_population=True,
        same_optimizer_update_count=len(logs['small192'])==len(logs['large2048']),same_lr_schedule=[r['lr'] for r in logs['small192']]==[r['lr'] for r in logs['large2048']],
        train_data_nested=True,samples_and_dropout_noise_not_paired_due_different_pool=True,checkpoint_selection_dev_only=True)
    if not paired['same_optimizer_update_count'] or not paired['same_lr_schedule']:raise AssertionError('unequal bounded budgets; interpret explicitly before reporting')
    out=directory/'report';out.mkdir(exist_ok=True);audit=json.loads((directory/'codec-preflight.json').read_text())
    summary=dict(results=results,configs=configs,paired=paired,pool_sha256=first['pool_manifest_sha256'],split_checks=m['checks'],
        codec_gpu_preflight=audit,codec_cpu_preflight=codec_audit,cpu_reload=checks)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,r in results.items():
        for ax,group in zip(axes,('train','dev','held_out')):
            ax.plot([p['step'] for p in r['history']],[100*p['groups'][group]['mu']['cer'] for p in r['history']],marker='.',label=name)
            ax.axvline(r['best_step'],alpha=.2);ax.set_title(group+' mean CER');ax.set_xlabel('optimizer updates');ax.set_ylabel('%');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    pages=[];maps={k:{r['sample_id']:r for r in v['lines']} for k,v in gpu.items()}
    with h5py.File(pool/'lines.h5') as targets,h5py.File(directory/'geometry-source.h5') as recon:
        for group in ('held_out','dev'):
            for page,ids in enumerate(paginate_ids(m['splits'][group],limit=8),1):
                fig,axes=plt.subplots(len(ids),2,figsize=(15,2.3*len(ids)),squeeze=False)
                for j,sid in enumerate(ids):
                    target=targets[sid]['point_seq'][:];target[:,:2]*=.01;fixed=recon[sid]['mean'][:]
                    lo=target[:,:2].min(0);hi=target[:,:2].max(0)
                    small=maps['small192'][sid]['mu']['decoded'];large=maps['large2048'][sid]['mu']['decoded']
                    for ax,points,title in [(axes[j,0],target,'IAM/RDP target: '+texts[sid]),(axes[j,1],fixed,'Shared FROZEN reconstruction\nsmall OCR: '+small+'\nlarge OCR: '+large)]:
                        draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=8)
                fig.tight_layout();fname=f'{group}-{page}.png';fig.savefig(out/fname,dpi=130);plt.close(fig);pages.append((group,page,fname))
    esc=html.escape
    chunks=['<meta charset="utf-8"><title>Frozen English OCR data-size controls</title><style>body{font-family:sans-serif;max-width:1450px;margin:auto}img{max-width:100%}td{padding:.4em}pre{white-space:pre-wrap}</style>',
        '<h1>English OCR: 192 versus2048 prompt-guarded training lines</h1>',
        '<p>Only fresh standalone OCR heads train. All original codec parameters/buffers, including its old OCR, remain bitwise unchanged. The same observed handwriting is reconstructed in BOTH arms. No geometry, pen, posterior, KL/style/readout optimization, no InkDiT or text generation.</p>',
        '<p>Nested192/2048 TRAIN pools cover the same186 writers.128 DEV lines from five previously reserved writers select checkpoints by CER then CTC. Original32 seen-writer lines are REPORT-only. TRAIN excludes all DEV/report prompt families (including IAM writer-version variants) and normalized exact transcripts.25 test writers are excluded entirely. The fixed81-character alphabet is inherited from the broad original training inventory; no DEV labels expand it.</p>',
        '<p>Both arms: identical fresh seed42 head weights, blankbias0/dropout.1, shared relative-X features and EXACT same moments fitted ONLY on small192 TRAIN. Batch16 cached means; raw encoder batch1 minimal padding. AdamW5e-4→1e-4 at75% of equal update budgets, betas.9/.99, decay1e-4, clip5. Sample/dropout realizations are not paired across different datasets. The changed training-pool size—not a changed feature contract or initialization—is the intervention.</p>',
        '<p>Important caveat: the frozen codec previously learned reconstruction on original192, whose forms overlap original32. This study isolates OCR supervision, NOT a fully independent pretrained-representation IAM benchmark, paper reproduction or semantic/generative latent proof. This stricter192 control has a different writer/form/text distribution from the earlier8-writer study; compare the TWO matched arms here, not raw earlier percentages.</p>',
        '<p>Polyphase40 adapter is research-only: valid phases through first final EOC, synthetic phases/344 unused channels removed ONLY at OCR input. Packed payloads and actual reconstructed mean geometry/pens were checked on EVERY pool line before any OCR update. Posterior OCR uses20 fixed GPU draws on128 DEV,32 report and32 common TRAIN probes; train-all2048 mean CER is not misrepresented as2048 posterior evaluations.</p>',
        '<p><a href="summary.json">Full metrics/hashes/reload/geometry audit</a> · <a href="../pool-manifest.json">Every ID, source fingerprint and split check</a> · <a href="../../../iam_frozen_ocr_study/research-summary/index.html">Prior named c/h fidelity audit</a></p>',
        '<table border="1"><tr><th>arm/DEV-selected step</th><th>train mean CER/exact</th><th>DEV mean/posterior CER</th><th>report32 mean/posterior CER/exact</th><th>common TRAIN32 mean/posterior CER</th><th>CPU/GPU mean differences</th></tr>']
    for name,r in results.items():
        t,d,h,p=(r['selected'][k] for k in ('train','dev','held_out','common_train_probe'))
        chunks.append(f'<tr><td>{name}/{r["best_step"]}</td><td>{t["mu"]["cer"]:.4%}/{t["mu"]["exact_lines"]}/{t["mu"]["evaluations"]}</td><td>{d["mu"]["cer"]:.4%}/{d["sampled"]["cer"]:.4%}</td><td>{h["mu"]["cer"]:.4%}/{h["sampled"]["cer"]:.4%}/{h["mu"]["exact_lines"]}/32</td><td>{p["mu"]["cer"]:.4%}/{p["sampled"]["cer"]:.4%}</td><td>{len(checks[name]["mean_cpu_gpu_transcript_differences"])}</td></tr>')
    chunks.append('</table><img src="learning.png"><h2>Frozen codec preflight</h2>')
    geometry_rows=audit['lines']
    brief=dict(lines=len(geometry_rows),passed=audit['passed'],maximum_packed_xy_difference=audit['maximum_packed_xy_difference'],
        mean_per_line_x_rmse=float(np.mean([r['geometry']['x_rmse'] for r in geometry_rows])),mean_per_line_y_rmse=float(np.mean([r['geometry']['y_rmse'] for r in geometry_rows])),
        mean_per_line_turn_p90_degrees=float(np.mean([r['geometry']['turn_angle_error_degrees']['p90'] for r in geometry_rows if r['geometry']['turn_angle_error_degrees']['p90'] is not None])),
        all_pen_boundaries_and_final_eocs_perfect=all(r['pen']['pen_up_f1']==1 and r['pen']['final_eoc_correct'] and not r['pen']['non_final_false_eoc_count'] for r in geometry_rows))
    chunks.append('<pre>'+esc(json.dumps(brief,indent=2))+'</pre><p>Target authentic RDP corners/microstructure retained, no generic smoothing. Point differences are by uneven sample index; angles are geometric. All tensors stay frozen after each arm; one shared source reconstruction saved before fitting. CPU/CUDA posterior RNGs differ, and argmax near-ties can differ; any transcript discrepancies are retained with margins.</p>')
    for group in ('held_out','dev'):
        chunks.append('<h2>'+group+' selected mean transcripts</h2><table border="1"><tr><th>ID</th><th>target</th><th>small192</th><th>large2048</th></tr>')
        for sid in m['splits'][group]:chunks.append('<tr><td>'+esc(sid)+'</td><td>'+esc(texts[sid])+'</td><td>'+esc(maps['small192'][sid]['mu']['decoded'])+'</td><td>'+esc(maps['large2048'][sid]['mu']['decoded'])+'</td></tr>')
        chunks.append('</table>')
    for group,page,fname in pages:chunks.append(f'<h2>{group} page{page}: target / shared frozen reconstruction</h2><img loading="lazy" src="{fname}">')
    (out/'index.html').write_text('\n'.join(chunks));annotate(directory);publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),paired=paired,codec=brief,arms={k:dict(best_step=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],held_out_cer=r['selected']['held_out']['mu']['cer']) for k,r in results.items()})


def annotate(directory):
    """Cheap interpretation update without reallocating GPU or reloading models."""
    out=Path(directory)/'report';summary=json.loads((out/'summary.json').read_text());r=summary['results']
    a,b=(r[k]['selected'] for k in ('small192','large2048'))
    text=f'''<section id="interpretation"><h2>What to look for / conclusion</h2>
<p>The two handwriting columns should look identical: left is the observed IAM/RDP target, right is its protected reconstruction, shared by BOTH OCR arms. The intervention changes recognition, not writing. Compare the <b>large OCR</b> caption with the target transcript; it is substantially more readable than <b>small OCR</b>, but still makes letter, case, spacing and deletion errors.</p>
<p>Matched-data comparison: original32 mean character error {a['held_out']['mu']['cer']:.4%} → {b['held_out']['mu']['cer']:.4%}; independent unseen-writer DEV {a['dev']['mu']['cer']:.4%} → {b['dev']['mu']['cer']:.4%}. Checkpoints selected on DEV, never original32. The much larger supervision pool helps strongly while the geometry stays unchanged. Near-perfect TRAIN reading alone was misleading.</p>
<p>This establishes data starvation/memorization as a major contributor under this feature contract, not a sole cause or proof more data will solve everything. About a quarter of unseen characters remain wrong. No head is promoted into unconstrained joint VAE/OCR/KL/style training; do not deform faithful handwriting to satisfy a still-unreliable reader. Next: more TRAIN-only supervision with preserved calibration/split/geometry guards. No novel handwriting generation or InkDiT readiness claim.</p></section>'''
    page=out/'index.html';old=page.read_text().split('<section id="interpretation">')[0];page.write_text(old+'\n'+text)
    return dict(output=str(page),dev_cer=b['dev']['mu']['cer'],held_out_cer=b['held_out']['mu']['cer'])
