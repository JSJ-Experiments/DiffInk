"""Matched continuation report: recognition captions change, geometry does not."""
import html,json
from pathlib import Path
import h5py,numpy as np,torch
from .ocr_pool_expansion import PARENT_HEAD,PARENT_SHA,PARENT_POOL,validate_parent
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_context_features import make_head
from .ocr_context_study import tensor_digest
from .writer_expansion import load
from .pen_ab import file_sha
from .report_pointer import publish_pointer



def baseline_equivalence(a,b,loss_tolerance=1e-5):
    """Same restored head: exact decodes/draws, bounded FP32 batch roundoff.

    Mean evaluation batches differ when sorted over2048 versus8192 examples;
    comparing complete float-valued group dicts bitwise is not meaningful.
    This is NOT tolerance for a changed transcript or posterior realization.
    """
    left={r['sample_id']:r for r in a['lines']};right={r['sample_id']:r for r in b['lines']}
    if not set(left)<=set(right):raise AssertionError('parent baseline IDs missing')
    maximum=0.
    for sid,r in left.items():
        other=right[sid]
        if r['mu']!=other['mu'] or r['sampled']!=other['sampled']:raise AssertionError('baseline transcript/posterior changed: '+sid)
        difference=abs(r['ctc_loss']-other['ctc_loss']);maximum=max(maximum,difference)
        if not np.isfinite(difference) or difference>loss_tolerance:raise AssertionError('baseline CTC differs beyond FP32 tolerance: '+sid)
    return dict(compared_parent_lines=len(left),all_mean_and_sampled_decode_records_identical=True,
                maximum_per_line_ctc_difference=maximum,absolute_ctc_tolerance=loss_tolerance,
                note='different masked evaluation batch shapes; floating CTC is not expected bitwise identical')


def report(directory,repo,root='/data',study='expansion'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    results=json.loads((directory/'result.json').read_text());configs={k:json.loads((directory/k/'config.json').read_text()) for k in results}
    if study not in ('expansion','convergence'):raise ValueError('known paired study required')
    convergence=study=='convergence'
    names=('lr1e-4','lr2e-4') if convergence else ('control2048','expanded8192')
    parent_name='parent12000' if convergence else 'parent6000';start_step=12000 if convergence else 6000
    if convergence:
        from .ocr_convergence import PARENT_HEAD as head_rel,PARENT_SHA as head_sha,POOL_SHA as parent_pool_sha,validate_parent as validate_convergence
    else:head_rel,head_sha,parent_pool_sha=PARENT_HEAD,PARENT_SHA,PARENT_POOL
    if set(results)!=set(names):raise AssertionError('both complete arms required')
    first=configs[names[0]];pool,m,vocab=load_pool(root,first['pool_manifest_sha256']);_,parent,_=load_pool(root,parent_pool_sha)
    if file_sha(root/head_rel)!=head_sha or (directory/'pool-manifest.json').read_bytes()!=(pool/'manifest.json').read_bytes():raise AssertionError('parent/dataset drift')
    saved_parent=torch.load(root/head_rel,weights_only=True,map_location='cpu')
    previous=validate_convergence(saved_parent,m) if convergence else validate_parent(saved_parent,parent,m)
    base,_,_,cfg,alphabet,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    if alphabet!=vocab:raise AssertionError('vocab drift')
    base.eval().requires_grad_(False);digest=tensor_digest(base.state_dict())
    # Independently re-encode/decode the entire evaluation gallery+common probe,
    # NOT all8192 CPU lines. GPU preflight already gates every8352 pool line.
    probe=previous['common_train_probe'];ids=probe+previous['dev_ids']+previous['held_out_ids']
    subset=dict(m,records={i:m['records'][i] for i in ids})
    cache,texts,codec_cpu=cache_corpus(base,pool,subset,vocab,directory,device='cpu',save_geometry=False)
    splits=dict(train=probe,dev=previous['dev_ids'],held_out=previous['held_out_ids'],common_train_probe=probe)
    checks={};maps={};gpu={};cpu={};beam={}
    for name in (parent_name,*names):
        folder=directory/name;folder.mkdir(exist_ok=True)
        saved=saved_parent if name==parent_name else torch.load(folder/'head-best.pt',weights_only=True,map_location='cpu')
        c=previous if name==parent_name else configs[name]
        if name!=parent_name:
            r=results[name]
            if file_sha(folder/'head-best.pt')!=r['selected_sha256'] or saved['config']!=c or saved['updates']!=r['best_step']:raise AssertionError('selected checkpoint drift')
            if c['initial_state']!=first['initial_state'] or c['feature_stats']!=previous['feature_stats'] or c['feature_calibration_ids']!=previous['feature_calibration_ids']:raise AssertionError('unpaired continuation')
        head=make_head(cfg,len(vocab)+1,c['feature_mode'],c['feature_stats'],c['attention_radius']);head.load_state_dict(saved['ocr_state_dict']);head.eval()
        row=evaluate(head,cache,texts,splits,vocab,folder,'cpu-reload',previous['posterior_evaluation_ids']);cpu[name]=row
        if name!=parent_name:
            original=json.loads((folder/f'ocr-{results[name]["best_step"]}.json').read_text());gpu[name]=original
            original_map={r['sample_id']:r for r in original['lines']}
            differences=[dict(sample_id=r['sample_id'],cpu=r['mu']['decoded'],gpu=original_map[r['sample_id']]['mu']['decoded']) for r in row['lines'] if r['mu']['decoded']!=original_map[r['sample_id']]['mu']['decoded']]
            # Full train metrics remain AS-RUN GPU only. CPU train=common32,
            # explicitly labelled here; never claim an8192 CPU reload.
            checks[name]=dict(mean_cpu_gpu_transcript_differences=differences,cpu_scope='128DEV+32report+32commonTRAIN; train group here is ONLY32probe',
                cpu_groups=row['groups'],posterior_device_rng_not_paired=True,selected_sha256=results[name]['selected_sha256'])
        maps[name]={r['sample_id']:r for r in row['lines']}
        if convergence:
            from .ctc_beam import audit_head
            beam[name]=audit_head(head,cache,texts,dict(dev=previous['dev_ids'],held_out=previous['held_out_ids']),vocab,width=10)
            (folder/'beam-mean-only.json').write_text(json.dumps(beam[name],indent=2)+'\n')
    if tensor_digest(base.state_dict())!=digest or file_sha(root/SOURCE)!=SHA:raise AssertionError('CPU codec changed')
    logs={k:[json.loads(s) for s in (directory/k/'metrics.jsonl').read_text().splitlines()] for k in results}
    paired=dict(same_parent_head_moments_counters_rng=True,same_feature_calibration=True,same_train_writer_population=True,
        same_additional_update_count=len(logs[names[0]])==len(logs[names[1]]),
        same_lr_schedule=[r['lr'] for r in logs[names[0]]]==[r['lr'] for r in logs[names[1]]],selection_dev_only=True)
    if convergence:
        paired.update(same_fixed8192_training_ids=configs[names[0]]['train_ids']==configs[names[1]]['train_ids']==previous['train_ids'],
            same_per_update_sample_ids=[r['sample_ids'] for r in logs[names[0]]]==[r['sample_ids'] for r in logs[names[1]]],
            same_schedule_digest=results[names[0]]['sample_schedule_sha256']==results[names[1]]['sample_schedule_sha256'],
            same_final_rng=results[names[0]]['final_rng_sha256']==results[names[1]]['final_rng_sha256'],
            same_rng_at_every_evaluation=[r.get('rng_sha256') for r in results[names[0]]['history']]==[r.get('rng_sha256') for r in results[names[1]]['history']],
            optimizer_except_lr_identical=configs[names[0]]['lr_intervention_proof']['optimizer_except_lr_sha256']==configs[names[1]]['lr_intervention_proof']['optimizer_except_lr_sha256'],
            samples_and_dropout_paired=True,parent_iterator_resumed_after6000=True,
            intended_lr_only_intervention=all(r['lr']==1e-4 for r in logs[names[0]]) and all(r['lr']==2e-4 for r in logs[names[1]]))
        if not all(v for k,v in paired.items() if k!='same_lr_schedule'):raise AssertionError('convergence pairing/intervention failed')
    else:
        paired.update(source2048_nested_in8192=True,schedules_restart_seed43=True,samples_and_dropout_not_paired=True)
        if not paired['same_additional_update_count'] or not paired['same_lr_schedule']:raise AssertionError('budgets unequal; interpretation required')
    before_a=json.loads((directory/names[0]/f'ocr-{start_step}.json').read_text());before_b=json.loads((directory/names[1]/f'ocr-{start_step}.json').read_text())
    if {r['sample_id'] for r in before_a['lines']}!=set(parent['records']) or {r['sample_id'] for r in before_b['lines']}!=set(m['records']):raise AssertionError('baseline pool evaluation incomplete')
    paired['baseline_evaluation']=baseline_equivalence(before_a,before_b)
    if convergence:paired['baseline_evaluation']['note']='same masked batches/head/features; before-run CTC and decode records expected identical'
    audit=json.loads((directory/'codec-preflight.json').read_text());grows=audit['lines']
    brief=dict(lines=len(grows),passed=audit['passed'],maximum_packed_xy_difference=audit['maximum_packed_xy_difference'],
        mean_per_line_x_rmse=float(np.mean([r['geometry']['x_rmse'] for r in grows])),mean_per_line_y_rmse=float(np.mean([r['geometry']['y_rmse'] for r in grows])),
        mean_per_line_turn_p90_degrees=float(np.mean([r['geometry']['turn_angle_error_degrees']['p90'] for r in grows if r['geometry']['turn_angle_error_degrees']['p90'] is not None])),
        all_pen_boundaries_final_eocs_perfect=all(r['pen']['pen_up_f1']==1 and r['pen']['final_eoc_correct'] and not r['pen']['non_final_false_eoc_count'] for r in grows))
    out=directory/'report';out.mkdir(exist_ok=True)
    summary=dict(results=results,configs=configs,paired=paired,before_groups=before_a['groups'],parent_head_sha256=head_sha,
        pool_sha256=first['pool_manifest_sha256'],extension_checks=m['extension_checks'],split_checks=m['checks'],codec=brief,
        metadata_note='Convergence config initial head digest identifies actual restored parent tensors; optimizer/RNG pairing and LR-only mutation proved.' if convergence else 'Historical expansion inherited legacy initial_head_tensor_sha256 from fresh parent; initial_state.head_tensor_sha256 is authoritative. As-run files are preserved, not edited.',
        codec_gpu_preflight=audit,codec_cpu_eval192=codec_cpu,cpu_reload=checks,cpu_parent_groups=cpu[parent_name]['groups'])
    if convergence:
        summary['ctc_prefix_beam_mean_only']=beam
        summary['final_update_groups']={name:r['history'][-1]['groups'] for name,r in results.items()}
    transitions={}
    for group in ('dev','held_out'):
        transitions[group]={}
        for left,right in [(parent_name,names[0]),(parent_name,names[1]),names]:
            rows=[dict(sample_id=i,before_errors=maps[left][i]['mu']['errors'],after_errors=maps[right][i]['mu']['errors']) for i in m['splits'][group]]
            transitions[group][left+'→'+right]=dict(improved=sum(r['after_errors']<r['before_errors'] for r in rows),
                same=sum(r['after_errors']==r['before_errors'] for r in rows),worsened=sum(r['after_errors']>r['before_errors'] for r in rows),lines=rows)
    summary['cpu_paired_line_error_transitions']=transitions
    # A predeclared index-compression slack diagnostic, not physical duration.
    slack={}
    for group in ('dev','held_out'):
        slack[group]={}
        for name in maps:
            bins={k:[] for k in ('margin≤0.25','0.25<margin≤0.5','margin>0.5')}
            for sid in m['splits'][group]:
                text=texts[sid];required=len(text)+sum(a==b for a,b in zip(text,text[1:]));frames=(m['records'][sid]['points']+7)//8
                margin=(frames-required)/len(text);key='margin≤0.25' if margin<=.25 else '0.25<margin≤0.5' if margin<=.5 else 'margin>0.5'
                bins[key].append(maps[name][sid]['mu'])
            slack[group][name]={k:dict(lines=len(v),errors=sum(x['errors'] for x in v),characters=sum(x['characters'] for x in v),
                cer=sum(x['errors'] for x in v)/sum(x['characters'] for x in v) if v else None) for k,v in bins.items()}
    summary['cpu_ctc_slack_diagnostic']=dict(definition='(ceil(real_points/8) - [characters+adjacent_repeats])/characters; nonuniform RDP index frames, NOT physical time',groups=slack)

    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,r in results.items():
        for ax,group in zip(axes,('train','dev','held_out')):
            ax.plot([p['step']-start_step for p in r['history']],[100*p['groups'][group]['mu']['cer'] for p in r['history']],marker='.',label=name)
            ax.axvline(r['best_step']-start_step,alpha=.2);ax.set_title(group+' mean CER');ax.set_xlabel('additional optimizer updates');ax.set_ylabel('%');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    pages=[]
    with h5py.File(pool/'lines.h5') as targets,h5py.File(directory/'geometry-source.h5') as recon:
        for group in ('held_out','dev'):
            for page,page_ids in enumerate(paginate_ids(m['splits'][group],limit=8),1):
                fig,axes=plt.subplots(len(page_ids),2,figsize=(15,2.55*len(page_ids)),squeeze=False)
                for j,sid in enumerate(page_ids):
                    target=targets[sid]['point_seq'][:];target[:,:2]*=.01;fixed=recon[sid]['mean'][:];lo=target[:,:2].min(0);hi=target[:,:2].max(0)
                    caption='Shared FROZEN reconstruction\nbefore: '+maps[parent_name][sid]['mu']['decoded']+'\n'+names[0]+': '+maps[names[0]][sid]['mu']['decoded']+'\n'+names[1]+': '+maps[names[1]][sid]['mu']['decoded']
                    for ax,points,title in ((axes[j,0],target,'IAM/RDP target: '+texts[sid]),(axes[j,1],fixed,caption)):
                        draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=8)
                fig.tight_layout();fname=f'{group}-{page}.png';fig.savefig(out/fname,dpi=130);plt.close(fig);pages.append((group,page,fname))
    esc=html.escape
    chunks=['<meta charset="utf-8"><title>Frozen English OCR pool expansion</title><style>body{font-family:sans-serif;max-width:1450px;margin:auto}img{max-width:100%}td{padding:.4em}pre{white-space:pre-wrap}</style>',
        '<h1>OCR continuation: same2048 vs expanded8192 TRAIN lines</h1>',
        '<h2>What to look for</h2><p>Handwriting is intentionally identical in every arm: left is the observed IAM/RDP target; right is the single shared protected reconstruction. Compare <b>before / control / expanded OCR text captions</b> against the target transcript. This tests reading, not new handwriting generation. It does not add smoothing or train geometry.</p>',
        '<p>Both branches start from selected2048 step6000: exact head parameters/buffers, Adam moments/counters, inherited LR1e-4 and CPU/CUDA RNG. Both restart bucket ordering seed43 (parent data iterator NOT resumed), identical additional update budgets. Different samples/lengths mean dropout realizations diverge. Fixed relative-X calibration from original nested192 is NOT recomputed/refit for expansion; exact moments independently checked.</p>',
        '<p>TRAIN2048 is exact prefix of8192, same186 writers/fixed81 chars, unchanged filters/RDP/scaling. Every parent record fingerprint, calibration192 ID/order, DEV128 and report32 ID/order are retained. DEV writers/forms/normalized transcripts are excluded from TRAIN;25 test writers excluded. DEV alone selects checkpoints, original32 only reports. Codec was previously trained on original192 with prompt overlap to original32; not a fully independent IAM benchmark or paper replication.</p>',
        '<p>Only standalone OCR heads train on cached mu, masked batch16. Whole codec including old OCR frozen, raw encoder batch1 minimal padding. Global attention, blankbias0, dropout.1; AdamW1e-4 betas.9/.99 decay1e-4 clip5.20 paired GPU posterior draws on DEV128/report32/common TRAIN32, not8192 draws. CPU independently reloads those192 evaluation/probe lines; full train metrics and all8352 codec gates are as-run GPU.</p>',
        '<p><a href="summary.json">Metrics/provenance/reload checks</a> · <a href="../pool-manifest.json">Every dataset record/split</a> · <a href="../../../iam_ocr_pool_study/20261007-033445/report/index.html">Previous192 vs2048 control</a></p>',
        '<table border="1"><tr><th>arm/selected total step</th><th>TRAIN mean CER / exact</th><th>DEV mean / posterior CER</th><th>report32 mean / posterior CER / exact</th><th>CPU/GPU mean differences</th></tr>']
    if convergence:
        chunks[0]=chunks[0].replace('Frozen English OCR pool expansion','Frozen English OCR LR convergence')
        chunks[1]='<h1>Fixed8192 OCR convergence: LR1e-4 versus2e-4</h1>'
        chunks[2]=chunks[2].replace('before / control / expanded','before / lr1e-4 / lr2e-4')
        chunks[3]='<p>Both branches resume the EXACT selected8192 step12000 head, Adam moments/counters, CPU/CUDA RNG, feature calibration and data iterator. Both regenerate seed43 bucket iterator and skip its6000 consumed parent batches; no epoch restart. Identical8192 IDs and physical batch16 give the same batch shapes and dropout stream; intermediate/final RNG hashes and every batch ID are checked. The ONLY intervention is LR1e-4 versus2e-4, each for the same8000 additional updates. No changed feature/geometry/label/readout policy.</p>'
        chunks[4]=chunks[4].replace('TRAIN2048 is exact prefix of8192,','TRAIN8192 is fixed,')
        chunks[5]=chunks[5].replace('AdamW1e-4','AdamW1e-4 or2e-4')
        chunks[6]=chunks[6].replace('../../../iam_ocr_pool_study/20261007-033445/report/index.html','../../../iam_ocr_pool_expansion/20261007-041949/report/index.html').replace('Previous192 vs2048 control','Previous2048 vs8192 control')
    b=before_a['groups'];chunks.append(f'<tr><td>{parent_name}</td><td>{b["train"]["mu"]["cer"]:.4%}</td><td>{b["dev"]["mu"]["cer"]:.4%} / {b["dev"]["sampled"]["cer"]:.4%}</td><td>{b["held_out"]["mu"]["cer"]:.4%} / {b["held_out"]["sampled"]["cer"]:.4%}</td><td>CPU baseline reloaded</td></tr>')
    for name,r in results.items():
        t,d,h=(r['selected'][k] for k in ('train','dev','held_out'))
        chunks.append(f'<tr><td>{name}/{r["best_step"]}</td><td>{t["mu"]["cer"]:.4%} / {t["mu"]["exact_lines"]}/{t["mu"]["evaluations"]}</td><td>{d["mu"]["cer"]:.4%} / {d["sampled"]["cer"]:.4%}</td><td>{h["mu"]["cer"]:.4%} / {h["sampled"]["cer"]:.4%} / {h["mu"]["exact_lines"]}/32</td><td>{len(checks[name]["mean_cpu_gpu_transcript_differences"])}</td></tr>')
    a,e=(results[n]['selected'] for n in names)
    conclusion=f'<h2>Interpretation</h2><p>Same-parent, equal-budget control: DEV mean CER {a["dev"]["mu"]["cer"]:.4%} vs expanded {e["dev"]["mu"]["cer"]:.4%}; report32 {a["held_out"]["mu"]["cer"]:.4%} vs {e["held_out"]["mu"]["cer"]:.4%}. Parent baseline: DEV {b["dev"]["mu"]["cer"]:.4%}, report32 {b["held_out"]["mu"]["cer"]:.4%}. Compare against the continuation control, not just the old checkpoint, to isolate data expansion from additional optimizer updates. Remaining recognition errors are preserved in every transcript/gallery below. Five DEV writers and repeated evaluation limit benchmark claims. No OCR head is promoted into unconstrained joint geometry training on these numbers alone.</p>'
    if convergence:
        conclusion=f'<h2>Interpretation</h2><p>Same-parent, same-data, paired-batch/dropout continuation: DEV CER {a["dev"]["mu"]["cer"]:.4%} at LR1e-4 vs {e["dev"]["mu"]["cer"]:.4%} at LR2e-4; report32 {a["held_out"]["mu"]["cer"]:.4%} vs {e["held_out"]["mu"]["cer"]:.4%}. Parent baseline DEV {b["dev"]["mu"]["cer"]:.4%}, report32 {b["held_out"]["mu"]["cer"]:.4%}. Checkpoint selection uses DEV alone; original32 only reports. Remaining errors and TRAIN convergence are shown, not hidden by aggregate improvement. Five DEV writers/repeated evaluation limit benchmark claims. Geometry stays frozen; no unrestricted joint training or generation readiness claim.</p>'
    chunks.append('</table>'+conclusion+'<h3>Paired CPU per-line error changes</h3><pre>'+esc(json.dumps({g:{k:{a:b for a,b in r.items() if a!='lines'} for k,r in v.items()} for g,v in transitions.items()},indent=2))+'</pre>')
    if convergence:
        final_rows=['<h3>Final update metrics (NOT DEV-selected)</h3><p>These are final8000-update heads, not automatic replacements for DEV-selected checkpoints. A lower TRAIN loss or report32 score cannot overrule worse DEV selection.</p><table border="1"><tr><th>arm/last step</th><th>TRAIN mean CER</th><th>DEV mean/posterior CER</th><th>report32 mean/posterior CER</th></tr>']
        for name,r in results.items():
            v=r['history'][-1]['groups'];final_rows.append(f'<tr><td>{name}/{r["last_step"]}</td><td>{v["train"]["mu"]["cer"]:.4%}</td><td>{v["dev"]["mu"]["cer"]:.4%}/{v["dev"]["sampled"]["cer"]:.4%}</td><td>{v["held_out"]["mu"]["cer"]:.4%}/{v["held_out"]["sampled"]["cer"]:.4%}</td></tr>')
        chunks.append(''.join(final_rows)+'</table>')
    if convergence:
        chunks.append('<h3>CPU readout diagnostic: greedy versus prefix beam10</h3><p>Same selected mean logits; beam sums CTC paths with all alphabet columns, no lexicon/language model. Width10 fixed in advance; checkpoints were selected on GREEDY DEV, never beam/report scores. No sampled-z beam evaluation, no training or geometry change.</p><pre>'+esc(json.dumps({k:v['groups'] for k,v in beam.items()},indent=2))+'</pre>')
    chunks.extend(['<img src="learning.png"><h2>Protected codec: all8352 lines preflight</h2><pre>'+esc(json.dumps(brief,indent=2))+'</pre>',
        '<p>Point/difference errors use nonuniform RDP sample index; reported turn errors are geometric within true strokes. Authentic target corners preserved. Source tensors unchanged after each arm; one shared source render saved before head fitting. No joint geometry/CTC/KL/style or InkDiT. Posterior CPU/CUDA random streams differ.</p>'])
    for group in ('held_out','dev'):
        chunks.append('<h2>'+group+' selected mean transcripts (CPU reload)</h2><table border="1"><tr><th>ID</th><th>target</th><th>before</th><th>'+esc(names[0])+'</th><th>'+esc(names[1])+'</th></tr>')
        for sid in m['splits'][group]:chunks.append('<tr><td>'+esc(sid)+'</td><td>'+esc(texts[sid])+'</td>'+''.join('<td>'+esc(maps[n][sid]['mu']['decoded'])+'</td>' for n in (parent_name,*names))+'</tr>')
        chunks.append('</table>')
    for group,page,fname in pages:chunks.append(f'<h2>{group} page{page}: target / shared frozen reconstruction</h2><img loading="lazy" src="{fname}">')
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    (out/'index.html').write_text('\n'.join(chunks));publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),paired=paired,codec=brief,arms={k:dict(best_step=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],held_out_cer=r['selected']['held_out']['mu']['cer']) for k,r in results.items()})
