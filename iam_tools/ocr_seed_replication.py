"""Pinned two-seed4/8-point OCR comparison; no ensemble or new DEV samples."""
import html,json
from pathlib import Path
from .ocr_convergence import POOL_SHA
from .frozen_ocr_study import SHA
from .pen_ab import file_sha

REFERENCE='checkpoints/iam_ocr_frame_study/20261007-052514/report/summary.json'
REFERENCE_SHA='7e6f8385567840df0863d060a7a5b587b1b7f54d2585414574e2226d2f6dca58'
SEEDS=(42,137)
# These are fixed intervention/selection/data fields, not cosmetic metadata.
FIXED_FIELDS=('cfg','source_rel','source_sha256','pool_rel','pool_manifest_sha256','points_per_frame',
    'feature_mode','feature_stats','feature_calibration_ids','train_ids','dev_ids','held_out_ids','common_train_probe',
    'posterior_evaluation_ids','posterior_draws','posterior_policy','attention_radius','physical_batch','encoder_physical_batch',
    'schedule_seed','schedule_sha256','base_lr','final_lr','lr_drop_step','betas','weight_decay','clip','dropout','blank_bias',
    'max_updates','max_wall_seconds','eval_every','train_latent','entire_codec_frozen','selection','intervention','checkpoint_contract')


def compare(first,second,records):
    summaries=(first,second);arms=('points8','points4');rows=[];writer_rows={};changes={};initials=[]
    for expected,s in zip(SEEDS,summaries):
        if s['pool_manifest_sha256']!=POOL_SHA or s['source_sha256']!=SHA or set(s['results'])!=set(arms):raise ValueError('pinned fixed pool/codec/two-arm study required')
        if not all(s['paired'].values()):raise ValueError('within-seed pairing failed')
        cfg=s['configs'];initials.append(s['results']['points8']['initial_head_tensor_sha256'])
        if cfg['points8']['seed']!=expected or cfg['points4']['seed']!=expected:raise ValueError('predeclared seeds42/137 required, not chosen best seeds')
        for arm in arms:
            r=s['results'][arm]
            if r['last_step']!=8000 or r['stop_reason']!='budget_completed' or not r['entire_codec_bitwise_unchanged']:raise ValueError('complete equal8000-update frozen-codec arms required')
            if s['cpu_reload'][arm]['mean_cpu_gpu_transcript_differences']:raise ValueError('selected CPU/GPU transcript reload disagreement')
        rows.append(dict(seed=expected,arms={a:dict(selected_step=s['results'][a]['best_step'],groups=s['results'][a]['selected'],selected_sha256=s['results'][a]['selected_sha256']) for a in arms}))
        writer_rows[str(expected)]={};changes[str(expected)]={}
        for group,key in [('dev','dev_ids'),('held_out','held_out_ids')]:
            ids=cfg['points8'][key];values=s['cpu_line_error_changes'][group]['lines'];by={v['sample_id']:v for v in values}
            if set(by)!=set(ids) or len(by)!=len(values):raise ValueError('complete unique paired evaluation rows required')
            writers={};totals={'errors8':0,'errors4':0,'characters':0}
            for sid in ids:
                r=records[sid];v=by[sid];w=str(r['writer_id']);row=writers.setdefault(w,dict(lines=0,characters=0,errors8=0,errors4=0))
                if not r['text'] or any(type(v[k]) is not int or v[k]<0 for k in ('errors8','errors4')):raise ValueError('nonnegative edit counts/nonempty target required')
                row['lines']+=1;row['characters']+=len(r['text']);row['errors8']+=v['errors8'];row['errors4']+=v['errors4']
                for k in totals:totals[k]+=len(r['text']) if k=='characters' else v[k]
            for row in writers.values():
                row['cer8']=row['errors8']/row['characters'];row['cer4']=row['errors4']/row['characters'];row['four_point_improves']=row['errors4']<row['errors8']
            for a,k in [('points8','errors8'),('points4','errors4')]:
                if abs(totals[k]/totals['characters']-s['results'][a]['selected'][group]['mu']['cer'])>1e-12:raise ValueError('paired line counts do not match selected CER')
            writer_rows[str(expected)][group]=writers
            changes[str(expected)][group]=dict(improved=sum(v['errors4']<v['errors8'] for v in values),tied=sum(v['errors4']==v['errors8'] for v in values),worsened=sum(v['errors4']>v['errors8'] for v in values),**totals)
    if initials[0]==initials[1]:raise ValueError('different-seed initial head states must actually differ')
    for arm in arms:
        for field in FIXED_FIELDS:
            if first['configs'][arm][field]!=second['configs'][arm][field]:raise ValueError('uncontrolled across-seed change: '+field)
        if first['results'][arm]['sample_schedule_sha256']!=second['results'][arm]['sample_schedule_sha256']:raise ValueError('across-seed sample schedule changed')
    averages={g:{a:sum(s['results'][a]['selected'][g]['mu']['cer'] for s in summaries)/2 for a in arms} for g in ('dev','held_out')}
    return dict(seeds=list(SEEDS),rows=rows,per_writer=writer_rows,paired_line_changes=changes,mean_cer_across_seeds=averages,
        both_seeds_four_point_improves={g:all(s['results']['points4']['selected'][g]['mu']['cer']<s['results']['points8']['selected'][g]['mu']['cer'] for s in summaries) for g in ('dev','held_out')},
        same_sample_schedule=True,different_initialization_dropout_seeds=True,geometry_frozen=True,
        caveats='Same128 DEV lines/five writers reused in both seeds, NOT256 independent samples, no ensemble/confidence interval/test benchmark. Feature/calibration/position granularity package; dropout not paired within differing-frame shapes. Original32 report-only and codec prior prompt overlap remain.')


def publish(directory,root='/data'):
    root=Path(root);directory=Path(directory);ref=root/REFERENCE;current=directory/'report/summary.json'
    if file_sha(ref)!=REFERENCE_SHA:raise ValueError('immutable seed42 reference summary changed')
    from .ocr_pool_study import load_pool
    _,m,_=load_pool(root,POOL_SHA)
    first=json.loads(ref.read_text());second=json.loads(current.read_text());result=compare(first,second,m['records'])
    result['provenance']=dict(reference_summary_rel=REFERENCE,reference_summary_sha256=REFERENCE_SHA,current_summary_sha256=file_sha(current),source_sha256=SHA,pool_sha256=POOL_SHA)
    out=directory/'replication';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    chunks=['<meta charset="utf-8"><title>Four-point OCR:two-seed replication</title><style>body{font-family:sans-serif;max-width:1300px;margin:2em auto}td{padding:.4em}pre{white-space:pre-wrap}</style><h1>4 versus8-point OCR:seeds42 and137</h1>',
      '<p>Head initialization AND training dropout seed changed; data-order seed43, all8192 training IDs, labels/source/calibration/hyperparameters and8000 updates unchanged. Same fresh weights within each pair; independent weights across seeds. Codec frozen throughout. Compare reader captions, NOT supposed new handwriting.</p>',
      '<p><a href="../../20261007-052514/report/index.html">Seed42 report/gallery</a> · <a href="../report/index.html">Seed137 report/gallery</a> · <a href="summary.json">Exact metrics/checks</a></p>',
      '<table border="1"><tr><th>seed</th><th>reader/selected step</th><th>TRAIN mean CER</th><th>DEV mean/posterior CER</th><th>report32 mean/posterior CER</th></tr>']
    for r in result['rows']:
        for arm,a in r['arms'].items():
            v=a['groups'];chunks.append(f'<tr><td>{r["seed"]}</td><td>{arm}/{a["selected_step"]}</td><td>{v["train"]["mu"]["cer"]:.4%}</td><td>{v["dev"]["mu"]["cer"]:.4%}/{v["dev"]["sampled"]["cer"]:.4%}</td><td>{v["held_out"]["mu"]["cer"]:.4%}/{v["held_out"]["sampled"]["cer"]:.4%}</td></tr>')
    chunks.append('</table><h2>Per-writer DEV changes</h2><table border="1"><tr><th>seed/writer</th><th>lines/characters</th><th>8-point CER/errors</th><th>4-point CER/errors</th></tr>')
    for seed,groups in result['per_writer'].items():
        for writer,w in sorted(groups['dev'].items()):chunks.append(f'<tr><td>{seed}/{html.escape(writer)}</td><td>{w["lines"]}/{w["characters"]}</td><td>{w["cer8"]:.4%}/{w["errors8"]}</td><td>{w["cer4"]:.4%}/{w["errors4"]}</td></tr>')
    chunks.append('</table><h2>Replication result and preserved regressions</h2><pre>'+html.escape(json.dumps({k:result[k] for k in ['both_seeds_four_point_improves','mean_cer_across_seeds','paired_line_changes']},indent=2))+'</pre><p>'+html.escape(result['caveats'])+'</p><p>No joint VAE/CTC/KL/style/InkDiT promotion based on these readout comparisons alone. Frozen geometry/pen gates and CPU selected192-line reload are in each dated report.</p>')
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes());(out/'index.html').write_text('\n'.join(chunks))
    return dict(output=str(out/'index.html'),both_seeds_four_point_improves=result['both_seeds_four_point_improves'],mean_cer_across_seeds=result['mean_cer_across_seeds'])
