"""Fail-closed matched roll-in report, all8 TRAIN, final AND selected outcomes."""
import hashlib
import html
import json
from pathlib import Path
import tarfile
import h5py
import numpy as np
from .history_rollin_study import ARMS,PARENT_SHA,validate,checked_path,teacher_gate
from .history_rollin import rollin_probability
from .report_autoregressive_study import svg
from .writer_expansion import training_schedule
from .pen_ab import file_sha
from .generation_capacity import DATA,SOURCE_H5_SHA


def verify_selection(result):
    eligible=[r for r in result['history'] if r['capacity_gate']['passed']]
    if not eligible:raise ValueError('baseline capacity must be eligible')
    best=min(eligible,key=lambda r:(r['free']['correct']['cer'],r['teacher']['offset_mse']))
    if result['best_step']!=best['step'] or result['best_train_score']!=[best['free']['correct']['cer'],best['teacher']['offset_mse']]:
        raise ValueError('capacity-guarded TRAIN-only freeCER selection must reproduce')


def verify_log(cfg,data,result,rows):
    if not rows or [r['step'] for r in rows]!=list(range(1,result['last_step']+1)):raise ValueError('complete sequential actual updates required')
    schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],cfg['batch'],cfg['schedule_seed']))
    if result['schedule_sha256']!=hashlib.sha256(json.dumps(schedule).encode()).hexdigest() or [r['sample_ids'] for r in rows]!=schedule[:len(rows)]:raise ValueError('matched declared TRAIN order required')
    for r in rows:
        p=rollin_probability(r['step'],cfg['rollin_max_probability'],cfg['rollin_ramp_steps']) if result['arm']=='rollin' else 0.
        if r['rollin_probability']!=p or r['lr']!=cfg['lr'] or r['pen_weight']!=cfg['pen_weight'] or r['total_step']!=3000+r['step'] or not np.isclose(r['loss'],r['offset_mse']+cfg['pen_weight']*r['pen_loss'],rtol=2e-6,atol=2e-7):raise ValueError('unaltered stated policy/objective/continuation required')
        if not 0<=r['selected_transitions']<=r['eligible_transitions'] or (p==0 and r['selected_transitions']) or r['selected_fraction']!=r['selected_transitions']/r['eligible_transitions']:raise ValueError('applied transition counts must agree')
        if len(r['draws_sha256'])!=64 or len(r['selected_sha256'])!=64 or not np.isfinite([r['loss'],r['raw_grad_norm'],r['mean_advance'],r['mean_sigma']]).all() or r['mean_advance']<=0:raise ValueError('finite audited updates and RNG/mask hashes required')
    verify_selection(result)


def generate(relative,root='data'):
    root=Path(root);p=checked_path(root,relative);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());validate(cfg,data);parent=json.loads((p/'parent-eval.json').read_text())
    if file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256'] or file_sha(root/cfg['parent_checkpoint_relative'])!=PARENT_SHA or file_sha(root/DATA/'source.h5')!=SOURCE_H5_SHA:raise ValueError('pinned source/archive/checkpoint drift')
    with tarfile.open(p/'as-run-source.tar.gz') as archive:
        for name in ['history_rollin.py','history_rollin_study.py','point_feedback_strokes.py','point_feedback_study.py']:
            if archive.extractfile('iam_tools/'+name).read()!=(Path(__file__).parent/name).read_bytes():raise ValueError('as-run implementation drift')
    results={a:json.loads((p/a/'result.json').read_text()) for a in ARMS};logs={a:[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for a,r in results.items():
        verify_log(cfg,data,r,logs[a])
        if r['initial_state_sha256']!=cfg['parent_initial_model_digest'] or file_sha(p/a/'checkpoint-last.pt')!=r['last_sha256'] or file_sha(p/a/'checkpoint-best.pt')!=r['selected_sha256']:raise ValueError('exact initial and selected/final parameters required')
    if results['teacher']['initial_optimizer_tensor_digest']!=results['rollin']['initial_optimizer_tensor_digest'] or [r['draws_sha256'] for r in logs['teacher'][:min(map(len,logs.values()))]]!=[r['draws_sha256'] for r in logs['rollin'][:min(map(len,logs.values()))]]:raise ValueError('matched initial optimizer and draws required')
    evals={};readings={}
    for a,r in results.items():
        evals[a]={};readings[a]={}
        for h in r['history']:
            step=h['step'];ev=json.loads((p/a/f'eval-{step}.json').read_text());reading=json.loads((p/a/f'teacher-reading-{step}.json').read_text());gate=teacher_gate(ev,reading,parent)
            if gate!=h['capacity_gate'] or gate!=reading['capacity_gate'] or file_sha(p/a/f'evaluation-{step}.h5')!=ev['packed_h5_sha256']:raise ValueError('all checkpoint gate/output hashes must reproduce')
            if set(x['sample_id'] for x in ev['teacher_lines'])!=set(data['training_ids']) or set(x['sample_id'] for x in ev['free_lines'] if x['policy']=='correct')!=set(data['training_ids']):raise ValueError('ALL8 teacher/free rows required')
            evals[a][step]=ev;readings[a][step]=reading
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());table=[];summary={}
    for a,r in results.items():
        final=evals[a][r['last_step']];selected=evals[a][r['best_step']];reading=readings[a][r['last_step']]
        summary[a]=dict(final_step=r['last_step'],total_step=3000+r['last_step'],selected_step=r['best_step'],final_teacher=final['teacher'],final_free=final['free'],final_reading=reading,selected_teacher=selected['teacher'],selected_free=selected['free'],train_seconds=r['train_seconds'],clip_fraction=r['clip_fraction'],final_capacity_gate=reading['capacity_gate'],history=r['history'],applied_rollin_fraction=sum(x['selected_transitions'] for x in logs[a])/sum(x['eligible_transitions'] for x in logs[a]))
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [a,r['last_step'],r['best_step'],f"{100*final['free']['correct']['cer']:.2f}%",f"{100*selected['free']['correct']['cer']:.2f}%",f"{final['teacher']['x_rmse']:.5f}",f"{final['teacher']['y_rmse']:.5f}",f"{100*reading['true_pen_cer']:.2f}%",reading['capacity_gate']['passed']])+'</tr>')
    galleries=[]
    with h5py.File(root/DATA/'source.h5') as src,h5py.File(root/cfg['parent']/ 'no_feedback/evaluation-3000.h5') as baseline:
        for sid in data['training_ids']:
            truth=src[sid]['target'][:];entries=[('Source IAM',truth),('PARENT step3000 FREE correct, not oracle length',baseline['free/correct/'+sid]['points'][:])]
            for a,r in results.items():
                with h5py.File(p/a/f"evaluation-{r['last_step']}.h5") as outputs:
                    q=outputs['teacher/'+sid]['points'][:];entries.extend([(a+' FINAL TRUE-HISTORY / TRUE-PEN, NOT generation',np.c_[q[:,:2],truth[:,2:]]),(a+' FINAL TRUE-HISTORY / PREDICTED-PEN, NOT generation',q)])
                    for policy in evals[a][r['last_step']]['free']:
                        g=outputs['free/'+policy+'/'+sid];row=json.loads(g.attrs['row']);entries.append((a+f' FINAL FREE {policy}, {row["generated_points"]}points EOC={row["found_eoc"]}, conditioning={row["conditioning_text"]!r}, reader={row["decoded"]!r}',g['points'][:]))
                if r['best_step']!=r['last_step']:
                    with h5py.File(p/a/f"evaluation-{r['best_step']}.h5") as selected:
                        entries.append((a+f' SELECTED FREE correct step{r["best_step"]}',selected['free/correct/'+sid]['points'][:]))
            chunks=[]
            for j,(label,q) in enumerate(entries):
                name=sid+f'-{j}.svg';(out/name).write_text(svg(q));chunks.append('<p>'+html.escape(label)+'</p><div class="strip"><img src="'+name+'"></div>')
            galleries.append('<h2>'+html.escape(sid+' | '+data['records'][sid]['text'])+'</h2>'+''.join(chunks))
    (out/'summary.json').write_text(json.dumps(dict(arms=summary,parent=PARENT_SHA,same_initial_model_optimizer=True,matched_order_rng_draws=True,scope='ALL8 establishedTRAIN/7writers; capacity-preserving history robustness, NOT newtext generalization or final model promotion',no_new_confirmation=True,not_promoted=True),indent=2)+'\n')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Own-history robustness continuation</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}.strip{overflow:auto;border:1px solid #ddd;max-height:600px}.strip img{display:block;max-width:none}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Matched generated-history roll-in versus teacher continuation</h1><p>Same ORIGINAL step3000 model/optimizer, eight fixedTRAIN lines across7writers, schedule, draws, architecture and objective.1000continuation updates at LR1e-5. Teacher p0; roll-in joint own offsets/hardpens p ramps to.2 at500, detached selected input feedback, normal hidden-state BPTT. No added smoothing, cumulative-position loss, KL/style/OCR training objective, augmentation or reference at free generation. This is scheduled sampling, not unbiased likelihood training; source-index targets can become ambiguous after divergence.</p><p>GENUINE free output uses own history and only requested text/writer plus common2048point cap, learned firstEOC, no oracle source length. Source-conditioned rows are NOT generation. All8 evaluated/displayed, no blind prompt opened. Select freeTRAIN CER only among perline preserved teacher-capacity checkpoints; final results displayed even when rejected. Reader is corpus-familiar; this toy does NOT prove unseen-text composition or clean multi-example-per-writer learning.</p><table><tr><th>Arm</th><th>Final update</th><th>Selected update</th><th>Final freeTRAIN CER</th><th>Selected freeTRAIN CER</th><th>Final teacher X RMSE</th><th>Final teacher Y RMSE</th><th>Teacher TRUE-PEN CER</th><th>Capacity guard</th></tr>'''+''.join(table)+'''</table><p><a href="summary.json">Complete metrics/guards</a>. Swapped conditioning error against original target is a control, not the requested-text score; teacher-reading JSON also logs CER against actually supplied swapped text. Marker-free SVG uses fixed100px/model-unit, exact aspect, proper up-positiveY display and actual singleton ink taps; no smoothing/clipping/width fitting. Scroll divergent outputs rather than rescaling them to look normal.</p>'''+''.join(galleries))
    return str(out)
