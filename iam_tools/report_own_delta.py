"""Fail-closed own target-displacement control, all8, final AND selected."""
import hashlib,html,json,tarfile
from pathlib import Path
import h5py,numpy as np
from . import own_delta_study
from .report_generated_prefix import verify_pairing
from .history_rollin_study import teacher_gate
from .report_history_rollin import verify_selection
from .report_autoregressive_study import svg
from .writer_expansion import training_schedule
from .pen_ab import file_sha
from .generation_capacity import DATA,SOURCE_H5_SHA
PARENT_SHA=own_delta_study.PARENT_SHA

def study_module(relative):
    if relative.startswith('checkpoints/iam_own_delta/'):return own_delta_study
    raise ValueError('declared own-displacement control required')

def core_files(relative,recovery=False):
    return ['own_delta.py','own_delta_study.py','continuous_prefix.py','generated_prefix.py','generated_prefix_study.py','history_rollin.py','point_feedback_strokes.py','point_feedback_study.py','autoregressive_strokes.py','cumulative_xy.py']

def verify_log(cfg,data,result,rows,module=own_delta_study):
    if not rows or [r['step'] for r in rows]!=list(range(1,result['last_step']+1)):raise ValueError('all actual sequential updates required')
    schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],cfg['batch'],cfg['schedule_seed']))
    if result['schedule_sha256']!=hashlib.sha256(json.dumps(schedule).encode()).hexdigest() or [r['sample_ids'] for r in rows]!=schedule[:len(rows)]:raise ValueError('matched declared TRAIN order required')
    wd=module.arm_weight(result['arm'],cfg)
    if result['initial_optimizer_tensor_digest']!=cfg['parent_initial_optimizer_digest'] or result['parent_checkpoint_sha256']!=PARENT_SHA or result['parent_joint_step']!=1000 or not result['reader_unchanged'] or not result['not_promoted']:raise ValueError('exact restored parent/moments/frozen reader required')
    if result['last_step']>cfg['max_updates'] or result['stop'] not in ('budget_completed','wall_limit') or (result['stop']=='budget_completed' and result['last_step']!=cfg['max_updates']):raise ValueError('actual bounded budget required')
    for r in rows:
        base=r['teacher_offset']+cfg['pen_weight']*r['teacher_pen']+cfg['teacher_anchor_weight']*r['teacher_xy'];complete=base+cfg['own_xy_weight']*r['own_xy']+cfg['own_pen_weight']*r['own_pen']
        delta=r['own_ink_delta'] if result['arm']=='ink_delta' else r['own_all_delta'];loss=complete+wd*delta
        if r['lr']!=cfg['lr'] or r['pen_weight']!=cfg['pen_weight'] or r['teacher_anchor_weight']!=cfg['teacher_anchor_weight'] or r['own_xy_weight']!=cfg['own_xy_weight'] or r['own_pen_weight']!=cfg['own_pen_weight'] or r['parent_joint_step']!=1000 or r['delta_weight']!=wd or not r['continuous_feedback_gradient']:raise ValueError('same complete parent scalar/continuous graph plus declared new term required')
        if any(not np.isclose(r[k],v,rtol=2e-6,atol=2e-7) for k,v in [('base_loss',base),('complete_loss',complete),('loss',loss),('own_all_delta',r['own_offset_diagnostic'])]):raise ValueError('actual base/complete/delta scalar objective must reproduce')
        if not np.isfinite([r['loss'],r['complete_loss'],r['own_xy'],r['own_pen'],r['own_ink_delta'],r['own_all_delta'],r['raw_grad_norm'],r['mean_teacher_advance'],r['mean_own_advance']]).all() or r['clipped']!=(r['raw_grad_norm']>cfg['clip']):raise ValueError('finite updates and exact clipping flags required')
        check=r['gradient_check']
        if bool(check)!=(r['step']==1 or r['step']%250==0):raise ValueError('actual scheduled gradient diagnostics required')
        if check and not np.isclose(check['weighted_delta_ratio'],wd*check['delta_norm']/check['complete_norm'],rtol=1e-10,atol=1e-12):raise ValueError('actual complete-objective gradient ratio required')
    if result['clip_fraction']!=sum(r['clipped'] for r in rows)/len(rows):raise ValueError('clipped fraction must reproduce from actual updates')
    verify_selection(result)

def generate(relative,root='data'):
    root=Path(root);module=study_module(relative);ARMS=module.ARMS;p=module.checked_path(root,relative);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());module.validate(cfg,data);parent=json.loads((p/'parent-eval.json').read_text())
    if file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256'] or file_sha(root/cfg['parent_checkpoint_relative'])!=PARENT_SHA or file_sha(root/DATA/'source.h5')!=SOURCE_H5_SHA:raise ValueError('pinned source/archive/checkpoint drift')
    with tarfile.open(p/'as-run-source.tar.gz') as archive:
        for name in core_files(relative,bool(cfg.get('recovery'))):
            if archive.extractfile('iam_tools/'+name).read()!=(Path(__file__).parent/name).read_bytes():raise ValueError('as-run implementation drift')
    results={a:json.loads((p/a/'result.json').read_text()) for a in ARMS};logs={a:[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for a,r in results.items():
        verify_log(cfg,data,r,logs[a],module)
        if r['initial_state_sha256']!=cfg['parent_initial_model_digest'] or file_sha(p/a/'checkpoint-last.pt')!=r['last_sha256'] or file_sha(p/a/'checkpoint-best.pt')!=r['selected_sha256']:raise ValueError('exact initial and selected/final parameters required')
    verify_pairing(cfg,results,logs,module)
    evals={};readings={}
    for a,r in results.items():
        evals[a]={};readings[a]={}
        for h in r['history']:
            step=h['step'];ev=json.loads((p/a/f'eval-{step}.json').read_text());reading=json.loads((p/a/f'teacher-reading-{step}.json').read_text());gate=teacher_gate(ev,reading,parent)
            if gate!=h['capacity_gate'] or gate!=reading['capacity_gate'] or file_sha(p/a/f'evaluation-{step}.h5')!=ev['packed_h5_sha256']:raise ValueError('all checkpoint gate/output hashes must reproduce')
            if set(x['sample_id'] for x in ev['source_rollout_lines'])!=set(data['training_ids']) or set(x['sample_id'] for x in ev['teacher_lines'])!=set(data['training_ids']) or set(x['sample_id'] for x in ev['free_lines'] if x['policy']=='correct')!=set(data['training_ids']):raise ValueError('ALL8 teacher/free rows required')
            evals[a][step]=ev;readings[a][step]=reading
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());table=[];timeline=[];stops=[];summary={}
    for a,r in results.items():
        final=evals[a][r['last_step']];selected=evals[a][r['best_step']];reading=readings[a][r['last_step']]
        summary[a]=dict(final_step=r['last_step'],parent_joint_step=1000,selected_step=r['best_step'],final_teacher=final['teacher'],final_free=final['free'],final_reading=reading,selected_teacher=selected['teacher'],selected_free=selected['free'],train_seconds=r['train_seconds'],clip_fraction=r['clip_fraction'],final_capacity_gate=reading['capacity_gate'],history=r['history'],final_source_rollout=final['source_rollout'],selected_source_rollout=selected['source_rollout'],gradient_checks=[dict(step=row['step'],**row['gradient_check']) for row in logs[a] if row['gradient_check']])
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [a,r['last_step'],r['best_step'],f"{100*final['free']['correct']['cer']:.2f}%",f"{100*selected['free']['correct']['cer']:.2f}%",f"{final['teacher']['x_rmse']:.5f}",f"{final['teacher']['y_rmse']:.5f}",f"{100*reading['true_pen_cer']:.2f}%",reading['capacity_gate']['passed']])+'</tr>')
    for a,r in results.items():
        for h in r['history']:
            step=h['step'];ev=evals[a][step];roll=ev['source_rollout'];m=roll['metrics'];g=h['capacity_gate']
            values=[a,step,f"{100*ev['free']['correct']['cer']:.2f}%",ev['free']['correct']['exact'],g['failures'],f"{m['x_rmse']:.5f}",f"{m['y_rmse']:.5f}",f"{m['mean_line_first_difference_rmse']:.5f}",f"{m['mean_line_turn_p90']:.2f}",f"{m['min_pen_f1']:.3f}",f"{100*roll['predicted_pen_cer']:.2f}%"]
            timeline.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
        for stage,step in [('selected',r['best_step']),('final',r['last_step'])]:
            for row in evals[a][step]['free_lines']:
                if row['policy']!='correct':continue
                values=[a,stage,step,row['sample_id'],row['generated_points'],row['text_characters'],f"{row['center_at_learned_stop_or_cap']:.2f}",f"{row['eos_attention_at_stop_or_cap']:.4f}",row['decoded']]
                stops.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    diagnostics_html='<h2>All checkpoints: actual generation versus source-length own-rollout diagnostics</h2><p>RMSE/difference/turn/pen columns below describe the supervised source-length own-history rollout, NOT genuinely stopped free output. Index differences are not physical velocity or curvature; target turn angles use within-stroke segments.</p><table><tr><th>Arm</th><th>Step</th><th>Actual free CER</th><th>Exact</th><th>Teacher guard failures</th><th>Own rollout X</th><th>Own rollout Y</th><th>First diff</th><th>Mean perline turn p90 (deg)</th><th>Min pen F1</th><th>Own predicted-pen CER</th></tr>'+''.join(timeline)+'</table><h2>Actual learned stops (all eight, selected AND final)</h2><p>Attention center is a learned token clock, NOT verified glyph alignment. No stop gate is applied to it. Source length is not supplied to free generation.</p><table><tr><th>Arm</th><th>Stage</th><th>Step</th><th>Sample</th><th>Generated points</th><th>Text chars</th><th>Clock</th><th>EOS attention</th><th>Reader output</th></tr>'+''.join(stops)+'</table>'
    galleries=[]
    with h5py.File(root/DATA/'source.h5') as src,h5py.File(root/cfg['parent']/'continuous_xy/evaluation-1000.h5') as baseline:
        for sid in data['training_ids']:
            truth=src[sid]['target'][:];entries=[('Source IAM',truth),('SOURCE selected continuous_xy1000 FREE correct, no oracle length',baseline['free/correct/'+sid]['points'][:])]
            for a,r in results.items():
                with h5py.File(p/a/f"evaluation-{r['last_step']}.h5") as outputs:
                    q=outputs['teacher/'+sid]['points'][:];entries.extend([(a+' FINAL TRUE-HISTORY / TRUE-PEN, NOT generation',np.c_[q[:,:2],truth[:,2:]]),(a+' FINAL TRUE-HISTORY / PREDICTED-PEN, NOT generation',q),(a+' FINAL FULL-OWN HISTORY / TRUE-PEN, SOURCE-LENGTH NOT generation',np.c_[outputs['source_rollout/'+sid]['points'][:,:2],truth[:,2:]]),(a+' FINAL FULL-OWN HISTORY / PREDICTED-PEN, SOURCE-LENGTH NOT generation',outputs['source_rollout/'+sid]['points'][:])])
                    for policy in evals[a][r['last_step']]['free']:
                        g=outputs['free/'+policy+'/'+sid];row=json.loads(g.attrs['row']);entries.append((a+f' FINAL FREE {policy}, {row["generated_points"]}points EOC={row["found_eoc"]}, conditioning={row["conditioning_text"]!r}, reader={row["decoded"]!r}',g['points'][:]))
                if r['best_step']!=r['last_step']:
                    with h5py.File(p/a/f"evaluation-{r['best_step']}.h5") as selected:
                        entries.append((a+f' SELECTED FREE correct step{r["best_step"]}',selected['free/correct/'+sid]['points'][:]))
            chunks=[]
            for j,(label,q) in enumerate(entries):
                name=sid+f'-{j}.svg';(out/name).write_text(svg(q));chunks.append('<p>'+html.escape(label)+'</p><div class="strip"><img src="'+name+'"></div>')
            galleries.append('<h2>'+html.escape(sid+' | '+data['records'][sid]['text'])+'</h2>'+''.join(chunks))
    (out/'summary.json').write_text(json.dumps(dict(arms=summary,parent=PARENT_SHA,same_initial_model_restored_optimizer=True,delta_calibration=cfg['delta_calibration'],matched_actual_order_initial_forward_terms=True,scope='ALL8 establishedTRAIN/7writers; capacity-preserving history robustness, NOT newtext generalization or final model promotion',no_new_confirmation=True,not_promoted=True),indent=2)+'\n')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Own-history target displacement controls</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}.strip{overflow:auto;border:1px solid #ddd;max-height:600px}.strip img{display:block;max-width:none}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Controlled own-history target displacement matching</h1><p>'''+html.escape(cfg['profile']+' | '+cfg['training']+' | '+cfg['loss']+' | '+cfg['caveats'])+'''</p><p>Same pinned source model AND whole-model AdamW state, order and architecture. LR1e-5,2000updates/1800TRAIN-seconds cap,order65142/trainseed65143,clip5,all8 familiarTRAIN/7writers. Source-length own rollouts continue after predictedEOC ONLY for supervised training/diagnostics; do NOT call them actual free generation. No smoothing, endpoint correction, posthoc inference gate, KL/style/OCR training loss. All3 restore the SAME selected continuous1000 checkpoint and whole AdamW state. Control retains its complete teacher+ownXY+ownpen objective. Ink_delta adds normalized target-displacement matching only within TRUE strokes (including cross-block links); all_delta includes origin/pen-up jumps too. Each added coefficient is fixed from25% of the initial COMPLETE-parent full-model gradient, not25% of loss or of each head. No generic smoothing; target corners preserved. Source-index timing can become ambiguous after divergence. Hardpen feedback stays argmax/nondifferentiable.</p><p>GENUINE free output uses own history and only requested text/writer plus common2048point cap, learned firstEOC, no oracle source length. Source-conditioned rows are NOT generation. All8 evaluated/displayed, no blind prompt opened. Select freeTRAIN CER only among perline preserved teacher-capacity checkpoints; final results displayed even when rejected. Reader is corpus-familiar; this toy does NOT prove unseen-text composition or clean multi-example-per-writer learning.</p><table><tr><th>Arm</th><th>Final update</th><th>Selected update</th><th>Final freeTRAIN CER</th><th>Selected freeTRAIN CER</th><th>Final teacher X RMSE</th><th>Final teacher Y RMSE</th><th>Teacher TRUE-PEN CER</th><th>Capacity guard</th></tr>'''+''.join(table)+'''</table><p><a href="summary.json">Complete metrics/guards</a>. Swapped conditioning error against original target is a control, not the requested-text score; teacher-reading JSON also logs CER against actually supplied swapped text. Marker-free SVG uses fixed100px/model-unit, exact aspect, proper up-positiveY display and actual singleton ink taps; no smoothing/clipping/width fitting. Scroll divergent outputs rather than rescaling them to look normal.</p>'''+diagnostics_html+''.join(galleries))
    return str(out)
