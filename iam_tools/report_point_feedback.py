"""Eight TRAIN-line point-feedback capacity report with fail-closed provenance."""
import hashlib
import html
import json
from pathlib import Path
import tarfile
import h5py
import numpy as np
from .point_feedback_study import ARMS,validate,learning_rate,checked_path
from .report_autoregressive_study import svg
from .pen_ab import file_sha
from .writer_expansion import training_schedule
from .generation_capacity import DATA,SOURCE_H5_SHA
from .offset_error_audit import offset_error_groups,aggregate_groups


def verify_log(cfg,result,rows,ids):
    if [r['step'] for r in rows]!=list(range(1,result['last_step']+1)) or not rows:raise ValueError('complete update log required')
    schedule=list(training_schedule(ids,cfg['max_updates'],accumulation=cfg['batch'],seed=cfg['schedule_seed']))
    if [r['sample_ids'] for r in rows]!=schedule[:len(rows)] or result['schedule_sha256']!=hashlib.sha256(json.dumps(schedule).encode()).hexdigest():raise ValueError('matched declared TRAIN schedule required')
    calibration=result['calibration'];weight=calibration['weight']
    if not calibration['state_rng_unchanged'] or len(calibration['rows'])!=cfg['calibration_batches'] or not np.isclose(weight*np.median([r['pen_norm'] for r in calibration['rows']])/np.median([r['offset_norm'] for r in calibration['rows']]),cfg['pen_gradient_fraction']):raise ValueError('fixed gradient calibration required')
    for r in rows:
        if r['lr']!=learning_rate(r['step'],cfg) or r['pen_weight']!=weight or not np.isclose(r['loss'],r['offset_mse']+weight*r['pen_loss'],rtol=2e-6,atol=2e-7) or not np.isfinite([r['loss'],r['raw_grad_norm'],r['mean_sigma']]).all() or r['mean_advance']<=0:raise ValueError('unaltered finite objective/schedule required')
    selected=min(result['history'],key=lambda r:(r['teacher']['offset_mse'],-r['teacher']['min_pen_f1']))
    if selected['step']!=result['best_step'] or [selected['teacher']['offset_mse'],-selected['teacher']['min_pen_f1']]!=result['best_train_score']:raise ValueError('TRAIN-only teacher selection must reproduce')


def generate(relative,root='data'):
    p=checked_path(root,relative);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());validate(cfg,data)
    if file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256'] or file_sha(root/DATA/'source.h5')!=SOURCE_H5_SHA:raise ValueError('immutable source drift')
    with tarfile.open(p/'as-run-source.tar.gz') as archive:
        for name in ['point_feedback_strokes.py','point_feedback_study.py','autoregressive_strokes.py']:
            if archive.extractfile('iam_tools/'+name).read()!=(Path(__file__).parent/name).read_bytes():raise ValueError('as-run implementation drift')
    results={a:json.loads((p/a/'result.json').read_text()) for a in ARMS}
    if results['no_feedback']['initial_state_sha256']!=results['point_feedback']['initial_state_sha256'] or results['no_feedback']['calibration']!=results['point_feedback']['calibration']:raise ValueError('same fresh weights/shared calibration required')
    evals={}
    for a,r in results.items():
        rows=[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()];verify_log(cfg,r,rows,data['training_ids'])
        if file_sha(p/a/'checkpoint-last.pt')!=r['last_sha256'] or file_sha(p/a/'checkpoint-best.pt')!=r['selected_sha256']:raise ValueError('checkpoint drift')
        ev=json.loads((p/a/f"eval-{r['last_step']}.json").read_text())
        if file_sha(p/a/f"evaluation-{r['last_step']}.h5")!=ev['packed_h5_sha256'] or set(q['sample_id'] for q in ev['teacher_lines'])!=set(data['training_ids']) or set(q['sample_id'] for q in ev['free_lines'] if q['policy']=='correct')!=set(data['training_ids']):raise ValueError('packed outputs/scope drift')
        evals[a]=ev
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());strips=[];decomposition={}
    with h5py.File(root/DATA/'source.h5') as source:
        files={a:h5py.File(p/a/f"evaluation-{results[a]['last_step']}.h5") for a in ARMS}
        try:
            for a in ARMS:decomposition[a]=aggregate_groups([offset_error_groups(source[s]['target'][:],files[a]['teacher/'+s]['points'][:,:2],cfg['offset_stats']) for s in data['training_ids']])
            for sid in data['training_ids']:
                truth=source[sid]['target'][:];entries=[('Source IAM',truth)]
                for a in ARMS:
                    q=files[a]['teacher/'+sid]['points'][:];entries.extend([(a+' TRUE-HISTORY/TRUE-PEN, NOT generation',np.c_[q[:,:2],truth[:,2:]]),(a+' TRUE-HISTORY/PREDICTED-PEN, NOT generation',q)])
                    for policy in evals[a]['free']:
                        g=files[a]['free/'+policy+'/'+sid];r=json.loads(g.attrs['row']);entries.append((a+f' FREE {policy}, {r["generated_points"]} points; learnedEOC={r["found_eoc"]}; reader={r["decoded"]!r}',g['points'][:]))
                chunks=[]
                for j,(label,q) in enumerate(entries):
                    name=sid+f'-{j}.svg';(out/name).write_text(svg(q));chunks.append('<p>'+html.escape(label)+'</p><div class="strip"><img src="'+name+'"></div>')
                strips.append('<h2>'+html.escape(sid+' | '+data['records'][sid]['text'])+'</h2>'+''.join(chunks))
        finally:
            for f in files.values():f.close()
    (out/'offset-error-decomposition.json').write_text(json.dumps(decomposition,indent=2)+'\n')
    summary=dict(arms={a:dict(step=results[a]['last_step'],selected_step=results[a]['best_step'],teacher=evals[a]['teacher'],free=evals[a]['free'],block_phase=evals[a]['block_phase'],train_seconds=results[a]['train_seconds'],clip_fraction=results[a]['clip_fraction']) for a in ARMS},same_fresh_parameters=True,shared_gradient_calibration=True,matched_training_schedule=True,source_archive_sha256=cfg['source_archive_sha256'],no_new_confirmation=True,not_promoted=True,scope='all8 established fixedTRAIN; capacity pilot, not generalization; true-history output NOT generation; final outputs shown regardless of selected checkpoint')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table=[]
    for a in ARMS:
        t=evals[a]['teacher'];f=evals[a]['free']['correct'];table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [a,results[a]['last_step'],f"{t['offset_mse']:.6f}",f"{t['x_rmse']:.4f}",f"{t['y_rmse']:.4f}",f"{t['min_pen_f1']:.3f}",f"{100*f['cer']:.2f}%",f['missing_eoc']])+'</tr>')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Point-feedback capacity gate</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}.strip{overflow:auto;border:1px solid #ddd;max-height:600px}.strip img{display:block;max-width:none}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Intra-block point-feedback: eight TRAIN-line capacity gate</h1><p>Matched fresh weights, same8 established TRAIN lines, order and gradient-calibrated objectives. Both use same adaptive eight-point text clock/coarse history. ONLY immediate previous-point feedback inside each8step pointGRU differs. Fused teacher pointGRU sees strictly shifted history; free generation consumes its own offsets/hard pens. No source length/reference/timing teacher/forcedEOC at generation. Common2048point cap. No KL/style/OCR training loss, DEV or new blind gate.</p><p>This NEW autoregressive model needs its own capacity test. Neither teacher forcing nor8TRAIN memorization establishes compositional new-text handwriting. Final checkpoints displayed independent of TRAIN-only teacher-loss selection. Marker-free exact100px/model-unit SVG with proper up-positiveY display, exact aspect, actual singleton tap strokes, no smoothing/width fitting. Scroll divergent strips; truepen rows are explicitly NOT generation.</p><table><tr><th>Arm</th><th>Step</th><th>Teacher offsetMSE</th><th>Teacher X RMSE</th><th>Teacher Y RMSE</th><th>Min penF1</th><th>Free TRAIN CER</th><th>MissingEOC</th></tr>'''+''.join(table)+'''</table><p><a href="summary.json">Final metrics/provenance</a> · <a href="offset-error-decomposition.json">Error topology</a></p>'''+''.join(strips))
    return str(out)
