"""Marker-free paired autoregressive pilot review, fail-closed provenance.

Teacher-forced output is explicitly not free generation. No blind gate is opened
and no candidate is reselected using exposed development/visual preference.
"""
import hashlib
import html
import json
from pathlib import Path
import tarfile
import h5py
import numpy as np
import torch
from .autoregressive_study import ARMS,validate,learning_rate,aggregate_teacher,aggregate_free
from .autoregressive_strokes import MonotonicStrokeWriter,StrokePool,decode_offsets
from .generation_capacity import DATA,SOURCE_H5_SHA,DATASET_SHA
from .writer_expansion import training_schedule
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha
from .offset_error_audit import offset_error_groups,aggregate_groups


def verify_log(cfg,result,rows):
    if not rows or [r['step'] for r in rows]!=list(range(1,result['last_step']+1)):raise ValueError('every actual update must be present once in sequence')
    weight=result['calibration']['weight'];c=result['calibration']
    ratio=weight*np.median([r['pen_norm'] for r in c['rows']])/np.median([r['offset_norm'] for r in c['rows']])
    if not c['state_rng_unchanged'] or len(c['rows'])!=cfg['calibration_batches'] or not np.isclose(ratio,cfg['pen_gradient_fraction'],atol=1e-12,rtol=1e-10):raise ValueError('fixed stated gradient calibration required')
    for r in rows:
        if r['pen_weight']!=weight or r['lr']!=learning_rate(r['step'],cfg) or not np.isclose(r['loss'],r['offset_mse']+weight*r['pen_loss'],atol=2e-7,rtol=2e-6):raise ValueError('no unreported objective/schedule mutation permitted')
        if r['mean_advance']<=0 or not np.isfinite([r['loss'],r['offset_mse'],r['pen_loss'],r['raw_grad_norm'],r['mean_advance'],r['mean_sigma']]).all():raise ValueError('finite objective/progress required')


def verify_selection(result):
    history=result['history']
    selected=min(history,key=lambda r:(r['free']['fixed_train8']['correct']['cer'],r['teacher']['offset_mse']))
    if result['best_step']!=selected['step'] or not np.allclose(result['best_train_score'],[selected['free']['fixed_train8']['correct']['cer'],selected['teacher']['offset_mse']],atol=0,rtol=0):raise ValueError('TRAIN-only declared selection must reproduce')


def svg(points):
    """Marker-free vector geometry, exact aspect ratio, fixed100px/model-unit.

    Unlike per-row fit-to-width plots, excessive accumulated drift remains visible
    as a physically tall/wide graphic; browser can scroll the strip. No clipping,
    smoothing, glyph reshaping or truth-boundary substitution on predicted rows.
    """
    a=np.asarray(points)
    if a.ndim!=2 or a.shape[1]!=5 or not len(a) or not np.isfinite(a).all():raise ValueError('finite nonempty XY+pen trajectory required')
    xy=a[:,:2];minimum=xy.min(0);maximum=xy.max(0);span=maximum-minimum
    width=max(40.,100*span[0]+12);height=max(40.,100*span[1]+12);p=(xy-minimum)*100+6; p[:,1]=(maximum[1]-xy[:,1])*100+6
    paths=[];start=0;states=a[:,2:].argmax(-1)
    for j in range(len(p)):
        if states[j]!=0 or j==len(p)-1:
            stroke=p[start:j+1]
            if len(stroke)==1:stroke=np.repeat(stroke,2,axis=0) # actual tap ink via round zero-length segment, NOT vertex markers
            paths.append('<path d="M '+' L '.join(f'{x:.4f},{y:.4f}' for x,y in stroke)+'"/>')
            start=j+1
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}" height="{height:.3f}" viewBox="0 0 {width:.3f} {height:.3f}"><g fill="none" stroke="#111" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round">'+''.join(paths)+'</g></svg>'


def report(directory,root='data',repo='third_party/DiffInk',progress_step=None):
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());validate(cfg,data)
    for file,sha in [(root/DATA/'source.h5',SOURCE_H5_SHA),(root/DATA/'dataset.json',DATASET_SHA),(p/'as-run-source.tar.gz',cfg['source_archive_sha256'])]:
        if file_sha(file)!=sha:raise ValueError('pinned source/archive drift')
    with tarfile.open(p/'as-run-source.tar.gz') as archive:
        for name in ['iam_tools/autoregressive_strokes.py','iam_tools/autoregressive_study.py']:
            if archive.extractfile(name).read()!=(Path(repo)/name).read_bytes():raise ValueError('training-source implementation drift')
    final=progress_step is None;results={};evals={};steps={};guards={}
    for arm in ARMS:
        if final:
            result=json.loads((p/arm/'result.json').read_text());rows=[json.loads(l) for l in (p/arm/'metrics.jsonl').read_text().splitlines()]
            verify_log(cfg,result,rows);verify_selection(result)
            schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],accumulation=cfg['batch'],seed=cfg['schedule_seed']))
            if [r['sample_ids'] for r in rows]!=schedule[:len(rows)] or result['schedule_sha256']!=hashlib.sha256(json.dumps(schedule).encode()).hexdigest():raise ValueError('matched actual TRAIN update order guard')
            if file_sha(p/arm/'checkpoint-best.pt')!=result['selected_sha256'] or file_sha(p/arm/'checkpoint-last.pt')!=result['last_sha256']:raise ValueError('selected/final checkpoint fingerprints required')
            if not result['reader_unchanged'] or result['last_step']>cfg['max_updates'] or result['stop'] not in ['budget_completed','wall_limit']:raise ValueError('bounded final result required')
            if result['stop']=='budget_completed' and result['last_step']!=cfg['max_updates']:raise ValueError('complete actual budget required')
            results[arm]=result;step=result['last_step']
        else:
            if type(progress_step) is not int or progress_step not in cfg['eval_steps']:raise ValueError('explicit completed fixed diagnostic step required')
            step=progress_step
        steps[arm]=step;ev=json.loads((p/arm/f'eval-{step}.json').read_text());h5=p/arm/f'evaluation-{step}.h5'
        if ev['step']!=step or file_sha(h5)!=ev['packed_h5_sha256']:raise ValueError('complete paired saved evaluation guard')
        if {r['sample_id'] for r in ev['teacher_lines']}!=set(data['training_ids']) or len(ev['teacher_lines'])!=256:raise ValueError('all256 teacher scope required')
        if aggregate_teacher(ev['teacher_lines'])!=ev['teacher']:raise ValueError('independently recomputed teacher metrics required')
        for split,ids in [('fixed_train8',data['fixed_train_ids']),('exposed_dev8',data['exposed_dev_ids'])]:
            for policy,group in ev['free'][split].items():
                lines=[r for r in ev['free_lines'] if r['split']==split and r['policy']==policy]
                if {r['sample_id'] for r in lines}!=set(ids) or len(lines)!=8 or aggregate_free(lines)!=group or any(not r['no_source_history'] or not r['no_source_length'] for r in lines):raise ValueError('true target-free fixed8 free evaluation scope required')
        evals[arm]=ev
    if final:
        if results['fixed']['initial_state_sha256']!=results['adaptive']['initial_state_sha256'] or results['fixed']['schedule_sha256']!=results['adaptive']['schedule_sha256'] or results['fixed']['calibration']!=results['adaptive']['calibration']:raise ValueError('identical fresh states/order/actual objective scale required')
        guards['identical_fresh_weights_order_calibration']=True
    folder=p/('report' if final else f'progress-{progress_step}');folder.mkdir(exist_ok=False);(folder/'report-source.py').write_bytes(Path(__file__).read_bytes())
    with h5py.File(root/DATA/'source.h5') as hf:targets={sid:hf[sid]['target'][:] for sid in data['fixed_train_ids']+data['exposed_dev_ids']}
    # No re-selection by visual appeal. Display all fixedTRAIN8 and exposedDEV8.
    with h5py.File(root/DATA/'source.h5') as source:
        decomposition={}
        for arm in ARMS:
            with h5py.File(p/arm/f'evaluation-{steps[arm]}.h5') as outputs:
                decomposition[arm]=aggregate_groups([offset_error_groups(source[sid]['target'][:],outputs['teacher/'+sid]['points'][:,:2],cfg['offset_stats']) for sid in data['training_ids']])
        decomposition['normalized_zero_baseline']=aggregate_groups([offset_error_groups(source[sid]['target'][:],None,cfg['offset_stats']) for sid in data['training_ids']])
    (folder/'offset-error-decomposition.json').write_text(json.dumps(decomposition,indent=2)+'\n')
    galleries=[]
    for split,ids in [('fixed_train8',data['fixed_train_ids']),('exposed_dev8',data['exposed_dev_ids'])]:
        for sid in ids:
            entries=[('source (reader-familiar real IAM)',targets[sid])]
            with h5py.File(p/'fixed'/f"evaluation-{steps['fixed']}.h5") as a,h5py.File(p/'adaptive'/f"evaluation-{steps['adaptive']}.h5") as b:
                if split=='fixed_train8':
                    for arm,hf in [('fixed',a),('adaptive',b)]:
                        entries.append((arm+' teacher-forced offsets / TRUE pen (NOT generation)',np.c_[hf['teacher/'+sid]['points'][:,:2],targets[sid][:,2:]]))
                        entries.append((arm+' teacher-forced / PREDICTED pen (NOT generation)',hf['teacher/'+sid]['points'][:]))
                for arm,hf in [('fixed',a),('adaptive',b)]:
                    for policy in evals[arm]['free'][split]:
                        group=hf['free/'+split+'/'+policy+'/'+sid];r=json.loads(group.attrs['row'])
                        entries.append((f'{arm} FREE {policy}; {r["generated_points"]}points; EOC={r["found_eoc"]}; reader={r["decoded"]!r}',group['points'][:]))
            parts=[]
            for j,(name,points) in enumerate(entries):
                filename=f'{split}-{sid}-{j}.svg';(folder/filename).write_text(svg(points));parts.append('<p>'+html.escape(name)+'</p><div class="strip"><img src="'+filename+'"></div>')
            gallery='<h2>'+html.escape(split+' '+sid+' | '+data['records'][sid]['text'])+'</h2>'+''.join(parts);galleries.append(gallery)
    # Independent checkpoint reload for selected/final weights can be expensive;
    # this report validates artifacts, not claiming numerical CPU/GPU parity.
    table=[]
    for arm in ARMS:
        ev=evals[arm];t=ev['teacher'];free=ev['free'];table.append('<tr><td>'+arm+'</td>'+''.join(f'<td>{v}</td>' for v in [steps[arm],f'{t["offset_mse"]:.6f}',f'{t["x_rmse"]:.4f}',f'{t["y_rmse"]:.4f}',f'{t["min_pen_f1"]:.3f}',f'{100*free["fixed_train8"]["correct"]["cer"]:.2f}%',f'{100*free["exposed_dev8"]["correct"]["cer"]:.2f}%',free['fixed_train8']['correct']['missing_eoc'],free['exposed_dev8']['correct']['missing_eoc']])+'</tr>')
    page='''<!doctype html><meta charset="utf-8"><title>Autoregressive stroke-feedback pilot</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}.strip{overflow:auto;background:white;border:1px solid #ddd;max-height:600px}.strip img{display:block;max-width:none}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style><h1>True autoregressive stroke-feedback pilot</h1><p><b>Teacher forcing is not generation.</b> Primary free-running outputs use only requested text/writer, own previous predicted8-point offset/pen blocks and one common2048point cap. First learnedEOC stops output; no source length/forced stop/reference/timing teacher at inference. Fixed and adaptive text clocks otherwise share fresh parameters/data/order/objective. This is a small deterministic recurrent alternative, NOT releasedInkDiT/semanticInkVAE or a reproduction of Graves.</p><p>Normalized chronological XY index displacements (including pen-up jumps), bounded focalpen; no KL/style/OCR training objective. MSE can average incompatible continuations and autoregression can accumulate error. EstablishedTRAIN256 and eight previously exposedDEV only; no new blind gate or unfamiliar-reader validation.</p><table><tr><th>Arm</th><th>Final/progress step</th><th>Teacher offsetMSE</th><th>Teacher X RMSE</th><th>Teacher Y RMSE</th><th>Teacher min penF1</th><th>Free TRAIN8 CER</th><th>Free exposedDEV8 CER</th><th>TRAIN missingEOC</th><th>DEV missingEOC</th></tr>'''+''.join(table)+'''</table><p>Teacher XY integrates predicted offsets from origin but recurrent history comes from truth; truepen rows isolate geometry. Free rows use only predicted states. Source and generated curves can legitimately have different lengths/spacing. Reader is corpus-familiar and CER must be inspected alongside these actual graphics.</p><p>All strips are marker-free SVG polylines, fixed100pixels per model coordinate unit with exact aspect ratio, own origin translation and up-positive modelY to down-positive SVG display conversion only. Actual one-point strokes are round-capped zero-length ink segments, not per-vertex markers. No smoothing/interpolation, target-boundary replacement on predicted rows, clipping or per-image width fitting. Scroll long/tall rows. A tall drifting output is deliberately not shrunk to look normal.</p><p>'''+('Both jobs terminal; completed logs and TRAIN-only checkpoint selection independently verified.' if final else 'INTERIM completed-step diagnostic; jobs may still be running. No model promotion or final outcome claimed.')+''' <a href="summary.json">Metrics/guards</a> · <a href="offset-error-decomposition.json">All256 offset-error topology</a></p>'''+''.join(galleries)
    (folder/'index.html').write_text(page)
    summary=dict(final=final,steps=steps,guards=guards,arms={a:dict(teacher=e['teacher'],free=e['free'],selected_step=results[a]['best_step'] if final else None) for a,e in evals.items()},
                 config_sha256=file_sha(p/'config.json'),dataset_sha256=file_sha(p/'dataset.json'),source_archive_sha256=cfg['source_archive_sha256'],no_new_confirmation=True,not_promoted=True,
                 scope='all256 teacher-forced rows and all fixedTRAIN8/exposedDEV8 free rows; exposedDEV never selects; teacher outputs are not generation; no numerical reload parity claim')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return str(folder)
