"""Offline, hash-verified pen-lift audit; never changes a running GPU protocol."""
import argparse,hashlib,html,json,shutil
from pathlib import Path
import h5py
from .pen_ab import file_sha
from .report_corpus_dit import verify_evaluation,_head,trajectory_section
from .stroke_diagnostics import compare,aggregate


def generate(relative,stage=1000,arms=None,root='data'):
    root=Path(root);p=root/relative;cfg=json.loads((p/'config.json').read_text());parent=root/cfg['parent_relative'] if 'parent_relative' in cfg else p;data=json.loads((parent/'dataset.json').read_text())
    if file_sha(parent/'dataset.json')!=cfg['dataset_sha256']:raise ValueError('immutable metadata mismatch')
    selected=list(arms) if arms else cfg.get('arms',['baseline'])
    if not selected or len(set(selected))!=len(selected) or any(a not in cfg.get('arms',['baseline']) for a in selected):raise ValueError('explicit declared arms required')
    stage=str(stage)
    if not (stage.isdigit() or stage in ('selected-confirmation','final-confirmation')):raise ValueError('saved explicit evaluation stage required')
    splits=dict(train=cfg['eval_train_ids'],dev_unseen_text=cfg['eval_dev_ids']) if stage.isdigit() else dict(exposed_held=data['scope']['splits']['exposed_held'],fresh_confirmation=data['scope']['splits']['fresh_confirmation'])
    controls=stage in ('0',str(cfg['max_updates']),'selected-confirmation','final-confirmation')
    out=p/f'stroke-audit-{stage}';out.mkdir(exist_ok=False);rows=[];sections=[];sourcepath=root/cfg['pool_relative']/'lines.h5'
    with h5py.File(sourcepath) as source:
        for arm in selected:
            ev=json.loads((p/arm/f'eval-{stage}.json').read_text());verify_evaluation(cfg,data,ev,splits,controls);path=p/arm/f'evaluation-{stage}.h5'
            if file_sha(path)!=ev['packed_h5_sha256']:raise ValueError('generated trajectory hash mismatch')
            with h5py.File(path) as hf:
                for row in ev['rows']:
                    sid=row['sample_id'];g=hf[f'{row["split"]}/{row["seed"]}/{row["guidance"]}/{row["policy"]}/{sid}'];q=g['points'][:];truth=source[sid]['point_seq'][:]
                    if json.loads(g.attrs['row'])!=row or len(q)!=row['generated_points']:raise ValueError('saved trajectory/row mismatch')
                    if hashlib.sha256(truth.tobytes()).hexdigest()!=data['records'][sid]['points_sha256']:raise ValueError('source point drift '+sid)
                    diagnostic=compare(q,truth,row['characters'],8*row['estimated_blocks'],row['found_eoc']);record=dict(arm=arm,stage=stage,sample_id=sid,split=row['split'],seed=row['seed'],guidance=row['guidance'],policy=row['policy'],**diagnostic);rows.append(record)
                    if row['policy']=='correct' and row['seed']==cfg['eval_noise_seeds'][0] and row['guidance']==1.:
                        gstats,sstats=diagnostic['generated'],diagnostic['source'];sections.append('<h2>'+html.escape(arm)+'</h2><p>'+f'Pen lifts: {gstats["pen_lifts"]} generated / {sstats["pen_lifts"]} source. Strokes: {gstats["rendered_strokes"]} / {sstats["rendered_strokes"]}. Points: {len(q)} / {len(truth)}. Severe under-lifting: {diagnostic["severe_underlifting"]}. Stop: {diagnostic["stop_stratum"]}.</p>'+trajectory_section(row,q))
    groups={}
    for r in rows:
        key=f'{r["arm"]}/{r["split"]}/g{r["guidance"]}/{r["policy"]}';groups.setdefault(key,[]).append(r)
    summary={k:aggregate(v) for k,v in groups.items()};result=dict(stage=stage,arms=selected,groups=summary,rows=rows,source_h5_relative=str(sourcepath.relative_to(root)),offline_only=True,training_protocol_unchanged=True,source_counts_not_used_for_generation=True,criterion='severe: median paired lift ratio per character AND per point <0.5; quality failure flag, generic writer may legitimately differ from source count',density_and_stop_caveats='Nonuniform RDP spacing; counts per point are NOT physical rates. Early means <80% estimated cap, not proved truncation. Counts may reflect wrong/shorter content as well as pen-state failure.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copyfile(__file__,out/'report-source.py')
    table=[]
    for k,v in summary.items():
        table.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [k,v['lines'],v['median_generated_pen_lifts'],v['median_source_pen_lifts'],v['median_generated_strokes'],v['median_source_strokes'],v['median_character_normalized_ratio'],v['median_point_normalized_ratio'],v['severe_underlifting_lines'],v['corpus_severe_underlifting']])+'</tr>')
    (out/'index.html').write_text(_head('Pen-lift and rendered-stroke audit')+'<h1>Pen-lift / rendered-stroke quality audit</h1><p>Offline analysis of saved generated points and same-prompt IAM sources. No training changes, no extra GPU, no target information at generation. A final EOC is not a pen-up and does not create an extra stroke. Singleton dots count as strokes. Point and character normalization prevent raw count alone from confusing short outputs with pen-class collapse. Generic writer styles need not match exact counts; severe corpus-relative deficits remain a quality failure flag.</p><p>Early-stop stratum = EOC before80% of TRAIN-estimated cap, not a known oracle length. Per-point rates are not physical velocity/time. All saved seeds/guidances/controls counted; primary seed/guidance1 marker-free galleries below. <a href="summary.json">Every line and exact criteria</a></p><table><tr><th>Arm/split/guidance/policy</th><th>Rows</th><th>Median gen lifts</th><th>Source lifts</th><th>Gen strokes</th><th>Source strokes</th><th>Char ratio</th><th>Point ratio</th><th>Severe lines</th><th>Corpus failure flag</th></tr>'+''.join(table)+'</table>'+''.join(sections))
    return str(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('relative');p.add_argument('--stage',default='1000');p.add_argument('--arms',nargs='+');p.add_argument('--root',default='data');a=p.parse_args();print(generate(a.relative,a.stage,a.arms,a.root))
