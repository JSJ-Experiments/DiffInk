"""Resume a captured interrupted control, never silently restart or duplicate updates."""
import json,shutil
from pathlib import Path
from .pen_ab import file_sha
from .writer_expansion import training_schedule
from .history_rollin_study import teacher_gate


def validate_checkpoint(saved,cfg,arm,step):
    if saved['arm']!=arm or saved['step']!=step or step not in cfg['eval_steps'] or not 0<step<cfg['max_updates']:
        raise ValueError('exact declared interrupted arm/evaluation step required')
    ignored={'source_archive_sha256','recovery'}
    expected={k:v for k,v in cfg.items() if k not in ignored}
    actual={k:v for k,v in saved['config'].items() if k not in ignored}
    if actual!=expected or saved['parent_body_step']!=4000 or saved['parent_head_updates']!=6000:
        raise ValueError('same original scalar objective/data/order/parent config required')
    if not saved['optimizer_state_dict']['state'] or not all(k in saved for k in ('model_state_dict','rng_cpu','rng_cuda')):
        raise ValueError('saved model, nonempty optimizer AND RNG states required')


def snapshot_info(source,cfg,data,arm,parent):
    source=Path(source);declared=cfg['recovery']['arms'][arm];step=declared['step']
    if file_sha(source/'checkpoint-last.pt')!=declared['last_sha256'] or file_sha(source/'checkpoint-best.pt')!=declared['best_sha256']:
        raise ValueError('byte-exact captured last AND best checkpoints required')
    rows=[json.loads(s) for s in (source/'metrics.jsonl').read_text().splitlines()]
    schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],cfg['batch'],cfg['schedule_seed']))
    if [r['step'] for r in rows]!=list(range(1,step+1)) or [r['sample_ids'] for r in rows]!=schedule[:step]:
        raise ValueError('complete prefix with no missing/duplicate/reordered updates required')
    history=[];best=None;best_step=None
    for s in cfg['eval_steps']:
        if s>step:continue
        ev=json.loads((source/f'eval-{s}.json').read_text());reading=json.loads((source/f'teacher-reading-{s}.json').read_text());gate=teacher_gate(ev,reading,parent)
        if file_sha(source/f'evaluation-{s}.h5')!=ev['packed_h5_sha256'] or gate!=ev['capacity_gate'] or gate!=reading['capacity_gate']:
            raise ValueError('captured checkpoint outputs and original teacher gates required')
        history.append(dict(step=s,teacher=ev['teacher'],free=ev['free'],source_rollout=ev['source_rollout'],reading=reading,capacity_gate=gate))
        if gate['passed']:
            score=(ev['free']['correct']['cer'],ev['teacher']['offset_mse'])
            if best is None or score<best:best=score;best_step=s
    if best is None:raise ValueError('eligible preserved baseline required')
    summary=json.loads((source/'resource-summary.json').read_text());seconds=summary['phases']['train/'+arm]['measured_step_seconds']
    if not 0<=seconds<cfg['max_train_wall_seconds']:raise ValueError('bounded actual prior TRAIN time required')
    return dict(step=step,rows=rows,history=history,best=best,best_step=best_step,seconds=seconds,clipped=sum(r['clipped'] for r in rows))


def copy_snapshot(source,destination):
    """Copy immutable evidence into new exclusive output; keep telemetry separately."""
    source=Path(source);destination=Path(destination)
    for f in source.iterdir():
        if f.is_file() and f.name not in ('resources.jsonl','resource-summary.json','alerts.jsonl'):
            if (destination/f.name).exists():raise ValueError('refusing recovery artifact overwrite')
            shutil.copyfile(f,destination/f.name)
