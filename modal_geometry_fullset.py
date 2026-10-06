"""Bounded T4 utilization measurement. No optimizer or checkpoint writes."""
"""Bounded existing-data geometry pilot; no OCR/KL/style or full-IAM training."""
from pathlib import Path
import modal

volume=modal.Volume.from_name('diffink-data')
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
       .workdir('/app')
       .add_local_dir(str(repo/'model'),'/app/model')
       .add_local_dir(str(repo/'dataset'),'/app/dataset')
       .add_local_dir(str(repo/'utils'),'/app/utils')
       .add_local_dir(str(repo/'trainer'),'/app/trainer')
       .add_local_dir('iam_tools','/app/iam_tools')
       .add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
SOURCE='checkpoints/iam_geometry_diversity/20261006-144049/control/checkpoint-best.pt'
SHA='ce4512fd4d17c643d29ac903a092452f36810db3bea852d0eeff2a658ea7b00b'

app=modal.App('diffink-english-192line-fullgradient-geometry')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,
              timeout=1800,retries=0,max_containers=1)
def research(steps:int):
    from iam_tools.fullset_joint import run
    from iam_tools.metric_workers import metric_pool
    try:
        with metric_pool(3) as pool:
            result=run('/app/configs/engineering_english.yaml','/app',steps=steps,
                       source_rel=SOURCE,source_sha=SHA,writer_id=None,accelerated=True,
                       metric_pool=pool,max_wall_seconds=1200,eval_every=20,
                       pen_weight=.02099049935353879,family='iam_geometry_fullset',
                       experiment_name='diffink-english-192line-fullgradient-geometry')
            return dict(output=result['output'],best_step=result['best_step'],last_step=result['last_step'],
                        closure_calls=result['closure_calls'],stop_reason=result['stop_reason'])
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1)
def render(relative:str):
    import json
    from iam_tools.verify_writer_checkpoint import verify
    from iam_tools.report_writer_expansion import report
    from iam_tools.reference_retention import report as reference_report
    from iam_tools.report_pointer import publish_pointer
    if not relative.startswith('checkpoints/iam_geometry_fullset/') or '..' in Path(relative).parts:raise ValueError('bounded study report required')
    directory=Path('/data')/relative
    try:
        check=verify(directory,'/app');info=report(directory,'/data');info['cpu_reload_check']=check
        (directory/'report/summary.json').write_text(json.dumps(info,indent=2)+'\n')
        reference_report(directory,'/data');publish_pointer(directory.parent,directory/'report')
        return dict(output=str(directory/'report/index.html'),selected_step=info['selected_step'],final=info['final'])
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=40,report_rel:str=''):
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated; --train required.');return
    if not 1<=steps<=40:raise ValueError('1–40 outer steps; 1200s loop budget required')
    result=research.remote(steps);print(result)
    print(render.remote(str(Path(result['output']).relative_to('/data'))))
