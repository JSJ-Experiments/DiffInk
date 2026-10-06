"""Explicitly guarded matched T4 reconstruction-conditioning study."""
from pathlib import Path
import os
import modal
volume = modal.Volume.from_name('diffink-data')
repo = Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.14.1', 'numpy==2.5.3', 'h5py==3.16.0', 'Pillow==12.3.0', 'matplotlib==3.11.2', 'PyYAML==6.0.3')
         .workdir('/app')
         .add_local_dir(str(repo/'model'), '/app/model')
         .add_local_dir(str(repo/'dataset'), '/app/dataset')
         .add_local_dir(str(repo/'utils'), '/app/utils')
         .add_local_dir(str(repo/'trainer'), '/app/trainer')
         .add_local_dir('iam_tools', '/app/iam_tools')
         .add_local_file(str(repo/'configs/engineering_english.yaml'), '/app/configs/engineering_english.yaml'))
EXPERIMENT_NAME=os.environ.get('DIFFINK_EXPERIMENT_NAME','diffink-english-fullset-joint')
app=modal.App(EXPERIMENT_NAME)

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=3000,retries=0,max_containers=1)
def research(steps:int,relative_fraction:float,source_rel:str,source_sha:str,experiment_name:str):
    from iam_tools.fullset_joint import run
    try:
        kwargs=dict(source_rel=source_rel,source_sha=source_sha) if source_rel else {}
        return run('/app/configs/engineering_english.yaml','/app',steps=steps,relative_fraction=relative_fraction,experiment_name=experiment_name,**kwargs)
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=900,retries=0,max_containers=1)
def render(report_rel:str):
    import json
    from iam_tools.verify_writer_checkpoint import verify
    from iam_tools.report_writer_expansion import report
    from iam_tools.reference_retention import report as reference_report
    if not report_rel.startswith('checkpoints/iam_fullset_joint/') or '..' in Path(report_rel).parts:raise ValueError('study directory required')
    p=Path('/data')/report_rel
    try:
        cfg=json.loads((p/'config.json').read_text())
        if cfg.get('conditioning_mode','control')!='control':
            from iam_tools.conditioning_study import verify as verify_conditioned
            check=verify_conditioned(p,'/app')
        else:check=verify(p,'/app')
        info=report(p,'/data');info['cpu_reload_check']=check
        (p/'report/summary.json').write_text(json.dumps(info,indent=2)+'\n');reference_report(p,'/data')
        from iam_tools.report_pointer import publish_pointer
        publish_pointer(p.parent,p/'report')
        return dict(output=str(p/'report'),final=info['final'])
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=240,report_rel:str='',relative_fraction:float=0.,source_rel:str='',source_sha:str='',comparison_json:str=''):
    if comparison_json:print(compare.remote(comparison_json));return
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated; explicit --train required.');return
    if not 1<=steps<=300 or not 0<=relative_fraction<=.5 or bool(source_rel)!=bool(source_sha):raise ValueError('bounded full-set diagnostic required')
    print(research.remote(steps,relative_fraction,source_rel,source_sha,EXPERIMENT_NAME))

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=900,retries=0,max_containers=1)
def compare(entries_json:str):
    import json
    from iam_tools.report_fullset_comparison import report
    entries=json.loads(entries_json)
    for entry in entries:
        path=Path(entry['directory'])
        if not str(path).startswith('checkpoints/iam_') or '..' in path.parts:raise ValueError('saved IAM research directory required')
    try:
        stages=report(entries,root='/data')
        return dict(output='/data/checkpoints/iam_fullset_joint/research-summary/index.html',stages=[dict(label=s['label'],step=s['selected_step']) for s in stages])
    finally:volume.commit()
