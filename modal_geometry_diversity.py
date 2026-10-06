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
app=modal.App('diffink-english-192line-geometry')
SOURCE='checkpoints/iam_fullset_joint/20261006-140945/checkpoint-best.pt'
SHA='89ec459de6496275a3712c08629daea10d8f4f03311712c77f49e652bca915a3'

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=1800,retries=0,max_containers=1)
def research(steps:int):
    from iam_tools.conditioning_study import run
    try:
        return run('/app/configs/engineering_english.yaml','/app',steps=steps,modes=('control',),
                   source_rel=SOURCE,source_sha=SHA,writer_id=None,family='iam_geometry_diversity')
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1)
def render(arm_rel:str):
    import json
    from iam_tools.conditioning_study import verify
    from iam_tools.report_writer_expansion import report
    from iam_tools.reference_retention import report as reference_report
    from iam_tools.report_pointer import publish_pointer
    path=Path(arm_rel)
    if not str(path).startswith('checkpoints/iam_geometry_diversity/') or '..' in path.parts:raise ValueError('diversity study directory required')
    p=Path('/data')/path
    try:
        check=verify(p,'/app');info=report(p,'/data');info['cpu_reload_check']=check
        (p/'report/summary.json').write_text(json.dumps(info,indent=2)+'\n');reference_report(p,'/data')
        publish_pointer(p.parent.parent,p/'report')
        return dict(output=str(p/'report/index.html'),selected_step=info['selected_step'],groups=info['final'])
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=1000,report_rel:str=''):
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated; explicit --train required.');return
    if not 1<=steps<=1000:raise ValueError('1–1000-update existing-data pilot only')
    result=research.remote(steps);print(result)
    arm=str(Path(result['output']).relative_to('/data')/'control')
    print(render.remote(arm))
