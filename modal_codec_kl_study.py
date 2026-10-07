"""Opt-in bounded matched KL continuations; source geometry is NOT reset."""
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
app=modal.App('diffink-english-codec-kl-study')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=2400,retries=0,max_containers=1)
def research(steps:int,protected_pen:bool=False):
    from iam_tools.codec_kl_study import run
    from iam_tools.metric_workers import metric_pool
    try:
        with metric_pool(3) as pool:return run('/app/configs/engineering_english.yaml','/app',steps=steps,metric_pool=pool,protected_pen=protected_pen)
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1)
def render(relative:str):
    if not relative.startswith('checkpoints/iam_codec_kl_study/') or '..' in Path(relative).parts:raise ValueError('bounded codec report required')
    from iam_tools.report_codec_kl import report
    try:return report(Path('/data')/relative,'/data')
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=600,retries=0,max_containers=1)
def audit(relative:str):
    if not relative.startswith('checkpoints/iam_codec_kl_study/') or '..' in Path(relative).parts:raise ValueError('bounded codec audit required')
    from iam_tools.codec_kl_study import tail_audit
    try:return tail_audit(Path('/data')/relative,'/app','/data')
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=200,report_rel:str='',tails_rel:str='',protected_pen:bool=False):
    if tails_rel:print(audit.remote(tails_rel));return
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated. Use --train for matched KL0/1e-6/1e-5 continuations.');return
    if not 1<=steps<=300:raise ValueError('1–300 continuation updates per arm required')
    print(research.remote(steps,protected_pen))
