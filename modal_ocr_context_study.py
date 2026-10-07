"""Paired frozen-codec OCR representation/context controls; explicit GPU opt-in."""
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
app=modal.App('diffink-english-ocr-context-study')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=2400,retries=0,max_containers=1)
def research(steps:int):
    from iam_tools.ocr_context_study import run
    try:return run('/app/configs/engineering_english.yaml','/app',steps=steps)
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1)
def render(relative:str,annotate_only:bool=False):
    if not relative.startswith('checkpoints/iam_ocr_context_study/') or '..' in Path(relative).parts:raise ValueError('bounded context study report required')
    from iam_tools.report_ocr_context import report,annotate
    try:return annotate(Path('/data')/relative) if annotate_only else report(Path('/data')/relative,'/app','/data')
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=1000,report_rel:str='',annotate_only:bool=False):
    if report_rel:print(render.remote(report_rel,annotate_only));return
    if not train:print('No GPU allocated; --train enables the paired frozen-head context study.');return
    if not 1<=steps<=1500:raise ValueError('1–1500 updates per arm required')
    print(research.remote(steps))
