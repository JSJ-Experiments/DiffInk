"""Paired4-point OCR local context residual; no GPU without explicit flags."""
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
app=modal.App('diffink-english-ocr-local-context')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=4800,retries=0,max_containers=1)
def research(steps:int,pool_sha:str):
    from iam_tools.ocr_local_study import run
    try:return run('/app/configs/engineering_english.yaml','/app',steps=steps,pool_sha=pool_sha)
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1)
def render(relative:str):
    if not relative.startswith('checkpoints/iam_ocr_local_context/') or '..' in Path(relative).parts:raise ValueError('bounded local study report required')
    from iam_tools.report_ocr_local import report
    try:return report(Path('/data')/relative,'/app','/data')
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=6000,pool_sha:str='',report_rel:str=''):
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated. --train enables paired4-point OCR local context; codec stays frozen.');return
    if not 1000<=steps<=6000:raise ValueError('1000–6000 additionalupdates required')
    from iam_tools.ocr_convergence import POOL_SHA
    if pool_sha!=POOL_SHA:raise ValueError('pin fixed8192 pool before allocation')
    print(research.remote(steps,pool_sha))
