"""Explicit bounded T4 geometry/pen versus frozen-GRU OCR integration."""
from pathlib import Path
import modal
volume=modal.Volume.from_name('diffink-data')
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model')
 .add_local_dir(str(repo/'dataset'),'/app/dataset').add_local_dir(str(repo/'utils'),'/app/utils')
 .add_local_dir(str(repo/'trainer'),'/app/trainer').add_local_dir('iam_tools','/app/iam_tools')
 .add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
app=modal.App('diffink-english-ocr-joint')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=5400,retries=0,max_containers=1)
def research(steps:int,pool_sha:str):
 from iam_tools.ocr_joint_study import run
 try:return run('/app/configs/engineering_english.yaml','/app',steps=steps,pool_sha=pool_sha)
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=2400,retries=0,max_containers=1)
def render(relative:str):
 if not relative.startswith('checkpoints/iam_ocr_joint_study/') or '..' in Path(relative).parts:raise ValueError('bounded joint report required')
 from iam_tools.report_ocr_joint import report
 try:return report(Path('/data')/relative,'/app','/data')
 finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=200,pool_sha:str='',report_rel:str=''):
 if report_rel:print(render.remote(report_rel));return
 if not train:print('No GPU allocated. --train enables matched initialized-transport geometry/pen versus frozenGRU; KL/style off.');return
 if type(steps) is not int or not 50<=steps<=400:raise ValueError('50–400 bounded updates per arm required')
 from iam_tools.ocr_convergence import POOL_SHA
 if pool_sha!=POOL_SHA:raise ValueError('pin immutable8192TRAIN pool before GPU allocation')
 print(research.remote(steps,pool_sha))
