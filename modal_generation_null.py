"""Explicit bounded T4 matched text-dropout ablation; no promotion."""
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
app=modal.App('diffink-english-generation-null')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=5400,retries=0,max_containers=1)
def research(steps:int):
 from iam_tools.generation_null_study import run
 try:return run('/app/configs/engineering_english.yaml','/app',steps=steps)
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=3600,retries=0,max_containers=1)
def render(relative:str):
 if not relative.startswith('checkpoints/iam_generation_null/') or '..' in Path(relative).parts:raise ValueError('bounded study report path')
 from iam_tools.report_generation_path import report
 try:return report(Path('/data')/relative,'/app','/data')
 finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=4000,report_rel:str=''):
 if report_rel:print(render.remote(report_rel));return
 if not train:print('No GPU allocated. --train runs two matched bounded text-dropout arms.');return
 from iam_tools.generation_path_study import validate_budget
 validate_budget(steps)
 print(research.remote(steps))
