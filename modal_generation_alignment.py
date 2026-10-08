"""Guarded fresh matched attention-alignment study on one T4; no promotion."""
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
app=modal.App('diffink-english-generation-alignment')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=7200,retries=0,max_containers=1)
def research(steps:int):
 from iam_tools.generation_alignment_study import run
 try:return run('/app/configs/engineering_english.yaml','/app',steps=steps)
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=3600,retries=0,max_containers=1)
def render(relative:str):
 if not relative.startswith(('checkpoints/iam_generation_alignment/','checkpoints/iam_generation_alignment_continuation/')) or '..' in Path(relative).parts:raise ValueError('bounded capacity report path')
 from iam_tools.report_generation_alignment import report
 try:return report(Path('/data')/relative,'/app','/data')
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=7200,retries=0,max_containers=1)
def continue_research(relative:str):
 from iam_tools.generation_alignment_continuation import run
 try:return run(relative,'/app')
 finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=24000,report_rel:str='',continue_rel:str=''):
 if continue_rel:
  if not train:print('No GPU allocated. Explicit --train --continue-rel required.');return
  print(continue_research.remote(continue_rel));return
 if report_rel:print(render.remote(report_rel));return
 if not train:print('No GPU allocated. --train runs fresh matched attention-alignment study.');return
 if type(steps) is not int or not 16000<=steps<=32000:raise ValueError('bounded16000–32000 steps required')
 print(research.remote(steps))
