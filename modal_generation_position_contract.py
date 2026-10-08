"""Guarded paired fresh positional contracts; two bounded T4 calls in parallel."""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
volume=modal.Volume.from_name('diffink-data')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'dataset'),'/app/dataset').add_local_dir(str(repo/'utils'),'/app/utils')
 .add_local_dir(str(repo/'trainer'),'/app/trainer').add_local_dir('iam_tools','/app/iam_tools').add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
app=modal.App('diffink-english-position-contract')

@app.function(image=image,volumes={'/data':volume},cpu=2,memory=8192,timeout=600,retries=0,max_containers=1)
def prepare():
 from iam_tools.generation_position_contract_study import prepare as make
 volume.reload()
 try:return make('/app')
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=2,memory=8192,timeout=5400,retries=0,max_containers=2)
def research(arm:str,relative:str):
 from iam_tools.generation_position_contract_study import run
 volume.reload()
 try:return run(arm,relative,'/app')
 finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False):
 if not train:
  print('No job allocated. --train runs two matched48000-update T4 arms in parallel.');return
 relative=prepare.remote();print(dict(study=relative),flush=True)
 calls=[research.spawn(a,relative) for a in ['relative100','absolute']]
 for call in calls:print(call.get(),flush=True)
