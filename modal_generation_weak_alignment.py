"""Guarded paired TRAIN-only weak character alignment; two bounded T4 calls in parallel."""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
volume=modal.Volume.from_name('diffink-data')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'dataset'),'/app/dataset').add_local_dir(str(repo/'utils'),'/app/utils')
 .add_local_dir(str(repo/'trainer'),'/app/trainer').add_local_dir('iam_tools','/app/iam_tools').add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
app=modal.App('diffink-english-weak-alignment')

@app.function(image=image,volumes={'/data':volume},cpu=2,memory=8192,timeout=600,retries=0,max_containers=1)
def prepare():
 from iam_tools.generation_weak_alignment_study import prepare as make
 volume.reload()
 try:return make('/app')
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=2,memory=8192,timeout=7200,retries=0,max_containers=2)
def research(arm:str,relative:str):
 from iam_tools.generation_weak_alignment_study import run
 volume.reload()
 try:return run(arm,relative,'/app')
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=0.125,memory=512,timeout=9000,retries=0,max_containers=1)
def coordinate():
 # Keep BOTH GPU calls under one durable remote input. A detached local
 # entrypoint can retain only its last call when it disconnects mid-spawn.
 relative=prepare.remote();print(dict(study=relative),flush=True)
 calls=[research.spawn(a,relative) for a in ['control','weak_alignment']]
 results=[call.get() for call in calls]
 volume.reload()
 import json
 from pathlib import Path
 (Path('/data')/relative/'orchestration.json').write_text(json.dumps(dict(study=relative,results=results,policy='detached remote coordinator retains BOTH spawned GPU calls'),indent=2)+'\n')
 volume.commit()
 return dict(study=relative,results=results)

@app.local_entrypoint()
def main(train:bool=False):
 if not train:
  print('No job allocated. Use modal run --detach modal_generation_weak_alignment.py --train for two matched48000-update causal T4 arms (TRAIN weak attention auxiliary only).');return
 call=coordinate.spawn();print(dict(coordinator_call=call.object_id),flush=True)
 print(call.get(),flush=True)
