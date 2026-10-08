"""Explicitly guarded bounded paired autoregressive T4 pilot; no new blind prompts."""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
volume=modal.Volume.from_name('diffink-data')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir('iam_tools','/app/iam_tools'))
app=modal.App('diffink-english-autoregressive-pilot')

@app.function(image=image,volumes={'/data':volume},cpu=2,memory=4096,timeout=600,retries=0,max_containers=1)
def prepare():
 from iam_tools.autoregressive_study import prepare as make
 volume.reload()
 try:return make('/app')
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=2,memory=8192,timeout=5400,retries=0,max_containers=2)
def research(arm:str,relative:str):
 from iam_tools.autoregressive_study import run
 volume.reload()
 try:return run(arm,relative,on_checkpoint=volume.commit)
 finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=0.125,memory=512,timeout=6600,retries=0,max_containers=1)
def coordinate():
 import json
 relative=prepare.remote();print(dict(study=relative),flush=True)
 calls=[research.spawn(a,relative) for a in ['fixed','adaptive']]
 results=[call.get() for call in calls]
 volume.reload();(Path('/data')/relative/'orchestration.json').write_text(json.dumps(dict(study=relative,results=results),indent=2)+'\n');volume.commit()
 return dict(study=relative,results=results)

@app.local_entrypoint()
def main(train:bool=False):
 if not train:
  print('No GPU allocated. Explicit --train launches paired fresh8000-update bounded fixed/adaptive autoregressive T4 pilots; no new blind sources.');return
 call=coordinate.spawn();print(dict(coordinator_call=call.object_id),flush=True);print(call.get(),flush=True)
