"""Explicit interrupted-control recovery on an isolated small Volume v2.

Original diffink-data is input-only. No fresh optimization from parent is allowed:
resume exact captured checkpoints, AdamW and RNG; preserve already executed order.
"""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
source=modal.Volume.from_name('diffink-data')
output=modal.Volume.from_name('diffink-experiments-v2')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'utils'),'/app/utils').add_local_dir('iam_tools','/app/iam_tools'))
app=modal.App('diffink-english-continuous-recovery')

def persist():
 import subprocess,time
 start=time.monotonic();print('PERSIST v2 begin',flush=True)
 subprocess.run(['sync','/runs'],timeout=120,check=True)
 print(dict(persist_v2_seconds=time.monotonic()-start),flush=True)

def work_root(relative):
 import json
 from iam_tools.launch_ledger import study_path
 folder=study_path('/runs',relative,'checkpoints/iam_continuous_prefix/')
 cfg=json.loads((folder/'config.json').read_text());root=Path('/work');root.mkdir(exist_ok=True)
 for name in [cfg['parent_checkpoint_relative'],cfg['source_data']+'/source.h5',cfg['reader_relative']]:
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.symlink_to(Path('/data')/name)
 dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():dest.symlink_to(folder)
 return root,folder,cfg

@app.function(image=image,volumes={'/data':source,'/runs':output},cpu=2,memory=4096,timeout=600,retries=0,max_containers=1)
def preflight(relative:str):
 import json,torch
 from iam_tools.continuous_prefix_study import validate,ARMS
 from iam_tools.continuous_recovery import snapshot_info,validate_checkpoint
 root,folder,cfg=work_root(relative);data=json.loads((folder/'dataset.json').read_text());parent=json.loads((folder/'parent-eval.json').read_text());validate(cfg,data)
 for arm in ARMS:
  path=folder/'resume-source'/arm;info=snapshot_info(path,cfg,data,arm,parent);saved=torch.load(path/'checkpoint-last.pt',map_location='cpu',weights_only=False);validate_checkpoint(saved,cfg,arm,info['step'])
 return dict(study=relative,recovery_steps={a:cfg['recovery']['arms'][a]['step'] for a in ARMS},no_gpu=True)

@app.function(image=image,volumes={'/data':source,'/runs':output},gpu='T4',cpu=2,memory=8192,timeout=3600,retries=0,max_containers=2)
def research(arm:str,relative:str):
 import traceback
 from iam_tools.continuous_prefix_study import run
 root,folder,cfg=work_root(relative)
 try:
  result=run(arm,relative,root=root,on_checkpoint=persist,resume_source=folder/'resume-source'/arm)
 except BaseException:
  traceback.print_exc() # Do not let a secondary persistence failure hide training errors.
  try:persist()
  except BaseException:traceback.print_exc()
  raise
 persist();return result

@app.function(image=image,volumes={'/runs':output},cpu=.125,memory=512,timeout=4200,retries=0,max_containers=1,nonpreemptible=True)
def coordinate(relative:str):
 import json
 from iam_tools.launch_ledger import calls_once,study_path
 folder=study_path('/runs',relative,'checkpoints/iam_continuous_prefix/')
 calls=calls_once(folder,['detached_xy','continuous_xy'],lambda a:research.spawn(a,relative),modal.FunctionCall.from_id,persist)
 results=[c.get() for c in calls]
 (folder/'orchestration.json').write_text(json.dumps(dict(study=relative,results=results),indent=2)+'\n');persist();return dict(study=relative,results=results)

@app.local_entrypoint()
def main(train:bool=False,study:str=''):
 if not train:print('No GPU allocated. Explicit --train --study resumes already captured paired controls on isolated v2 storage.');return
 if not study:raise ValueError('one explicitly prepared immutable recovery study required')
 print(preflight.remote(study),flush=True);call=coordinate.spawn(study);print(dict(study=study,coordinator_call=call.object_id),flush=True);print(call.get(),flush=True)
