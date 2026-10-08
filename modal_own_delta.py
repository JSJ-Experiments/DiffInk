"""Guarded matched own-history displacement control; all outputs on small v2.

The original data Volume is read-only. This is a new matched continuation, NOT
an interrupted-run recovery. Default invocation allocates no GPU.
"""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
source=modal.Volume.from_name('diffink-data').with_mount_options(read_only=True)
output=modal.Volume.from_name('diffink-experiments-v2')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'utils'),'/app/utils').add_local_dir('iam_tools','/app/iam_tools'))
app=modal.App('diffink-english-own-delta')

def persist():
 import subprocess,time
 start=time.monotonic();print('PERSIST v2 begin',flush=True)
 subprocess.run(['sync','/runs'],timeout=120,check=True)
 print(dict(persist_v2_seconds=time.monotonic()-start),flush=True)

def work_root(relative):
 import json
 from iam_tools.launch_ledger import study_path
 folder=study_path('/runs',relative,'checkpoints/iam_own_delta/')
 cfg=json.loads((folder/'config.json').read_text());root=Path('/work');root.mkdir(exist_ok=True)
 for name in [cfg['parent_checkpoint_relative'],cfg['source_data']+'/source.h5',cfg['reader_relative']]:
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.symlink_to(Path('/data')/name)
 dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():dest.symlink_to(folder)
 return root,folder,cfg

@app.function(image=image,volumes={'/data':source,'/runs':output},cpu=2,memory=4096,timeout=600,retries=0,max_containers=1)
def preflight(relative:str):
 import json,tarfile
 from iam_tools.own_delta_study import validate,PARENT_SHA
 from iam_tools.pen_ab import file_sha
 from iam_tools.generation_capacity import SOURCE_H5_SHA
 from iam_tools.ocr_joint_adapter import READER_SHA
 root,folder,cfg=work_root(relative);data=json.loads((folder/'dataset.json').read_text());validate(cfg,data)
 for p,sha in [(folder/'as-run-source.tar.gz',cfg['source_archive_sha256']),(root/cfg['parent_checkpoint_relative'],PARENT_SHA),(root/cfg['source_data']/'source.h5',SOURCE_H5_SHA),(root/cfg['reader_relative'],READER_SHA)]:
  if file_sha(p)!=sha:raise ValueError('prepared immutable source drift')
 with tarfile.open(folder/'as-run-source.tar.gz') as archive:
  for name in ['own_delta.py','own_delta_study.py','continuous_prefix.py','generated_prefix_study.py','point_feedback_strokes.py','history_rollin.py']:
   if archive.extractfile('iam_tools/'+name).read()!=(Path('/app/iam_tools')/name).read_bytes():raise ValueError('uploaded as-run implementation drift')
 return dict(study=relative,calibration=cfg['delta_calibration'],no_gpu=True,source_read_only=True)

@app.function(image=image,volumes={'/data':source,'/runs':output},gpu='T4',cpu=2,memory=8192,timeout=3600,retries=0,max_containers=3)
def research(arm:str,relative:str):
 import traceback
 from iam_tools.own_delta_study import run
 root,folder,cfg=work_root(relative)
 try:result=run(arm,relative,root=root,on_checkpoint=persist)
 except BaseException:
  traceback.print_exc()
  try:persist()
  except BaseException:traceback.print_exc()
  raise
 persist();return result

@app.function(image=image,volumes={'/runs':output},cpu=.125,memory=512,timeout=4200,retries=0,max_containers=1,nonpreemptible=True)
def coordinate(relative:str):
 import json
 from iam_tools.launch_ledger import calls_once,study_path
 folder=study_path('/runs',relative,'checkpoints/iam_own_delta/')
 calls=calls_once(folder,['control','ink_delta','all_delta'],lambda a:research.spawn(a,relative),modal.FunctionCall.from_id,persist)
 results=[c.get() for c in calls]
 (folder/'orchestration.json').write_text(json.dumps(dict(study=relative,results=results),indent=2)+'\n');persist();return dict(study=relative,results=results)

@app.local_entrypoint()
def main(train:bool=False,study:str=''):
 if not train:print('No GPU allocated. Explicit --train --study starts one preprepared matched three-arm control on v2.');return
 if not study:raise ValueError('one explicitly prepared immutable study required')
 print(preflight.remote(study),flush=True);call=coordinate.spawn(study);print(dict(study=study,coordinator_call=call.object_id),flush=True);print(call.get(),flush=True)
