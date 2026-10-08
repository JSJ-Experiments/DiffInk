"""Guarded corpus factorial, at most two simultaneous T4s, original read-only.

Default allocates no GPU. Four persisted arm IDs; no speculative resubmission.
"""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
source=modal.Volume.from_name('diffink-data').with_mount_options(read_only=True)
output=modal.Volume.from_name('diffink-experiments-v2')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3','x-transformers==1.42.26','einops==0.8.2','einx==0.4.3','tqdm==4.67.1')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'utils'),'/app/utils').add_local_dir('iam_tools','/app/iam_tools'))
app=modal.App('diffink-english-compact-dit')

def persist():
 import subprocess
 subprocess.run(['sync','/runs'],timeout=120,check=True)

def work_root(relative):
 import json
 from iam_tools.launch_ledger import study_path
 folder=study_path('/runs',relative,'checkpoints/iam_compact_dit/');cfg=json.loads((folder/'config.json').read_text());parent=Path('/runs')/cfg['parent_relative'];pc=json.loads((parent/'config.json').read_text());root=Path('/work');root.mkdir(exist_ok=True)
 for name in [pc['codec_relative'],pc['reader_relative'],pc['history_relative'],pc['pool_relative']]:
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.symlink_to(Path('/data')/name)
 for name in [relative,cfg['parent_relative']]:
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.symlink_to(Path('/runs')/name)
 return root,folder,cfg

@app.function(image=image,volumes={'/data':source,'/runs':output},cpu=2,memory=4096,timeout=600,retries=0,max_containers=1)
def preflight(relative:str):
 from iam_tools.compact_dit_study import verify
 from iam_tools.compact_dit import ARMS
 from model.dit import DiT
 from utils.utils import ModelConfig
 root,folder,cfg=work_root(relative);parent,pc,data=verify(root,folder,cfg)
 counts={a:sum(p.numel() for p in DiT(ModelConfig(dict(cfg['model'],latent_dim=c))).parameters()) for a,(c,pred) in ARMS.items()}
 return dict(study=relative,train_lines=len(data['scope']['splits']['train']),train_writers=len(data['training_writer_counts']),parameters=counts,at_most_two_T4s=True,no_confirmations_opened=True,no_gpu=True)

@app.function(image=image,volumes={'/data':source,'/runs':output},gpu='T4',cpu=2,memory=8192,timeout=5000,retries=0,max_containers=2)
def research(relative:str,arm:str):
 import traceback
 from iam_tools.compact_dit_study import run
 root,folder,cfg=work_root(relative)
 try:result=run(relative,arm,root,on_checkpoint=persist)
 except BaseException:
  traceback.print_exc()
  try:persist()
  except BaseException:traceback.print_exc()
  raise
 persist();return result

@app.function(image=image,volumes={'/runs':output},cpu=.125,memory=512,timeout=11000,retries=0,max_containers=1,nonpreemptible=True)
def coordinate(relative:str):
 import json
 from iam_tools.launch_ledger import calls_once,study_path
 folder=study_path('/runs',relative,'checkpoints/iam_compact_dit/');cfg=json.loads((folder/'config.json').read_text());calls=calls_once(folder,cfg['arms'],lambda a:research.spawn(relative,a),modal.FunctionCall.from_id,persist);result={a:c.get() for a,c in zip(cfg['arms'],calls)};(folder/'orchestration.json').write_text(json.dumps(result,indent=2)+'\n');persist();return result

@app.local_entrypoint()
def main(train:bool=False,study:str=''):
 if not train:print('No GPU allocated. --train --study launches one preprepared four-arm factorial, at most two T4s.');return
 if not study:raise ValueError('explicit preprepared immutable study required')
 print(preflight.remote(study),flush=True);call=coordinate.spawn(study);print(dict(study=study,coordinator_call=call.object_id),flush=True);print(call.get(),flush=True)
