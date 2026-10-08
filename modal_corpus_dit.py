"""Guarded corpus-scale actual DiT pilot. Default allocates no GPU.

Original IAM/checkpoints read-only; durable prepared study and all outputs on v2.
One T4 worker, no speculative restart or duplicate pipeline after re-entry.
"""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
source=modal.Volume.from_name('diffink-data').with_mount_options(read_only=True)
output=modal.Volume.from_name('diffink-experiments-v2')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3','x-transformers==1.42.26','einops==0.8.2','einx==0.4.3','tqdm==4.67.1')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'utils'),'/app/utils').add_local_dir('iam_tools','/app/iam_tools'))
app=modal.App('diffink-english-corpus-dit')

def persist():
 import subprocess,time
 start=time.monotonic();print('PERSIST corpus v2 begin',flush=True);subprocess.run(['sync','/runs'],timeout=120,check=True);print(dict(persist_v2_seconds=time.monotonic()-start),flush=True)

def work_root(relative):
 import json
 from iam_tools.launch_ledger import study_path
 folder=study_path('/runs',relative,'checkpoints/iam_corpus_dit/');cfg=json.loads((folder/'config.json').read_text());root=Path('/work');root.mkdir(exist_ok=True)
 for name in [cfg['codec_relative'],cfg['reader_relative'],cfg['history_relative'],cfg['pool_relative']]:
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.symlink_to(Path('/data')/name)
 dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():dest.symlink_to(folder)
 return root,folder,cfg

@app.function(image=image,volumes={'/data':source,'/runs':output},cpu=2,memory=4096,timeout=600,retries=0,max_containers=1)
def preflight(relative:str):
 import json
 from iam_tools.corpus_dit_study import verify_sources
 from model.dit import DiT
 from utils.utils import ModelConfig
 root,folder,cfg=work_root(relative);data=json.loads((folder/'dataset.json').read_text());verify_sources(root,folder,cfg,data);model=DiT(ModelConfig(cfg['model']))
 return dict(study=relative,train_lines=len(data['scope']['splits']['train']),train_writers=len(data['training_writer_counts']),parameters=sum(p.numel() for p in model.parameters()),fresh_confirmation_not_encoded=True,no_gpu=True)

@app.function(image=image,volumes={'/data':source,'/runs':output},gpu='T4',cpu=2,memory=8192,timeout=7200,retries=0,max_containers=1)
def research(relative:str):
 import traceback
 from iam_tools.corpus_dit_study import run
 root,folder,cfg=work_root(relative)
 try:result=run(relative,root,on_checkpoint=persist)
 except BaseException:
  traceback.print_exc()
  try:persist()
  except BaseException:traceback.print_exc()
  raise
 persist();return result

@app.function(image=image,volumes={'/runs':output},cpu=.125,memory=512,timeout=7500,retries=0,max_containers=1,nonpreemptible=True)
def coordinate(relative:str):
 import json
 from iam_tools.launch_ledger import calls_once,study_path
 folder=study_path('/runs',relative,'checkpoints/iam_corpus_dit/');calls=calls_once(folder,['baseline'],lambda a:research.spawn(relative),modal.FunctionCall.from_id,persist);result=calls[0].get();(folder/'orchestration.json').write_text(json.dumps(result,indent=2)+'\n');persist();return result

@app.local_entrypoint()
def main(train:bool=False,study:str=''):
 if not train:print('No GPU allocated. --train --study runs one preprepared actual-DiT corpus pilot on a T4.');return
 if not study:raise ValueError('one explicit preprepared immutable corpus study required')
 print(preflight.remote(study),flush=True);call=coordinate.spawn(study);print(dict(study=study,coordinator_call=call.object_id),flush=True);print(call.get(),flush=True)
