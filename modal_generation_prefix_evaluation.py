"""Guarded paired fresh positional contracts; two bounded T4 calls in parallel."""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
volume=modal.Volume.from_name('diffink-data')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'dataset'),'/app/dataset').add_local_dir(str(repo/'utils'),'/app/utils')
 .add_local_dir(str(repo/'trainer'),'/app/trainer').add_local_dir('iam_tools','/app/iam_tools').add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
app=modal.App('diffink-english-prefix-budget-evaluation')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=2,memory=8192,timeout=1800,retries=0,max_containers=1)
def evaluate(relative:str):
 from iam_tools.generation_prefix_budget import audit
 volume.reload()
 try:
  result=audit('/data/'+relative,'/app',root='/data',device='cuda')
  return {arm:s['groups'] for arm,s in result['arms'].items()}
 finally:volume.commit()

@app.local_entrypoint()
def main(run:bool=False,relative:str=''):
 if not run:
  print('No job allocated. --run --relative checkpoints/iam_generation_prefix_contract/<completed> performs frozen output-budget evaluation ONLY.');return
 from iam_tools.generation_prefix_contract_study import checked_path
 checked_path(relative,'/data')
 call=evaluate.spawn(relative);print(dict(call=call.object_id),flush=True)
 print(call.get(),flush=True)
