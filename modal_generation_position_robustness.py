"""Guarded matched T4 nuisance-position experiment; no target smoothing."""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
volume=modal.Volume.from_name('diffink-data')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3').workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'dataset'),'/app/dataset').add_local_dir(str(repo/'utils'),'/app/utils').add_local_dir(str(repo/'trainer'),'/app/trainer').add_local_dir('iam_tools','/app/iam_tools').add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
app=modal.App('diffink-english-position-robustness')
@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=8192,timeout=7200,retries=0,max_containers=1)
def research():
 from iam_tools.generation_position_robustness_study import run
 try:return run('/app')
 finally:volume.commit()
@app.local_entrypoint()
def main(train:bool=False):
 if not train:print('No GPU allocated. --train runs matched8000update control/jitter.');return
 print(research.remote())
