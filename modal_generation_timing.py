"""CPU-only frozen timing autopsy. No GPU or training allocation."""
from pathlib import Path
import modal
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
volume=modal.Volume.from_name('diffink-data')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3').workdir('/app').add_local_dir(str(repo/'model'),'/app/model').add_local_dir(str(repo/'dataset'),'/app/dataset').add_local_dir(str(repo/'utils'),'/app/utils').add_local_dir(str(repo/'trainer'),'/app/trainer').add_local_dir('iam_tools','/app/iam_tools').add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
app=modal.App('diffink-english-timing-autopsy')
@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=3600,retries=0,max_containers=1)
def research():
 from iam_tools.generation_timing_study import run
 try:return run('/app')
 finally:volume.commit()
@app.local_entrypoint()
def main(probe:bool=False):
 if not probe:print('No job allocated. --probe runs CPU-only frozen timing diagnosis.');return
 print(research.remote())
