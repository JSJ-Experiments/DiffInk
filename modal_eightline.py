"""Opt-in bounded T4 engineering test, through production VAE trainer."""
from pathlib import Path
import modal

app=modal.App('diffink-english-eightline')
volume=modal.Volume.from_name('diffink-data')
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
       .workdir('/app')
       .add_local_dir(str(repo/'model'),'/app/model')
       .add_local_dir(str(repo/'dataset'),'/app/dataset')
       .add_local_dir(str(repo/'utils'),'/app/utils')
       .add_local_dir(str(repo/'trainer'),'/app/trainer')
       .add_local_dir('iam_tools','/app/iam_tools')
       .add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=900,retries=0,max_containers=1)
def run_eightline():
    from iam_tools.eightline import train
    try:
        result=train('/app/configs/engineering_english.yaml','/app',allow_experimental=True)
        return {k:v for k,v in result.items() if k not in ('initial','final')}
    finally:volume.commit()


def require_opt_in(train,allow_experimental):
    if not train:return False
    if not allow_experimental:raise ValueError('explicit experimental acknowledgement required')
    return True

@app.local_entrypoint()
def main(train:bool=False,allow_experimental:bool=False):
    if not require_opt_in(train,allow_experimental):
        print('Not launched. One T4, max 200 optimizer updates / 1600 physical microbatches / 600 loop seconds.');return
    print(run_eightline.remote())
