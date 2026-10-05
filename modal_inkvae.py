"""Prepared single-T4 mechanics job. NOT launched without both explicit flags."""
from pathlib import Path
import modal

app=modal.App('diffink-english-inkvae-overfit')
volume=modal.Volume.from_name('diffink-data')
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
       .add_local_dir(str(repo/'model'),'/app/model')
       .add_local_dir(str(repo/'dataset'),'/app/dataset')
       .add_local_dir(str(repo/'utils'),'/app/utils')
       .add_local_dir('iam_tools','/app/iam_tools')
       .add_local_file(str(repo/'configs/vae_iam_overfit.yaml'),'/app/configs/vae_iam_overfit.yaml')
       .workdir('/app'))

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=900,retries=0,max_containers=1)
def run_overfit():
    from iam_tools.inkvae import train_opt_in
    try:
        return train_opt_in('/app/configs/vae_iam_overfit.yaml','/app',allow_experimental=True)
    finally:
        volume.commit()


def require_opt_in(train, allow_experimental):
    if not train:return False
    if not allow_experimental:
        raise ValueError('Pass --allow-experimental to acknowledge normalization/input-scale/end-state policy')
    return True

@app.local_entrypoint()
def main(train: bool=False,allow_experimental: bool=False):
    # Refusal is before ANY .remote invocation; importing this module allocates no GPU.
    if not require_opt_in(train,allow_experimental):
        print('Training not launched. Prepared single-T4 config: at most 200 steps / 600 seconds.');return
    print(run_overfit.remote())
