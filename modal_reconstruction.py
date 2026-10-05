"""Guarded T4 joint-compatibility + independent deterministic-XY diagnostics."""
from pathlib import Path
import modal

app = modal.App('diffink-english-reconstruction-diagnostics')
volume = modal.Volume.from_name('diffink-data')
repo = Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.14.1', 'numpy==2.5.3', 'h5py==3.16.0', 'Pillow==12.3.0', 'matplotlib==3.11.2', 'PyYAML==6.0.3')
         .workdir('/app')
         .add_local_dir(str(repo / 'model'), '/app/model')
         .add_local_dir(str(repo / 'dataset'), '/app/dataset')
         .add_local_dir(str(repo / 'utils'), '/app/utils')
         .add_local_dir('iam_tools', '/app/iam_tools')
         .add_local_file(str(repo / 'configs/vae_iam_autopsy_resume.yaml'), '/app/configs/vae_iam_autopsy_resume.yaml')
         .add_local_file(str(repo / 'configs/vae_iam_reconstruction.yaml'), '/app/configs/vae_iam_reconstruction.yaml'))


@app.function(image=image, volumes={'/data': volume}, gpu='T4', cpu=4, memory=16384,
              timeout=900, retries=0, max_containers=1)
def run_diagnostics(mse_only: bool = False):
    from iam_tools.reconstruction import train_pair
    try:
        return train_pair('/app/configs/vae_iam_reconstruction.yaml', '/app', allow_experimental=True, mse_only=mse_only)
    finally:
        volume.commit()


def require_opt_in(train, allow_experimental):
    if not train:
        return False
    if not allow_experimental:
        raise ValueError('Pass --allow-experimental to acknowledge experimental IAM representation and deterministic MSE diagnostic')
    return True


@app.local_entrypoint()
def main(train: bool = False, allow_experimental: bool = False, mse_only: bool = False):
    if not require_opt_in(train, allow_experimental):
        print('Not launched. One T4: up to 300 joint + 1000 independent MSE updates / 600 combined loop seconds.')
        return
    print(run_diagnostics.remote(mse_only))
