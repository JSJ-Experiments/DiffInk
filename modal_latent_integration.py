"""Guarded T4 posterior/KL/OCR integration; no GPU allocated without --train."""
from pathlib import Path
import modal

app = modal.App('diffink-english-latent-integration')
volume = modal.Volume.from_name('diffink-data')
repo = Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.14.1', 'numpy==2.5.3', 'h5py==3.16.0', 'Pillow==12.3.0', 'matplotlib==3.11.2', 'PyYAML==6.0.3')
         .workdir('/app')
         .add_local_dir(str(repo/'model'), '/app/model')
         .add_local_dir(str(repo/'dataset'), '/app/dataset')
         .add_local_dir(str(repo/'utils'), '/app/utils')
         .add_local_dir(str(repo/'trainer'), '/app/trainer')
         .add_local_dir('iam_tools', '/app/iam_tools')
         .add_local_file(str(repo/'configs/engineering_english.yaml'), '/app/configs/engineering_english.yaml'))


@app.function(image=image, volumes={'/data': volume}, gpu='T4', cpu=4, memory=16384,
              timeout=3600, retries=0, max_containers=1)
def research(steps: int, lr: float, stages: str):
    from iam_tools.latent_integration import run
    try:
        return run('/app/configs/engineering_english.yaml', '/app', steps=steps, lr=lr, stages=tuple(stages.split(',')))
    finally:
        volume.commit()


@app.local_entrypoint()
def main(train: bool = False, steps: int = 400, lr: float = 5e-7, stages: str = 'sampled,kl,ocr'):
    if not train:
        print('No GPU allocated. Explicit --train required.'); return
    if not 1 <= steps <= 1000 or not 0 < lr <= 1e-5 or any(s not in ('sampled','kl','ocr') for s in stages.split(',')):
        raise ValueError('bounded integration requires known stages/steps/LR')
    print(research.remote(steps, lr, stages))
