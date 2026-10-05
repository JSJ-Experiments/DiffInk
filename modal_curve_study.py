"""Guarded T4 curve-fidelity research; no GPU allocated without --train."""
from pathlib import Path
import modal

app = modal.App('diffink-english-curve-study-v2')
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
              timeout=3000, retries=0, max_containers=1)
def research(steps: int, arms: str, source_path: str = '', source_sha: str = '', deterministic: bool = False):
    from iam_tools.curve_study import run, SOURCE_REL, SOURCE_SHA
    try:
        return run('/app/configs/engineering_english.yaml', '/app', steps=steps, arms=tuple(arms.split(',')),
                   source_rel=source_path.removeprefix('/data/') if source_path else SOURCE_REL,
                   source_sha=source_sha or SOURCE_SHA, deterministic=deterministic)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(train: bool = False, steps: int = 80, arms: str = 'point,delta20', source_path: str = '', source_sha: str = '', deterministic: bool = False):
    if not train:
        print('No GPU allocated. Explicit --train required.'); return
    if not 1 <= steps <= 100 or any(a not in ('point', 'delta20', 'delta50', 'tangent20') for a in arms.split(',')):
        raise ValueError('bounded study requires 1..100 steps and known arms')
    if bool(source_path) != bool(source_sha): raise ValueError('source path and SHA required together')
    print(research.remote(steps, arms, source_path, source_sha, deterministic))
