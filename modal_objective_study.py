"""Explicit opt-in T4 objective research; old 200-update guard remains intact."""
from pathlib import Path
import modal

app = modal.App('diffink-english-objective-study')
volume = modal.Volume.from_name('diffink-data')
repo = Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
         .workdir('/app')
         .add_local_dir(str(repo/'model'),'/app/model')
         .add_local_dir(str(repo/'dataset'),'/app/dataset')
         .add_local_dir(str(repo/'utils'),'/app/utils')
         .add_local_dir(str(repo/'trainer'),'/app/trainer')
         .add_local_dir('iam_tools','/app/iam_tools')
         .add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))

@app.function(image=image, volumes={'/data':volume}, gpu='T4', cpu=4, memory=16384,
              timeout=3000, retries=0, max_containers=1)
def research(arm: str, steps: int = 1000, resume_checkpoint: str = '', resume_sha: str = ''):
    from iam_tools.objective_study import run
    try:
        if arm == 'lbfgs':
            from iam_tools.lbfgs_geometry import run as lbfgs
            return lbfgs('/app/configs/engineering_english.yaml', '/app', resume_checkpoint, resume_sha, steps)
        if arm == 'ctc':
            from iam_tools.ctc_head_ab import run as ctc
            return ctc('/app/configs/engineering_english.yaml', '/app', resume_checkpoint, resume_sha, steps)
        return run('/app/configs/engineering_english.yaml', '/app', arm, steps, resume_checkpoint or None, resume_sha or None)
    finally:
        volume.commit()

@app.local_entrypoint()
def main(train: bool = False, arm: str = 'xy_pen', steps: int = 1000, resume_checkpoint: str = '', resume_sha: str = ''):
    if not train:
        print('Not launched. Pass --train explicitly for an authorized research chunk.'); return
    if arm not in ('xy_pen', 'gmm_xy_pen', 'lbfgs', 'ctc') or not 1 <= steps <= (100 if arm == 'lbfgs' else 3000 if arm == 'ctc' else 5000):
        raise ValueError('unsupported research arm/chunk')
    if bool(resume_checkpoint) != bool(resume_sha): raise ValueError('resume path and SHA must be supplied together')
    if arm in ('lbfgs','ctc') and not resume_checkpoint: raise ValueError('diagnostic needs pinned geometry source')
    print(research.remote(arm, steps, resume_checkpoint, resume_sha))
