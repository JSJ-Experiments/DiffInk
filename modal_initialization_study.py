"""Bounded opt-in initialization stability study. No KL/CTC/style."""
from pathlib import Path
import modal

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
import os
app=modal.App(os.environ.get('DIFFINK_STUDY_APP','diffink-english-initialization-stability'))

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,
              timeout=2400,retries=0,max_containers=1)
def research(steps:int, arm:str):
    from iam_tools.initialization_study import run
    from iam_tools.metric_workers import metric_pool
    try:
        with metric_pool(3) as pool:
            return run('/app/configs/engineering_english.yaml','/app',steps=steps,modes=(arm,) if arm else ('uniform','scaled_readout','frozen_readout'),metric_pool=pool)
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1800,retries=0,max_containers=1)
def render(relative:str):
    from iam_tools.report_initialization import report
    from iam_tools.initialization_study import verify,update_attribution
    if not relative.startswith('checkpoints/iam_initialization_study/') or '..' in Path(relative).parts:raise ValueError('bounded initialization report required')
    try:
        directory=Path('/data')/relative
        import json
        from iam_tools.initialization_study import SOURCE,SHA
        (directory/'resolved-source-contract.json').write_text(json.dumps(dict(source_rel=SOURCE,source_sha256=SHA,source_role='geometry explicitly reinitialized; dormant weights and frozen heads reused',note='Initial dated runs inherit nonoperative source_checkpoint/output_base from YAML; guarded load constants and provenance.json are authoritative.'),indent=2)+'\n')
        for arm in directory.iterdir():
            if arm.is_dir() and (arm/'checkpoint-last.pt').exists():
                from iam_tools.pen_ab import file_sha
                checked=arm/'cpu-reload-check.json'
                cached=json.loads(checked.read_text()) if checked.exists() else None
                if not cached or cached.get('final_sha256')!=file_sha(arm/'checkpoint-last.pt'):
                    verify(arm,'/app')
                if arm.name=='uniform' and not (arm/'first-update-attribution.json').exists():update_attribution(arm,'/app')
        summary=report(directory,'/data',include_draws=(directory/'protected_noise').is_dir())
        from iam_tools.report_pointer import publish_pointer
        if 'protected_noise' in summary['results']:publish_pointer(directory.parent,directory/'report')
        return dict(output='/data/'+relative+'/report/index.html',final=summary['final'])
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=600,retries=0,max_containers=1)
def summary():
    from iam_tools.report_initialization import overview
    from iam_tools.initialization_study import verify
    try:
        verify('/data/checkpoints/iam_initialization_study/20261006-170142/protected_noise','/app')
        return overview('/data')
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=100,arm:str='',report_rel:str='',overview:bool=False):
    if overview:print(summary.remote());return
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated; --train required.');return
    if not 1<=steps<=300:raise ValueError('1–300 steps per arm required')
    if arm and arm not in ('uniform','scaled_readout','frozen_readout','protected_noise'):raise ValueError('known arm required')
    print(research.remote(steps,arm))
