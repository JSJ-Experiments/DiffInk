"""Opt-in frozen English OCR head study; handwriting codec cannot update."""
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
app=modal.App('diffink-english-frozen-ocr-study')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=2400,retries=0,max_containers=1)
def research(steps:int,sampled:bool=False,source:str='',sha:str=''):
    from iam_tools.frozen_ocr_study import run,SOURCE,SHA
    from iam_tools.metric_workers import metric_pool
    if source and (not source.startswith('checkpoints/iam_frozen_ocr_study/') or '..' in Path(source).parts):
        raise ValueError('bounded source checkpoint required')
    try:
        with metric_pool(3) as pool:return run('/app/configs/engineering_english.yaml','/app',steps=steps,metric_pool=pool,sampled=sampled,source=source or SOURCE,sha=sha or SHA)
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1)
def render(relative:str):
    if not relative.startswith('checkpoints/iam_frozen_ocr_study/') or '..' in Path(relative).parts:raise ValueError('bounded OCR report required')
    from iam_tools.report_frozen_ocr import report,research_summary
    try:
        if relative=='checkpoints/iam_frozen_ocr_study/research-summary':return research_summary(Path('/data')/'checkpoints/iam_frozen_ocr_study','/data')
        return report(Path('/data')/relative,'/app','/data')
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=1000,sampled:bool=False,source:str='',sha:str='',report_rel:str=''):
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated. Use --train for the frozen codec OCR study.');return
    if not 1<=steps<=3000:raise ValueError('1–3000 head updates required')
    print(research.remote(steps,sampled,source,sha))
