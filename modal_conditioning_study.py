"""Explicitly guarded matched T4 reconstruction-conditioning study."""
from pathlib import Path
import modal
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
app=modal.App('diffink-english-conditioning-study')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=4200,retries=0,max_containers=1)
def research(steps:int,modes:str,source_rel:str,source_sha:str):
    from iam_tools.conditioning_study import run
    try:
        kwargs=dict(source_rel=source_rel,source_sha=source_sha) if source_rel else {}
        return run('/app/configs/engineering_english.yaml','/app',steps=steps,modes=tuple(modes.split(',')),**kwargs)
    finally:volume.commit()

@app.function(image=image,volumes={'/data':volume},cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1)
def render(directory_rel:str):
    import json
    from iam_tools.conditioning_study import verify
    from iam_tools.report_writer_expansion import report
    from iam_tools.reference_retention import report as reference_report
    if not directory_rel.startswith('checkpoints/iam_conditioning_study/') or '..' in Path(directory_rel).parts:raise ValueError('study directory required')
    directory=Path('/data')/directory_rel;results={}
    try:
        for arm in sorted(directory.iterdir()):
            if not (arm/'result.json').exists():continue
            check=verify(arm,'/app');info=report(arm,'/data');info['cpu_reload_check']=check
            (arm/'report/summary.json').write_text(json.dumps(info,indent=2)+'\n');reference_report(arm,'/data');results[arm.name]=info['final']
        from iam_tools.report_conditioning import report as combined_report
        combined_report(directory,'/data')
        from iam_tools.report_pointer import publish_pointer
        publish_pointer(directory.parent,directory/'report')
        return dict(output=str(directory/'report'),arms=results)
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,steps:int=1000,modes:str='control,center,channel',report_rel:str='',source_rel:str='',source_sha:str=''):
    if report_rel:print(render.remote(report_rel));return
    if not train:print('No GPU allocated; explicit --train required.');return
    arms=modes.split(',')
    if not 1<=steps<=2000 or not 1<=len(arms)<=3 or len(set(arms))!=len(arms) or any(m not in ('control','center','channel','edge_pad') for m in arms) or bool(source_rel)!=bool(source_sha):raise ValueError('bounded study and pinned source required')
    print(research.remote(steps,modes,source_rel,source_sha))
