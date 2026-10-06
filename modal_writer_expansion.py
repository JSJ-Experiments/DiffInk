"""Guarded T4 same-writer reconstruction expansion; no GPU allocated without --train."""
from pathlib import Path
import modal

app = modal.App('diffink-english-writer-expansion')
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
def research(steps: int, lr: float, mode: str, source_rel: str, source_sha: str, resume: bool):
    try:
        if mode == 'polish':
            from iam_tools.writer_polish import run
            return run('/app/configs/engineering_english.yaml', '/app', '/data', source_rel, source_sha, steps=steps)
        from iam_tools.writer_expansion import run
        kwargs=dict(source_rel=source_rel,source_sha=source_sha) if source_rel else {}
        kwargs['balanced']=mode=='balanced';kwargs['resume']=resume
        return run('/app/configs/engineering_english.yaml', '/app', steps=steps, lr=lr, **kwargs)
    finally:
        volume.commit()


@app.function(image=image, volumes={'/data': volume}, cpu=4, memory=8192,
              timeout=900, retries=0, max_containers=1)
def render_report(report_rel: str):
    import json, shutil
    from iam_tools.report_writer_expansion import report
    from iam_tools.pen_ab import file_sha
    from iam_tools.verify_writer_checkpoint import verify
    directory=Path('/data')/report_rel
    if not report_rel.startswith(('checkpoints/iam_writer_expansion/','checkpoints/iam_writer_polish/')) or '..' in Path(report_rel).parts:
        raise ValueError('study directory required')
    try:
        reload_check=verify(directory,'/app','/data')
        info=report(directory,'/data')
        info['cpu_reload_check']=reload_check
        sha=file_sha(directory/'checkpoint-best.pt')
        info['selected_checkpoint_sha256']=sha
        info['source_code_sha256']={str(p.relative_to(directory/'source-code')):file_sha(p) for p in (directory/'source-code').rglob('*.py')}
        (directory/'report/summary.json').write_text(json.dumps(info,indent=2)+'\n')
        latest=directory.parent/'latest'
        if latest.exists():shutil.rmtree(latest)
        shutil.copytree(directory/'report',latest)
        return dict(output=str(directory/'report'),selected_sha256=sha,selected_step=info['selected_step'],metrics=info['final'])
    finally:volume.commit()


@app.function(image=image, volumes={'/data': volume}, cpu=2, memory=4096, timeout=300)
def checkpoint_info(checkpoint_rel: str):
    import torch,json
    from iam_tools.pen_ab import file_sha
    p=Path('/data')/checkpoint_rel
    if not checkpoint_rel.startswith('checkpoints/iam_writer_expansion/') or '..' in Path(checkpoint_rel).parts:
        raise ValueError('writer expansion checkpoint required')
    saved=torch.load(p,map_location='cpu',weights_only=True)
    result=dict(path=str(p),sha256=file_sha(p),updates=saved['optimizer_updates'],config=saved['config'],
                sample_ids=saved['sample_ids'],rng_saved=all(k in saved for k in ('rng_state_cpu','rng_state_cuda')))
    (p.parent/'resume-info.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result


@app.local_entrypoint()
def main(train: bool = False, steps: int = 1200, lr: float = 1e-5, mode: str = 'adam', source_rel: str = '', source_sha: str = '', report_rel: str = '', inspect_rel: str = '', resume: bool = False):
    if inspect_rel:
        print(checkpoint_info.remote(inspect_rel)); return
    if report_rel:
        print(render_report.remote(report_rel)); return
    if not train:
        print('No GPU allocated. Explicit --train required.'); return
    if mode not in ('adam','polish','balanced') or not 1 <= steps <= (100 if mode=='polish' else 2000) or not 0 < lr <= 5e-5 or bool(source_rel)!=bool(source_sha) or (mode in ('polish','balanced') and not source_rel) or (resume and (mode=='polish' or not source_rel)):
        raise ValueError('bounded expansion requires valid steps/LR')
    print(research.remote(steps, lr, mode, source_rel, source_sha, resume))
