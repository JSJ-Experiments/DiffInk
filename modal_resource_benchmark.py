"""Explicit bounded T4 telemetry + dispatch/batch benchmark; no promotion."""
from pathlib import Path
import modal
volume=modal.Volume.from_name('diffink-data')
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
 .workdir('/app').add_local_dir(str(repo/'model'),'/app/model')
 .add_local_dir(str(repo/'dataset'),'/app/dataset').add_local_dir(str(repo/'utils'),'/app/utils')
 .add_local_dir(str(repo/'trainer'),'/app/trainer').add_local_dir('iam_tools','/app/iam_tools')
 .add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
app=modal.App('diffink-english-resource-benchmark')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=1200,retries=0,max_containers=1)
def benchmark(seconds:int):
 from iam_tools.generation_benchmark import run
 try:return run(seconds=seconds)
 finally:volume.commit()

@app.local_entrypoint()
def main(benchmark_gpu:bool=False,seconds:int=45):
 if not benchmark_gpu:print('No GPU allocated. --benchmark-gpu runs5 disposable35–90s cases.');return
 if not 35<=seconds<=90:raise ValueError('bounded35–90s per case')
 print(benchmark.remote(seconds))
