"""CPU-only tiny English data conversion; no training/model/GPU imports."""
import modal
app=modal.App('diffink-iam-convert')
volume=modal.Volume.from_name('diffink-data')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0')
       .add_local_python_source('iam_tools'))
@app.function(image=image,volumes={'/data':volume},cpu=2,memory=2048,timeout=900)
def prepare(train_size: int=250,val_size: int=50):
    from iam_tools.build import build
    report=build('/data/raw/iam','/data/canonical/iam/tiny','/data/diffink/iam',train_size=train_size,val_size=val_size)
    volume.commit()
    return {key:report[key] for key in ('stage','training','gpu','paper_equivalent','paired_lines','paired_writers','vocabulary_size','processed_length_range','sample_writers')}
@app.local_entrypoint()
def main(train_size: int=250,val_size: int=50):
    import json
    print(json.dumps(prepare.remote(train_size,val_size),indent=2))
