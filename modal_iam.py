"""CPU-only IAM raw-data preparation and visual parser gate. Never trains."""
import modal

app = modal.App('diffink-iam-preview')
volume = modal.Volume.from_name('diffink-data')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('Pillow==12.3.0').add_local_python_source('iam_tools'))

@app.function(image=image, volumes={'/data':volume}, cpu=2, timeout=600)
def prepare(limit: int = 10):
    from pathlib import Path
    import shutil
    import tarfile
    import hashlib
    from iam_tools.preview import preview

    raw = Path('/data/raw/iam')
    raw.mkdir(parents=True, exist_ok=True)
    for directory in ('canonical/iam', 'diffink/iam', 'checkpoints'):
        (Path('/data') / directory).mkdir(parents=True, exist_ok=True)
    # Retain the existing /iam files. Never rename or delete the user's originals.
    for name in ('lineStrokes-all.tar.gz','ascii-all.tar.gz','forms.txt','writers.xml'):
        dst = raw / name
        src = Path('/data/iam') / name
        if not dst.exists():
            if not src.is_file():
                raise FileNotFoundError(f'Expected {src} or {dst}')
            shutil.copyfile(src,dst)
        elif src.exists():
            def digest(p):
                with p.open('rb') as f:
                    return hashlib.file_digest(f,'sha256').hexdigest()
            if digest(src) != digest(dst):
                raise ValueError(f'Existing raw copy differs from {src}; refusing overwrite')
    for archive, folder in (('lineStrokes-all.tar.gz','lineStrokes'),('ascii-all.tar.gz','ascii')):
        marker = raw / (archive+'.extracted')
        if not marker.exists():
            # data filter rejects traversal, escaping links, devices, etc.
            with tarfile.open(raw / archive) as tar:
                tar.extractall(raw, filter='data')
            if not (raw / folder).is_dir():
                raise ValueError(f'{archive}: expected extracted {folder}/')
            marker.write_text('complete\n')
    volume.commit()  # Keep extraction even if a later preview fails.
    report = preview(raw, '/data/canonical/iam/preview', limit=limit)
    volume.commit()
    return report

@app.local_entrypoint()
def main(limit: int = 10):
    import json
    print(json.dumps(prepare.remote(limit), indent=2))
