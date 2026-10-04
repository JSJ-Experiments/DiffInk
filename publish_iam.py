"""Persist locally generated tiny artifacts to Modal; never uploads data to GitHub."""
from pathlib import Path
import json
import modal

def main():
    root=Path('data/diffink/iam')
    report=json.loads((root/'manifest.json').read_text())
    batch=json.loads((root/'batch_check.json').read_text())
    if batch['training'] or batch['gpu'] or not batch['writer_disjoint']:
        raise ValueError('unexpected validation report')
    volume=modal.Volume.from_name('diffink-data')
    with volume.batch_upload(force=True) as upload:
        # Exact output files and current preview IDs; no raw archive upload needed.
        for name in ('tiny_train.h5','tiny_val.h5','chars.json','writers.json','manifest.json','batch_check.json'):
            upload.put_file(root/name,'/diffink/iam/'+name)
        upload.put_file(root/'previews/index.html','/diffink/iam/previews/index.html')
        sheet=root/'previews/contact-sheet.png'
        if sheet.exists():upload.put_file(sheet,'/diffink/iam/previews/contact-sheet.png')
        for sid in report['previews']:
            for suffix in ('-raw.png','-raw.svg','-processed.png','-processed.svg','-comparison.png'):
                upload.put_file(root/'previews'/(sid+suffix),'/diffink/iam/previews/'+sid+suffix)
        for split,ids in report['sample_ids'].items():
            for sid in ids:
                upload.put_file(Path('data/canonical/iam/tiny')/(sid+'.json'),'/canonical/iam/tiny/'+sid+'.json')
    print('Persisted 250 train + 50 val lines, metadata, raw canonical JSON and comparison previews to diffink-data.')

if __name__=='__main__':main()
