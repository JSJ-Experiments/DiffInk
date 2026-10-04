"""Persist validated generated artifacts; prune stale generated files, never raw IAM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import modal


def publish(overfit=False):
    name='iam_overfit' if overfit else 'iam'
    kind='overfit' if overfit else 'tiny'
    root=Path('data/diffink')/name
    canonical=Path('data/canonical/iam')/kind
    remote_root=f'/diffink/{name}'
    remote_canonical=f'/canonical/iam/{kind}'
    report=json.loads((root/'manifest.json').read_text())
    batch=json.loads((root/'batch_check.json').read_text())
    if batch['training'] or batch['gpu'] or not batch['line_disjoint']:
        raise ValueError('unexpected validation report')
    if overfit != batch['overfit_same_writers'] or (not overfit and not batch['writer_disjoint']):
        raise ValueError('wrong split mode')
    volume=modal.Volume.from_name('diffink-data')
    uploads={}
    for name in ('tiny_train.h5','tiny_val.h5','chars.json','writers.json','manifest.json','batch_check.json','vae_smoke.json'):
        if (root/name).exists():uploads[remote_root+'/'+name]=root/name
    uploads[remote_root+'/previews/index.html']=root/'previews/index.html'
    for sid in report['previews']:
        for suffix in ('-raw.png','-raw.svg','-processed.png','-processed.svg','-comparison.png'):
            uploads[remote_root+'/previews/'+sid+suffix]=root/'previews'/(sid+suffix)
    for ids in report['sample_ids'].values():
        for sid in ids:uploads[remote_canonical+'/'+sid+'.json']=canonical/(sid+'.json')
    # Upload a complete valid build before pruning old generated artifacts.
    with volume.batch_upload(force=True) as upload:
        for dst,src in uploads.items():upload.put_file(src,dst)
    stale=[]
    for directory,suffixes in [(remote_canonical,('.json',)),(remote_root+'/previews',('.png','.svg','.html'))]:
        for entry in volume.iterdir(directory,recursive=True):
            path='/'+entry.path.lstrip('/')
            if not path.startswith(directory+'/'):raise ValueError('unexpected Volume listing path')
            if path.endswith(suffixes) and path not in uploads:stale.append(path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(volume.remove_file,stale))
    result={'train':len(report['sample_ids']['train']),'val':len(report['sample_ids']['val']),
            'remote':remote_root,'stale_generated_files_removed':len(stale),'training':False}
    print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--overfit',action='store_true')
    publish(**vars(p.parse_args()))
if __name__=='__main__':main()
