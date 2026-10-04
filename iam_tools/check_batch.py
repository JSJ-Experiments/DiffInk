"""Read a tiny dataset with DiffInk's released loader. No model/training run."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from .preprocess import validate_sequence


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def check(out, repo, batch_size=4):
    out,repo=Path(out),Path(repo)
    ds=load_module(repo/'dataset/vae_dataset.py','diffink_dataset_check')
    masks=load_module(repo/'utils/mask.py','diffink_mask_check')
    chars=json.loads((out/'chars.json').read_text());report={}
    membership={};writers={}
    manifest=json.loads((out/"manifest.json").read_text())
    reserved=set(manifest["heldout_writers"])
    for split,cls in [('train',ds.TrainDataset),('val',ds.ValDataset)]:
        with h5py.File(out/f'tiny_{split}.h5','r') as hf:
            membership[split]=set(hf.keys());writers[split]=set()
            for key in hf:
                g=hf[key];seq=g['point_seq'][:];ends=g['stroke_points_idx'][:]
                validate_sequence(seq,ends)
                if len(g['char_points_idx']): raise AssertionError('fabricated char alignment')
                if set(g['line_text'][()].decode())-set(chars):raise AssertionError('OOV')
                writers[split].add(g['writer_id'][()].decode())
                line=g['line_text'][()].decode()
                required=len(line)+sum(a==b for a,b in zip(line,line[1:]))
                if (len(seq)+7)//8 < required:raise AssertionError('CTC infeasible target')
        dataset=cls(str(out/f'tiny_{split}.h5'),str(out/'chars.json'),str(out/'writers.json'),transform=None)
        try:
            if len(dataset)!=len(membership[split]): raise AssertionError('loader silently filtered samples')
            loader=DataLoader(dataset,batch_size=batch_size,num_workers=0,shuffle=False,collate_fn=cls.collate_fn)
            seq,mask,text,char_indices,writer_ids=next(iter(loader))
            if seq.device.type!='cpu' or seq.shape[2]!=5 or seq.shape[1]%8: raise AssertionError('batch shape/device')
            lengths=[len(dataset[i][1]) for i in range(seq.shape[0])]
            if mask.sum(dim=1).tolist()!=lengths:raise AssertionError('padding collision')
            temporal=masks.build_prefix_mask_from_char_points(char_indices,mask,compression_factor=8,point_seq=seq)
            latent,pad,suffix=temporal
            for b,n in enumerate(lengths):
                cutoff=int((suffix[b]==0).sum())
                ends=torch.nonzero(seq[b,:n,2]==0).flatten()+1
                if cutoff and cutoff not in ends.tolist(): raise AssertionError('prefix not stroke aligned')
                if cutoff>=n:raise AssertionError('full-line reference')
            if not torch.isfinite(seq).all():raise AssertionError('nonfinite batch')
            # Loader text IDs start at 0. OCR shifts by +1 before CTCLoss(blank=0).
            for b in range(seq.shape[0]):
                actual=dataset[b][2]
                decoded=''.join(list(chars)[i] for i in text[b].tolist() if i>=0)
                if decoded!=actual:raise AssertionError('text changed or synthetic validation token')
            report[split]={'samples':len(dataset),'batch_shape':list(seq.shape),
                           'valid_lengths':lengths,'text_shape':list(text.shape),
                           'prefix_end_points':[(row==0).sum().item() for row in suffix],
                           'latent_mask_shape':list(latent.shape),'writer_count':len(writers[split])}
        finally:dataset.hf.close()
    if membership['train']&membership['val'] or writers['train']&writers['val']:
        raise AssertionError('split leakage')
    if reserved & (writers['train']|writers['val']):raise AssertionError('test writer leakage')
    report.update({'reserved_test_writers':len(reserved),'ctc_targets_feasible':True,'training':False,'gpu':False,'device':'cpu','writer_disjoint':True,
                   'repo':str(repo),'upstream_commit':'97bc6a3c39a5bdaa9728daaab6d3707480006343',
                   'patch':'English val token + stroke prefix + exact CTC feasibility; TrainDataset unchanged'})
    (out/'batch_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='data/diffink/iam')
    default='third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    p.add_argument('--repo',default=default);p.add_argument('--batch-size',type=int,default=4)
    args=vars(p.parse_args());print(json.dumps(check(**args),indent=2))
if __name__=='__main__':main()
