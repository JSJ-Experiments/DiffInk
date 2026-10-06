"""Independent CPU reload checks against saved GPU mean arrays."""
import json
from pathlib import Path
import numpy as np
import torch
from .writer_expansion import load,device_batches
from .curve_study import forward_xy
from .pen_ab import file_sha


def equal_nested(a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def parent_from_provenance(directory,root):
    """Read the RUN input, not the selected output's newly recomputed metadata."""
    directory,root=Path(directory),Path(root)
    original=json.loads((directory/'provenance.json').read_text())
    parent_path=root/original['source_rel']
    if parent_path.resolve()==(directory/'checkpoint-best.pt').resolve():
        raise ValueError('cannot audit a checkpoint against itself')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    return parent,original


def verify(directory,repo,root='/data'):
    torch.set_num_threads(4);directory=Path(directory);root=Path(root)
    selected=directory/'checkpoint-best.pt';sha=file_sha(selected)
    model,samples,raw_batches,cfg,vocab,provenance=load(Path(repo)/'configs/engineering_english.yaml',repo,root,
                                             str(selected.relative_to(root)),sha)
    result=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else None
    step=result['best_step'] if result else torch.load(selected,map_location='cpu',weights_only=True)['optimizer_updates']
    parent,original=parent_from_provenance(directory,root)
    if original['samples'] != provenance['samples'] or original['splits'] != provenance['splits']:
        raise AssertionError('sample/split provenance mismatch')
    state=model.state_dict();finite=all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
    frozen=all(torch.equal(v,parent['model_state_dict'][k]) for k,v in state.items() if k.startswith(('style_classifier.','ocr_model.')))
    sigma=all(torch.equal(state[k][63:],parent['model_state_dict'][k][63:]) for k in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias'))
    rows=[]
    with torch.no_grad():
        for sid,(raw,mask,_) in device_batches(raw_batches,'cpu').items():
            xy,_,out=forward_xy(model,raw,mask);n=int(mask.sum());cpu=xy[0,:n].numpy()
            saved=np.load(directory/f'step-{step}/{sid}/mu.npy');pens=out[0,:3,:n].argmax(0).numpy()
            rows.append(dict(sample_id=sid,max_absolute_xy_difference=float(np.abs(cpu-saved[:,:2]).max()),
                             xy_difference_rmse=float(np.sqrt(np.mean((cpu-saved[:,:2])**2))),
                             pen_argmax_mismatches=int((pens!=saved[:,2:].argmax(1)).sum())))
    report=dict(selected_sha256=sha,selected_step=step,finite_model=finite,ocr_style_parameters_unchanged=frozen,
                sigma_rho_rows_unchanged=sigma,source_checkpoint_unchanged=file_sha(root/original['source_rel'])==original['source_sha256'],
                max_cpu_gpu_xy_difference=max(r['max_absolute_xy_difference'] for r in rows),
                pen_argmax_mismatches=sum(r['pen_argmax_mismatches'] for r in rows),lines=rows)
    if not finite or not frozen or not sigma or not report['source_checkpoint_unchanged'] or report['max_cpu_gpu_xy_difference']>1e-4:raise AssertionError(report)
    if result and result.get('resumed_model_optimizer_rng'):
        initial=torch.load(directory/'checkpoint-initial.pt',map_location='cpu',weights_only=True)
        report['resume_state_bitwise_equal']={k:equal_nested(initial[k],parent[k]) for k in ('model_state_dict','optimizer_state_dict','rng_state_cpu','rng_state_cuda')}
        if not all(report['resume_state_bitwise_equal'].values()):raise AssertionError('resume state differs from saved parent')
    (directory/'cpu-reload-check.json').write_text(json.dumps(report,indent=2)+'\n');return report
