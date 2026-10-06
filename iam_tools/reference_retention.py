"""Marker-free target / immutable eight-line reference / expanded-model gallery."""
import json
from pathlib import Path
import numpy as np
import h5py
from .curve_audit import draw,split_xy
from .pen_ab import file_sha

REFERENCE='checkpoints/iam_latent_integration/20261005-133620/ocr/step-400'
REFERENCE_SHA='fffe1405db10f8f6c2ce6ec1e030706b7947c93d83fa4eaeffec3b6a8c5b08f9'


def validate_sequences(sequences):
    """Require aligned finite N x 5 trajectories before any visual comparison."""
    arrays=[np.asarray(a) for a in sequences]
    if len(arrays)!=3 or arrays[0].ndim!=2 or arrays[0].shape[1]!=5 or not len(arrays[0]):
        raise ValueError('three aligned nonempty N x 5 trajectories required')
    if any(a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('comparison trajectories must be aligned and finite')
    return arrays


def report(directory,data_root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);root=Path(data_root);out=directory/'report'
    info=json.loads((out/'summary.json').read_text());ids=info['provenance']['splits']['old'];step=info['selected_step']
    source=root/REFERENCE
    if not all((source/sid/'mu.npy').exists() for sid in ids):return None
    arrays={};hashes={}
    with h5py.File(root/'diffink/iam_overfit/tiny_train.h5') as hf:
        for sid in ids:
            target=hf[sid]['point_seq'][:].copy();target[:,:2]*=.01
            arrays[sid]=validate_sequences([target,np.load(source/sid/'mu.npy'),np.load(directory/f'step-{step}/{sid}/mu.npy')])
            hashes[sid]=file_sha(source/sid/'mu.npy')
    headings=['IAM/RDP target','immutable eight-line reference','expanded model']
    fig,axes=plt.subplots(len(ids),3,figsize=(18,2*len(ids)),squeeze=False)
    for row,sid in enumerate(ids):
        allxy=np.concatenate([a[:,:2] for a in arrays[sid]]);lo=allxy.min(0);hi=allxy.max(0)
        for ax,a,label in zip(axes[row],arrays[sid],headings):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_title(sid+' — '+label,fontsize=9)
            ax.set_xlim(lo[0]-.02,hi[0]+.02);ax.set_ylim(lo[1]-.05,hi[1]+.05)
    fig.tight_layout();fig.savefig(out/'original-reference-retention.png',dpi=150);plt.close(fig)
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axes=plt.subplots(3,2,figsize=(8,9))
    for col,(sid,xlim,ylim,label) in enumerate(regions):
        for ax,a,heading in zip(axes[:,col],arrays[sid],headings):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(label+' — '+heading,fontsize=9)
    fig.tight_layout();fig.savefig(out/'original-reference-user-regions.png',dpi=160);plt.close(fig)
    meta=dict(source_checkpoint_sha256=REFERENCE_SHA,source_arrays_sha256=hashes,reference=str(source),target_pen_policy='ground truth',reconstruction_pen_policy='predicted')
    info['immutable_original_reference']=meta;(out/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    page=out/'index.html';text=page.read_text().replace('All predicted pens, no smoothing.', 'Ground-truth target pens; predicted reconstruction pens. No smoothing.')
    if 'original-reference-retention.png' not in text:
        text+='\n<h2>Against the immutable faithful eight-line reference</h2><p>Ground-truth target pens; predicted reconstruction pens. No smoothing. This highlights retention regressions instead of comparing only to the already-degraded continuation source.</p><img style="max-width:100%" src="original-reference-user-regions.png"><img style="max-width:100%" src="original-reference-retention.png">'
    page.write_text(text)
    return meta
