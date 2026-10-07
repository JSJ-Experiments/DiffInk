"""Reuse a source/data-bound frozen geometry audit; refresh all encoder/point checks.

Avoid repeatedly rendering/scoring 8352 identical decoder outputs on a paid GPU.
This does NOT claim a new full-pool decoder or stochastic trajectory certification.
"""
import json,hashlib,shutil
from pathlib import Path
import torch,h5py
from .ocr_pool_study import single_batch,SOURCE,SHA
from .latent_integration import encoded
from .ocr_context_features import unpack
from .pen_ab import file_sha
GATE_REL='checkpoints/iam_ocr_augmentation/20261007-085841'


def validate_gate(root,pool,m,cfg):
    d=Path(root)/GATE_REL
    c=json.loads((d/'clean_control/config.json').read_text())
    if c['cfg']!=cfg or c['source_rel']!=SOURCE or c['source_sha256']!=SHA:
        raise ValueError('reused decoder audit must bind exact codec/config')
    if (d/'pool-manifest.json').read_bytes()!=(Path(pool)/'manifest.json').read_bytes():
        raise ValueError('reused decoder audit must bind exact pool bytes')
    gate=json.loads((d/'codec-preflight.json').read_text())
    if not gate['passed'] or gate['failed_checks'] or len(gate['lines'])!=len(m['records']):
        raise ValueError('complete previously-passed mean decoder/pen gate required')
    if {r['sample_id'] for r in gate['lines']}!=set(m['records']):
        raise ValueError('reused audit sample IDs changed')
    if file_sha(Path(root)/SOURCE)!=SHA:raise ValueError('immutable codec SHA changed')
    return d,gate


@torch.no_grad()
def cache_reader_corpus(model,pool,m,vocab,out,root,cfg,device='cuda'):
    d,gate=validate_gate(root,pool,m,cfg);out=Path(out);cache={};texts={i:r['text'] for i,r in m['records'].items()};maximum=0.
    with h5py.File(Path(pool)/'lines.h5') as hf:
        for sid in sorted(m['records']):
            points=hf[sid]['point_seq'][:];r=m['records'][sid]
            if hashlib.sha256(points.tobytes()).hexdigest()!=r['points_sha256']:raise ValueError('point fingerprint drift')
            raw,mask,labels=single_batch(points,texts[sid],vocab,device);target,mu,lv,lm=encoded(model,raw,mask);n=len(points)
            f,real=unpack(mu,lm);packed=f.permute(0,3,1,2).reshape(1,-1,5)[0,:n]
            diff=float((packed[:,:2]-target[0,:n]).abs().max());maximum=max(maximum,diff)
            required=len(texts[sid])+sum(a==b for a,b in zip(texts[sid],texts[sid][1:]))
            if int(real.sum())!=n or (packed[:,2:].argmax(1)!=raw[0,2:,:n].argmax(0)).any() or diff>.001 or int(lm.sum())<required:
                raise AssertionError('refreshed packed encoder/CTC check failed: '+sid)
            cache[sid]=dict(mu=mu.detach(),lv=lv.detach(),mask=lm.detach(),labels=labels.detach())
    for name in ('codec-preflight.json','geometry-source.h5'):shutil.copyfile(d/name,out/name)
    proof=dict(prior_decoder_gate_rel=GATE_REL,prior_decoder_gate_sha256=file_sha(d/'codec-preflight.json'),prior_geometry_h5_sha256=file_sha(d/'geometry-source.h5'),
               source_sha256=SHA,pool_manifest_sha256=file_sha(Path(pool)/'manifest.json'),new_encoder_checks=len(cache),maximum_new_packed_xy_difference=maximum,
               all_new_real_points_masks_pens_ctc_checked=True,decoder_scope='Exact prior source/config/pool8352 mean gate reused; NOT a newly decoded8352 or sampledtrajectory check. CPU report independently decodes192 aftertraining.')
    (out/'geometry-gate-reuse.json').write_text(json.dumps(proof,indent=2)+'\n')
    return cache,texts,gate
