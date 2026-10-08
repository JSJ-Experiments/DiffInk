"""Refresh target-geometry anchors at the current TRAIN checkpoint, not smoothing."""
import torch
from .latent_diffusion import masked_mse
from .generation_geometry import physical_terms,coefficients
from .generation_coverage_study import writer_tensor


def calibrate(model,pool,records,writers,stats,batches,train_ids,fractions=(.25,.10)):
    if not batches or any(not b or not set(b)<=set(train_ids) for b in batches):
        raise ValueError('nonempty actual TRAIN-only calibration required')
    parameters=[p for name,p in model.named_parameters() if not name.startswith('final.')]
    if not parameters:raise ValueError('shared backbone parameters required')
    buffers={k:[torch.zeros_like(p) for p in parameters] for k in ('base','xy','first_difference')}
    was=model.training;model.eval()
    try:
        for batch in batches:
            clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,writers,clean.device)
            pred=model(torch.zeros_like(clean),torch.ones(len(batch),device=clean.device),text,mask,writer_ids=wi)
            terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');terms['base']=masked_mse(pred,clean,mask)
            for key in buffers:
                grads=torch.autograd.grad(terms[key],parameters,retain_graph=True,allow_unused=True)
                for buffer,g in zip(buffers[key],grads):
                    if g is not None:buffer.add_(g.detach()/len(batches))
        norms={k:float(sum(g.square().sum() for g in gs).sqrt()) for k,gs in buffers.items()}
        weights=coefficients(norms['base'],norms['xy'],norms['first_difference'],fractions)
        return dict(weights=weights,aggregate_gradient_norms=norms,gradient_fractions=dict(xy=fractions[0],first_difference=fractions[1]),batches=batches,
                    scope='mean gradient across listed TRAIN batches, shared parameters excluding final readout; no backward buffers or optimizer updates',
                    definition='real-point isotropic XY and true-stroke target index differences; not generic smoothness or physical velocity')
    finally:model.train(was)
