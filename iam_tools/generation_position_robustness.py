"""Relative-PE nuisance jitter only; target geometry/index sequence never changed."""
import torch


def jitter_lengths(lengths,generator,*,probability=.5,fraction=.2):
    if lengths.ndim!=1 or lengths.dtype!=torch.long or (lengths<2).any() or not 0<=probability<=1 or not 0<fraction<=.25:raise ValueError('B-long>=2 and bounded nuisance-jitter settings required')
    # CPU generator makes each predeclared row nuisance reproducible across devices.
    use=torch.rand(len(lengths),generator=generator)<probability
    scale=1+fraction*(2*torch.rand(len(lengths),generator=generator)-1)
    proposal=(lengths.cpu()*scale).round().long().clamp(2,256)
    return torch.where(use,proposal,lengths.cpu()).to(lengths.device)
