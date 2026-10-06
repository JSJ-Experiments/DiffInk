"""Capture-safe research loss and CUDA replay; identical minimal physical padding.

No optimizer, no architecture changes, no approximate precision. Explicit noise
is drawn outside capture, so replay does not freeze the sampled posterior.
"""
import torch
from contextlib import contextmanager
from torch.nn import functional as F
from .latent_integration import DELTA_WEIGHT


def masked_geometry(xy, truth, states, mask):
    # Select-safe arithmetic even if unused padding contains NaN.
    residual = torch.where(mask[..., None], xy-truth, torch.zeros_like(xy))
    point = residual.square().sum()/(mask.sum()*2)
    edges = mask[:, :-1] & mask[:, 1:] & (states[:, :-1] == 0)
    delta = torch.where(edges[..., None], residual[:, 1:]-residual[:, :-1], 0.)
    return point+DELTA_WEIGHT*delta.square().sum()/(edges.sum().clamp_min(1)*2)


def masked_pen(logits, states, mask):
    counts = torch.stack([((states == i) & mask).sum() for i in range(3)]).to(logits.dtype)
    weights = (counts[0]/counts.clamp_min(1)).sqrt().clamp(max=8)
    weights = torch.where(counts > 0, weights, 0.)
    logits = torch.where(mask[:, None], logits, torch.zeros_like(logits))
    ce = F.cross_entropy(logits, states, reduction='none')
    return (weights[states]*(1-ce.neg().exp()).square()*ce*mask).sum()/mask.sum()


@contextmanager
def dense_training_decoder(model):
    # PyTorch checks mask left-alignment BEFORE noticing gradients disable the
    # nested inference path. That Boolean check synchronizes CUDA and is unsafe
    # in capture. Skip an inapplicable inference optimization, not the mask.
    decoder = getattr(model, 'transformer_decoder', None)
    transformer = getattr(decoder, 'transformer', None)
    previous = getattr(transformer, 'use_nested_tensor', None)
    if previous is not None: transformer.use_nested_tensor = False
    try: yield
    finally:
        if previous is not None: transformer.use_nested_tensor = previous


def fast_terms(model, raw, mask, epsilon=None):
    """Geometry/pen only. Unused KL/CTC/style are not computed.

    Caller validates nonempty finite raw batches outside capture. Rectangular
    masked reductions avoid Boolean indexing/nonzero and scalar CPU reads.
    """
    from model.losses import mixture_expectation
    scaled = model.to_model_space(raw)
    features = model.encoder(scaled)
    mu = model.conv_mu(features)
    truth, states = scaled[:, :2].transpose(1, 2), raw[:, 2:].argmax(1)
    with dense_training_decoder(model):
        out = model.decode(mu, padding_mask=~mask)
    mean = masked_geometry(mixture_expectation(out), truth, states, mask)
    pen = masked_pen(out[:, :3], states, mask)
    sampled = mean*0
    if epsilon is not None:
        lv = model.conv_logvar(features)
        z = mu+epsilon*(.5*lv).exp()
        with dense_training_decoder(model):
            sampled_out = model.decode(z, padding_mask=~mask)
        sampled = masked_geometry(mixture_expectation(sampled_out), truth, states, mask)
        pen = .5*(pen+masked_pen(sampled_out[:, :3], states, mask))
    return torch.stack([mean, sampled, pen])


class GeometryGraphs:
    """One capture per EXACT padded length; persistent shared parameter gradients.

    Graphs use independent memory pools. Gradient pointers must never be replaced:
    zero with set_to_none=False, and clip/update only between effective batches.
    Capture/warmup does not update weights or consume the caller's RNG stream.
    """
    def __init__(self, model, pen_weight, sampled_weight=.1, max_graphs=80):
        if next(model.parameters()).device.type != 'cuda' or model.training:
            raise ValueError('CUDA eval-mode model required (dropout must stay off)')
        if max_graphs < 1 or pen_weight < 0 or sampled_weight < 0:
            raise ValueError('nonnegative bounded loss weights and graph count required')
        self.model, self.cache = model, {}
        self.sampled = sampled_weight > 0
        self.weights = torch.tensor([1., sampled_weight, pen_weight], device='cuda')
        self.base_weights = self.weights.clone()
        self.max_graphs = max_graphs
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        for p in self.parameters:
            if p.grad is None: p.grad = torch.zeros_like(p)
        self.pointers = [p.grad.data_ptr() for p in self.parameters]

    def prepare(self, batches):
        # Capture is preparation only; require a clean effective-batch boundary.
        for raw, mask, _ in batches:
            if raw.shape[0] != 1 or raw.shape[-1]%8 or not mask.any() or not torch.isfinite(raw).all():
                raise ValueError('finite minimally padded one-line multiple-of-eight input required')
            key = raw.shape[-1]
            n = int(mask.sum())
            if ((n+7)//8)*8 != key or not torch.equal(mask[0], torch.arange(key,device=mask.device)<n):
                raise ValueError('prefix mask and minimal multiple-of-eight padding required')
            if key in self.cache: continue
            if len(self.cache) >= self.max_graphs:
                raise ValueError('graph cache cap reached; do not silently change padding')
            static_raw, static_mask = raw.clone(), mask.clone()
            epsilon = torch.zeros((1, self.model.config.latent_dim, key//8), device=raw.device) if self.sampled else None
            with torch.random.fork_rng(devices=[raw.device.index or 0]):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.model.zero_grad(set_to_none=False)
                        (fast_terms(self.model, static_raw, static_mask, epsilon)*self.weights).sum().backward()
                torch.cuda.current_stream().wait_stream(stream)
                self.model.zero_grad(set_to_none=False)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    metrics = fast_terms(self.model, static_raw, static_mask, epsilon)
                    (metrics*self.weights).sum().backward()
            # CUDA graph owns device work; do not keep Python autograd graphs
            # (and their AccumulateGrad stream references) alive between captures.
            metrics = metrics.detach()
            self.cache[key] = (graph, static_raw, static_mask, epsilon, metrics)
        self.model.zero_grad(set_to_none=False)

    def replay(self, batch, divisor=8, epsilon=None):
        raw, mask, _ = batch
        if divisor <= 0: raise ValueError('positive effective-batch divisor required')
        if [p.grad.data_ptr() if p.grad is not None else None for p in self.parameters] != self.pointers:
            raise RuntimeError('captured gradient buffers were replaced; never set_to_none=True')
        graph, static_raw, static_mask, noise, metrics = self.cache[raw.shape[-1]]
        static_raw.copy_(raw); static_mask.copy_(mask)
        if noise is not None:
            noise.copy_(torch.randn_like(noise) if epsilon is None else epsilon)
        elif epsilon is not None: raise ValueError('mean-only graph cannot use sampled noise')
        # Weight scaling lives in a persistent input buffer, not in graph constants.
        self.weights.copy_(self.base_weights/divisor)
        graph.replay()
        return metrics.detach().clone()
