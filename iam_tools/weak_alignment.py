"""TRAIN-only forced-reader timing auxiliary; NOT exact IAM segmentation.

Two four-index CTC frames feed each packed-eight query. Blank frames provide no
label; no interpolation across blanks, inferred glyph borders, or smoothing.
If two characters occur in a query, each gets half the mass. Space is an ordinary
transcript token. Supervise only the final block's three local heads; keep the
fourth global head unconstrained for delayed marks. Inference is unchanged.
"""
from contextlib import contextmanager
import math
import numpy as np
import torch
from torch.nn import functional as F


def packed_labels(token_indices, character_count, points):
    """Return [ceil(points/8),2] token indices; -1 means unsupervised blank."""
    a = np.asarray(token_indices)
    if type(points) is not int or points < 1 or type(character_count) is not int or character_count < 1:
        raise ValueError('positive actual points and transcript length required')
    if a.ndim != 1 or len(a) != (points+3)//4 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('one integer token index per real four-index reader frame required')
    if (a < -1).any() or (a >= character_count).any():
        raise ValueError('blank=-1 or valid transcript token INDEX required')
    out = np.full(((points+7)//8, 2), -1, dtype=np.int64)
    out.flat[:len(a)] = a
    return out


class PackedAlignmentPool:
    """Packed TRAIN-only targets, explicitly reject any held/evaluation lookup."""
    def __init__(self, labels, train_ids, device='cpu'):
        if not train_ids or len(set(train_ids)) != len(train_ids) or set(labels) != set(train_ids):
            raise ValueError('exact unique TRAIN-only timing target scope required')
        self.index = {s:i for i,s in enumerate(train_ids)}
        self.lengths = {s:len(labels[s]) for s in train_ids}
        a = np.full((len(train_ids), max(self.lengths.values()), 2), -1, dtype=np.int64)
        for i,s in enumerate(train_ids):
            x = np.asarray(labels[s])
            if x.shape != (self.lengths[s],2) or not self.lengths[s] or not np.issubdtype(x.dtype,np.integer) or (x < -1).any():
                raise ValueError('nonempty L,2 packed integer targets required')
            a[i,:len(x)] = x
        self.labels = torch.tensor(a, dtype=torch.long, device=device)

    def select(self, ids):
        if not ids or any(s not in self.index for s in ids):
            raise ValueError('timing supervision restricted to actual TRAIN IDs')
        index = torch.tensor([self.index[s] for s in ids],device=self.labels.device)
        return self.labels.index_select(0,index)[:,:max(self.lengths[s] for s in ids)]


def attention_log_probs(module, query, key, *, attn_mask=None, key_padding_mask=None):
    """Differentiable QK readout beside unchanged need_weights=False SDPA forward.

    Supported contract is batch-first equal Q/K/V widths, no extra key/value
    slots, zero dropout. No sampled attention or changed forward kernel. Log-space
    softmax avoids underflow for labels far outside the soft Gaussian prior.
    """
    if not module.batch_first or not module._qkv_same_embed_dim or module.dropout != 0 or module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError('batch-first standard zero-dropout MHA contract required')
    if query.ndim != 3 or key.ndim != 3 or len(query) != len(key) or query.shape[-1] != module.embed_dim or key.shape[-1] != module.embed_dim:
        raise ValueError('B,L,E query/key contract required')
    b,l,e = query.shape; s = key.shape[1]; h = module.num_heads; d = e//h
    weight = module.in_proj_weight; bias = module.in_proj_bias
    q = F.linear(query,weight[:e],None if bias is None else bias[:e]).reshape(b,l,h,d).transpose(1,2)
    k = F.linear(key,weight[e:2*e],None if bias is None else bias[e:2*e]).reshape(b,s,h,d).transpose(1,2)
    logits = (q/math.sqrt(d)) @ k.transpose(-2,-1)
    if attn_mask is not None:
        if attn_mask.shape == (l,s): m = attn_mask[None,None]
        elif attn_mask.shape == (b*h,l,s): m = attn_mask.reshape(b,h,l,s)
        else: raise ValueError('supported MHA L,S or B*H,L,S mask required')
        logits = logits.masked_fill(m,float('-inf')) if m.dtype == torch.bool else logits+m
    if key_padding_mask is not None:
        if key_padding_mask.shape != (b,s) or key_padding_mask.dtype != torch.bool:
            raise ValueError('Boolean B,S padding mask required')
        logits = logits.masked_fill(key_padding_mask[:,None,None],float('-inf'))
    return logits.log_softmax(-1)


@contextmanager
def capture_last_alignment(model):
    """Capture only while requested; no model state/hooks survive context exit."""
    captured = {}
    def before(module,args,kwargs):
        if kwargs.get('need_weights',True):
            raise ValueError('original need_weights=False forward must remain unchanged')
        captured['log_probs'] = attention_log_probs(module,args[0],args[1],
            attn_mask=kwargs.get('attn_mask'),key_padding_mask=kwargs.get('key_padding_mask'))
    hook = model.blocks[-1].cross_attention.register_forward_pre_hook(before,with_kwargs=True)
    try: yield captured
    finally: hook.remove()


def alignment_loss(log_probs, labels, query_mask, text, *, validate=True):
    """CE / valid query / local head. BOS included in softmax, not in targets.

    Equal query weights; one emitted label gets mass1, two emitted labels mass.5
    each. Identical repeated frame labels correctly sum to1. Blanks/padding are
    excluded. Fourth/global head is NOT supervised.
    """
    b,h,l,s = log_probs.shape
    if h < 2 or labels.shape != (b,l,2) or labels.dtype != torch.long or query_mask.shape != (b,l) or query_mask.dtype != torch.bool or text.shape != (b,s-1):
        raise ValueError('compatible B,H,L,S log probs, B,L,2 labels, masks and text required')
    valid = labels >= 0
    if validate and ((labels < -1).any() or ((valid.any(-1)) & ~query_mask).any()):
        raise ValueError('no timing labels on padded trajectory queries')
    if validate and (labels >= s-1).any(): raise ValueError('timing label outside transcript tensor')
    indices = labels.clamp_min(0)
    if validate:
        token_valid = (text >= 0).gather(1,indices.reshape(b,-1)).reshape(b,l,2)
        if (valid & ~token_valid).any(): raise ValueError('timing labels cannot point at text padding')
    counts = valid.sum(-1); supervised = query_mask & (counts > 0)
    weights = valid.to(log_probs.dtype)/counts.clamp_min(1)[...,None]
    selected = log_probs[:,:h-1].gather(-1,(indices+1)[:,None].expand(-1,h-1,-1,-1))
    # where, not multiplication: padded -inf * zero would give NaN.
    selected = torch.where(valid[:,None],selected,0.)
    loss = -(selected*weights[:,None]).sum()/((h-1)*supervised.sum().clamp_min(1))
    if validate and not torch.isfinite(loss): raise FloatingPointError('nonfinite alignment auxiliary')
    return loss


def alignment_summary(log_probs, labels, query_mask, text):
    """Conditional-character attention means are descriptive, not causal proof."""
    loss = alignment_loss(log_probs,labels,query_mask,text)
    with torch.no_grad():
        probs = log_probs[:,:-1,:,1:].exp()
        expected = (probs*torch.arange(probs.shape[-1],device=probs.device)).sum(-1)/probs.sum(-1).clamp_min(1e-30)
        mean = expected.mean(1); valid = labels >= 0
        error = (mean[...,None]-labels).abs()[valid]
        mass = probs.gather(-1,labels.clamp_min(0)[:,None].expand(-1,probs.shape[1],-1,-1))
        counts = valid.sum(-1).clamp_min(1)
        target_mass = (mass.mean(1)*valid).sum(-1)/counts
        supervised = valid.any(-1)&query_mask
        return dict(cross_entropy=float(loss),supervised_queries=int(supervised.sum()),
            target_mass=float(target_mass[supervised].mean()) if supervised.any() else None,
            mean_index_error={key:float(torch.quantile(error,q)) for key,q in [('median',.5),('p90',.9),('p99',.99)]} if len(error) else None)
