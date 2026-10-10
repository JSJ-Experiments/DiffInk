"""Source-only discrete counterfactuals for the untrained motion56 topology path.

A nonzero straight-through gradient is not evidence that its suggested hard
decision decreases CTC or preserves ink. Compare finite hard flips with the
local surrogate separately; never shorten the supervised TRAIN prefix.
No training, sampler, loss-policy or immutable experiment changes.
"""
import torch


def boundary_candidates(states, per_class=2):
    """Deterministic index-spaced probes, not arc-length or time-spaced probes."""
    if states.ndim != 1 or len(states) < 3 or per_class < 1:
        raise ValueError("nonempty 1D pen states and positive probe count required")
    if not torch.isin(states, states.new_tensor([0, 1, 2])).all():
        raise ValueError("three-state pen labels required")
    result = []
    for label in (0, 1):
        indices = torch.where(states[:-1] == label)[0]
        indices = indices[indices > 0]  # exclude anchor and final EOC
        if len(indices):
            picks = torch.linspace(0, len(indices)-1, min(per_class, len(indices))).round().long()
            result.extend(int(indices[i]) for i in picks)
    return sorted(set(result))


def flip_probe(reader, raw, text, points, index, new_state, *, content_loss, hard_forward):
    """Frozen-field single flip: exact hard CTC change vs local logit derivative.

    Explicit callbacks keep this diagnostic independent of experimental runners.
    Callers supply their actual content_loss(reader,raw,stats,text,lengths,joint)
    and hard_forward(raw,joint) implementation; it is never reimplemented here.
    Motion is held fixed, including inactive branches. A branch switch can
    displace every later XY even though only one hard pen label changed.
    raw stores unwhitened motion56. Identity stats here are a coordinate-system
    choice for this input-gradient test, NOT fitted production TRAIN moments.
    Both CTC forwards retain the actual TRAIN length, including interior EOC.
    """
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 56 or not torch.isfinite(raw).all():
        raise ValueError("finite single-line B=1,L,56 raw motion required")
    if type(points) is not int or not 2 < points <= 8*raw.shape[1]:
        raise ValueError("real point length within packed capacity required")
    if type(index) is not int or not 0 < index < points-1 or type(new_state) is not int or new_state not in (0,1,2):
        raise ValueError("internal real-point flip and valid new pen class required")
    before = raw.detach().clone().requires_grad_(True)
    old = int(before.reshape(1, -1, 7)[0, index, 4:].argmax())
    if old == new_state:
        raise ValueError("counterfactual must change the hard class")
    stats = dict(mean=raw.new_zeros(56), std=raw.new_ones(56))
    lengths = torch.tensor([points], dtype=torch.long, device=raw.device)
    loss = content_loss(reader, before, stats, text, lengths, joint=True)
    grad = torch.autograd.grad(loss, before)[0].reshape(1, -1, 7)[0, index, 4:]
    after = raw.detach().clone()
    scores = after.reshape(1, -1, 7)[0, index, 4:]
    displacement = torch.nn.functional.one_hot(torch.tensor(new_state, device=raw.device), 3).to(raw)-scores
    scores.copy_(scores+displacement)
    with torch.no_grad():
        changed = content_loss(reader, after, stats, text, lengths, joint=False)
        q0 = hard_forward(raw, False).reshape(-1,5)[:points]
        q1 = hard_forward(after, False).reshape(-1,5)[:points]
        difference = q1[:,:2]-q0[:,:2]
        labels = q1[:,2:].argmax(-1)
        eoc = torch.where(labels==2)[0]
        retained = int(eoc[0])+1 if len(eoc) else points
    actual = float(changed-loss.detach())
    predicted = float((grad*displacement).sum())
    return dict(index=index,old_state=old,new_state=new_state,
        ctc_before=float(loss.detach()),ctc_after=float(changed),exact_ctc_delta=actual,
        surrogate_directional_ctc_delta=predicted,
        sign_agrees=None if abs(actual)<1e-9 or abs(predicted)<1e-12 else (actual>0)==(predicted>0),
        xy_rmse=float(difference.square().mean().sqrt()),
        suffix_shift_norm=float(difference[index+1].norm()),
        free_first_eoc_points=retained,free_retained_fraction=retained/points,
        hard_internal_pen_lifts=int((labels[:-1]==1).sum()),
        fixed_train_prefix_points=points,
        caveat="local biased logit surrogate versus finite hard flip; not a valid infinitesimal argmax derivative")
