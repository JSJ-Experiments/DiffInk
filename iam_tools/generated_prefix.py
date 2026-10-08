"""Differentiable own-history rollout for paired trajectory-capacity supervision.

No ground-truth trajectory fields reach this graph. The block budget is a
TRAINING unroll bound, not a duration predictor or a free-generation endpoint.
It deliberately continues past predicted EOC so early false stops get supervised;
the genuine generator still stops at its learned first EOC and uses a common cap.
History inputs/hard pen choices are detached; hidden-state BPTT remains active.
Paired index-time targets are a capacity diagnostic, not an unbiased likelihood
objective for different valid handwritten realizations of the same text.
"""
import torch
from .history_rollin import history_forward
from .autoregressive_strokes import reconstruction_terms
from .cumulative_xy import cumulative_xy_loss, anchor_coefficient


def own_prefix_forward(model, text, writer_ids, blocks):
    if type(blocks) is not int or not 1 <= blocks <= 256:
        raise ValueError('bounded positive common training unroll budget required')
    feedback = model.writer.weight.new_zeros(len(text), blocks, 40)
    transitions = torch.ones(len(text), blocks-1, dtype=torch.bool, device=feedback.device)
    return history_forward(model, feedback, text, writer_ids, transitions, transitions)


def paired_losses(model, feedback, text, writer_ids, targets, states, real_mask, cfg):
    if feedback.shape[:2] != real_mask.shape[:2]:
        raise ValueError('same training unroll extent for paired graphs required')
    zero = torch.zeros(len(text), feedback.shape[1]-1, dtype=torch.bool, device=feedback.device)
    pred, pen, trace = history_forward(model, feedback, text, writer_ids, zero, zero)
    terms = reconstruction_terms(pred, pen, targets, states, real_mask, pred.new_tensor(cfg['pen_weights']))
    teacher_xy = cumulative_xy_loss(pred, targets, real_mask, cfg['offset_stats'])
    base = terms['offset'] + cfg['pen_weight']*terms['pen'] + cfg['teacher_anchor_weight']*teacher_xy
    own, own_pen, own_trace = own_prefix_forward(model, text, writer_ids, feedback.shape[1])
    own_xy = cumulative_xy_loss(own, targets, real_mask, cfg['offset_stats'])
    own_terms = reconstruction_terms(own, own_pen, targets, states, real_mask, pred.new_tensor(cfg['pen_weights']))
    return dict(base=base, teacher_offset=terms['offset'], teacher_pen=terms['pen'], teacher_xy=teacher_xy,
                own_xy=own_xy, own_pen=own_terms['pen'], own_offset_diagnostic=own_terms['offset'],
                teacher_trace=trace, own_trace=own_trace)


def auxiliary_coefficients(base_norm, own_xy_norm, own_pen_norm):
    return dict(xy=anchor_coefficient(base_norm, own_xy_norm, .25),
                pen=anchor_coefficient(base_norm, own_pen_norm, .10))


def free_stop_rows(generated, ids, texts):
    """Token-clock state at actual learned stop/cap; NOT verified glyph coverage."""
    if len(ids)!=len(texts) or len(ids)!=len(generated['stops']):
        raise ValueError('same actual generated batch/sample/text scope required')
    rows=[]
    for j,(sid,text) in enumerate(zip(ids,texts)):
        n=int(generated['stops'][j]);block=(n-1)//8;tr=generated['traces']
        center=float(tr['center'][j,block]);sigma=float(tr['sigma'][j,block])
        rows.append(dict(sample_id=sid,generated_points=n,found_eoc=bool(generated['found_eoc'][j]),
                         text_characters=len(text),center_at_learned_stop_or_cap=center,
                         sigma_at_learned_stop_or_cap=sigma,eos_attention_at_stop_or_cap=float(tr['attention'][j,block,len(text)]),
                         remaining_characters_by_clock=len(text)-center,clock_is_not_verified_glyph_alignment=True,
                         no_source_history=True,no_source_length=True))
    return rows
