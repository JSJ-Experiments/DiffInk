"""Controlled intra-block autoregression, separate from completed block forecast.

Coarse history/text clock stays at eight points. A point GRU predicts just the
next displacement/pen; teacher inputs are strictly shifted within each block.
No-feedback control keeps previous block history but hides points 0..6 from
predictions 1..7. Both arms have identical parameters and initialization.
Teacher point GRUs run as one fused batch over all blocks, not Python GRUCells.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .autoregressive_strokes import MonotonicStrokeWriter, decode_offsets


class PointFeedbackStrokeWriter(MonotonicStrokeWriter):
    def __init__(self, *args, point_feedback=True, **kwargs):
        if type(point_feedback) is not bool:
            raise ValueError('explicit point-feedback policy required')
        super().__init__(*args, **kwargs)
        self.config['point_feedback'] = point_feedback
        # Remove unused eight-future-point readout, not a pretrained transplant.
        del self.readout
        width = self.config['width']
        self.point_gru = nn.GRU(6, width, batch_first=True, dropout=0.)
        self.point_readout = nn.Linear(width, 5)
        nn.init.normal_(self.point_readout.weight, std=.005)
        nn.init.zeros_(self.point_readout.bias)

    def coarse_step(self, previous, state, memory, valid, writer, start):
        h, d, center, context = state
        h = self.history(torch.cat((previous, start[:, None], context, writer), -1), h)
        raw = self.clock(h)
        if self.config['adaptive']:
            advance = .01 + F.softplus(raw[:, 0])
            sigma = (.5 + F.softplus(raw[:, 1])).clamp_max(4.)
        else:
            advance = torch.full_like(center, self.config['initial_advance'])
            sigma = torch.ones_like(center)
        center = center + advance
        index = torch.arange(memory.shape[1], device=memory.device, dtype=memory.dtype)
        logits = (-.5*((index[None]-center[:, None])/sigma[:, None]).square()).masked_fill(~valid, float('-inf'))
        weights = logits.softmax(-1)
        context = torch.bmm(weights[:, None], memory).squeeze(1)
        d = self.decoder(torch.cat((h, context, writer), -1), d)
        return (h, d, center, context), dict(center=center, advance=advance, sigma=sigma, attention=weights)

    def teacher(self, feedback, text, writer_ids):
        if feedback.ndim != 3 or feedback.shape[-1] != 40 or not feedback.shape[1]:
            raise ValueError('B,L,40 offset/pen feedback required')
        memory, valid, writer = self.memory(text, writer_ids)
        state = self.initial_state(memory)
        previous = feedback.new_zeros(len(feedback), 40)
        coarse = []; traces = []
        for j in range(feedback.shape[1]):
            state, trace = self.coarse_step(previous, state, memory, valid, writer, feedback.new_full((len(feedback),), float(j == 0)))
            coarse.append(state[1]); traces.append(trace)
            previous = feedback[:, j]
        b, length, _ = feedback.shape
        fields = torch.cat((feedback[:, :, :16].reshape(b, length, 8, 2), feedback[:, :, 16:].reshape(b, length, 8, 3)), -1)
        last_previous_block = torch.cat((fields.new_zeros(b, 1, 5), fields[:, :-1, -1]), 1)
        shifted = torch.cat((last_previous_block[:, :, None], fields[:, :, :-1]), 2)
        if not self.config['point_feedback']:
            # Keep the SAME previous-block last point; only hide within-block feedback.
            shifted = torch.cat((shifted[:, :, :1], torch.zeros_like(shifted[:, :, 1:])), 2)
        bos = feedback.new_zeros(b, length, 8, 1)
        bos[:, 0, 0] = 1.
        point_inputs = torch.cat((shifted, bos), -1).reshape(b*length, 8, 6)
        initial = torch.stack(coarse, 1).reshape(b*length, -1)[None]
        decoded, _ = self.point_gru(point_inputs, initial)
        outputs = self.point_readout(decoded).reshape(b, length, 8, 5)
        return outputs[..., :2], outputs[..., 2:], {k:torch.stack([t[k] for t in traces], 1) for k in traces[0]}

    @torch.no_grad()
    def generate(self, text, writer_ids, stats, max_blocks=256):
        if self.training or type(max_blocks) is not int or not 1 <= max_blocks <= 256:
            raise ValueError('eval-mode bounded target-free generation required')
        memory, valid, writer = self.memory(text, writer_ids)
        state = self.initial_state(memory)
        previous = memory.new_zeros(len(text), 40)
        offsets = []; pens = []; traces = []
        stopped = torch.zeros(len(text), device=text.device, dtype=torch.bool)
        stops = torch.full((len(text),), max_blocks*8, device=text.device, dtype=torch.long)
        found = stopped.clone()
        for j in range(max_blocks):
            state, trace = self.coarse_step(previous, state, memory, valid, writer, memory.new_full((len(text),), float(j == 0)))
            hidden = state[1][None]
            prior_point = torch.cat((previous[:, 14:16], previous[:, 37:40]), -1)
            block_o = []; block_p = []
            for t in range(8):
                point_input = torch.cat((prior_point, memory.new_full((len(text), 1), float(j == 0 and t == 0))), -1)
                decoded, hidden = self.point_gru(point_input[:, None], hidden)
                output = self.point_readout(decoded[:, 0]); o = output[:, :2]; hard = output[:, 2:].argmax(-1)
                block_o.append(o); block_p.append(hard)
                prior_point = torch.cat((o, F.one_hot(hard, 3).to(o.dtype)), -1) if self.config['point_feedback'] else torch.zeros_like(prior_point)
            o = torch.stack(block_o, 1); hard = torch.stack(block_p, 1)
            offsets.append(decode_offsets(o, stats)); pens.append(hard); traces.append(trace)
            hits = hard == 2; has = hits.any(-1) & ~stopped
            first = hits.long().argmax(-1) + j*8 + 1
            stops = torch.where(has, first, stops); found |= has; stopped |= has
            previous = torch.cat((o.flatten(1), F.one_hot(hard, 3).to(o.dtype).flatten(1)), -1)
            if bool(stopped.all()):
                break
        delta = torch.stack(offsets, 1).reshape(len(text), -1, 2)
        states = torch.stack(pens, 1).reshape(len(text), -1)
        xy = delta.cumsum(1)
        return dict(points=torch.cat((xy, F.one_hot(states, 3).to(xy.dtype)), -1), stops=stops, found_eoc=found,
                    traces={k:torch.stack([t[k] for t in traces], 1) for k in traces[0]})
