"""Fresh positional-contract ablation for the standalone mapper, NOT InkDiT.

Absolute queries cannot change their positional features when a caller changes
the output window. Self-attention context can still change; this is not a claim
of full prefix invariance, an autoregressive model, or semantic composition.
"""
import torch
from .generation_alignment import AlignedWriterDenoiser, gaussian_bias
from .latent_diffusion import positions


class PositionContractWriter(AlignedWriterDenoiser):
    def __init__(self, position_policy='relative100', **config):
        if position_policy not in ('relative100', 'absolute'):
            raise ValueError('explicit relative100/absolute positional contract required')
        super().__init__(**config)
        if not self.alignment:
            raise ValueError('both positional arms retain the soft Gaussian prior')
        self.position_policy = position_policy
        self.config = dict(self.config, position_policy=position_policy)

    def query_positions(self, length, mask, dtype):
        p = torch.arange(length, device=mask.device, dtype=dtype)
        absolute = positions(p, self.config['width'])[None].expand(len(mask), -1, -1)
        if self.position_policy == 'absolute':
            return absolute
        relative = p[None] / (mask.sum(1) - 1).clamp_min(1)[:, None]
        return absolute + positions(relative * 100, self.config['width'])

    def forward(self, x, time, text, mask, drop_text=None, writer_ids=None):
        # The control delegates exactly to the previous soft model.
        if self.position_policy == 'relative100':
            return super().forward(x, time, text, mask, drop_text, writer_ids)
        if writer_ids is None or writer_ids.shape != (len(x),) or writer_ids.dtype != torch.long:
            raise ValueError('explicit B-long writer IDs required')
        width = self.config['width']; b, length, _ = x.shape
        drop = torch.zeros(b, device=x.device, dtype=torch.bool) if drop_text is None else drop_text
        tokens = torch.cat((torch.ones(b, 1, dtype=torch.long, device=x.device),
                            torch.where(text >= 0, text + 2, 0)), 1)
        text_mask = tokens != 0
        text_mask[:, 1:] &= ~drop[:, None]
        tokens[:, 1:] = torch.where(drop[:, None], 0, tokens[:, 1:])
        memory = (self.text(tokens) + self.text_pe_scale * positions(
            torch.arange(tokens.shape[1], device=x.device, dtype=x.dtype), width)[None]
        ).masked_fill(~text_mask[..., None], 0.)
        x = self.project(x.masked_fill(~mask[..., None], 0.)) + self.ink_pe_scale * self.query_positions(length, mask, x.dtype)
        x = x + self.time(positions(time.to(x.dtype) * 1000, width))[:, None] + self.writer(writer_ids)[:, None]
        x = x.masked_fill(~mask[..., None], 0.)
        bias = gaussian_bias(text_mask, length, self.config['heads'], self.blocks_per_character,
                             sigma=self.prior_sigma, cap=self.prior_cap, dtype=x.dtype)
        for block in self.blocks:
            h = block.norms[0](x)
            x = x + block.self_attention(h, h, h, key_padding_mask=~mask, need_weights=False)[0]
            h = block.norms[1](x)
            x = x + block.cross_attention(h, memory, memory, attn_mask=bias, need_weights=False)[0]
            x = (x + block.ff(block.norms[2](x))).masked_fill(~mask[..., None], 0.)
        return self.final(x).masked_fill(~mask[..., None], 0.)


def contract_lr(step):
    """Same 24000-update fresh schedule then matched 24000-update low-LR phase."""
    from .generation_composition import learning_rate
    if type(step) is not int or not 1 <= step <= 48000:
        raise ValueError('bounded 1..48000 actual update required')
    return learning_rate(step, 24000) if step <= 24000 else 1e-5
