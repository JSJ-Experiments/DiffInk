"""Length-independent absolute queries + causal hidden-query attention.

This is NOT teacher-forced/autoregressive stroke generation: input remains zero,
all text remains visible, and all hidden queries are evaluated in parallel.
Causal query context removes output-budget identity information from prefixes.
No new learned parameters; the noncausal control delegates byte-exact behavior.
"""
import torch
from .generation_position_contract import PositionContractWriter, contract_lr
from .generation_alignment import gaussian_bias
from .latent_diffusion import positions


class PrefixContractWriter(PositionContractWriter):
    def __init__(self, causal_queries=False, **config):
        if type(causal_queries) is not bool:
            raise ValueError('explicit Boolean causal_queries required')
        config.setdefault('position_policy', 'absolute')
        if config['position_policy'] != 'absolute':
            raise ValueError('prefix experiment requires absolute query positions')
        super().__init__(**config)
        self.causal_queries = causal_queries
        self.config = dict(self.config, causal_queries=causal_queries)

    def forward(self, x, time, text, mask, drop_text=None, writer_ids=None):
        # The control delegates exactly to the previous soft model.
        if not self.causal_queries:
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
        # MHA boolean True FORBIDS a key; SDPA has the opposite convention.
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        for block in self.blocks:
            h = block.norms[0](x)
            x = x + block.self_attention(h, h, h, key_padding_mask=~mask, attn_mask=causal, need_weights=False)[0]
            h = block.norms[1](x)
            x = x + block.cross_attention(h, memory, memory, attn_mask=bias, need_weights=False)[0]
            x = (x + block.ff(block.norms[2](x))).masked_fill(~mask[..., None], 0.)
        return self.final(x).masked_fill(~mask[..., None], 0.)

