"""Identity-initialized local OCR residual; never transforms rendered trajectories."""
import copy
import torch
from .ocr_context_features import transform, unpack
from .ocr_context_study import tensor_digest
from .ocr_pool_expansion import state_digest

SPEC = dict(active_fields=20, hidden=64, kernel=5, seed=613,
            placement='relative-scaled 4-point fields before parent input projection',
            initialization='zero final projection; exact parent function',
            geometry='no trajectory/data change; acquisition-index neighborhood, NOT spatial arc length')


class LocalContext(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(20, 64, 5, padding=2)
        self.out = torch.nn.Conv1d(64, 20, 1, bias=False)
        torch.nn.init.zeros_(self.out.weight)

    def forward(self, features, valid):
        clean = features[:, :20].masked_fill(~valid[:, None], 0.)
        hidden = torch.nn.functional.gelu(self.conv(clean))
        hidden = hidden.masked_fill(~valid[:, None], 0.)
        return self.out(hidden).masked_fill(~valid[:, None], 0.)


def attach_context(head, optimizer=None):
    """Add identical zero-output residual to both arms AFTER strict parent restore.

    Parent AdamW moments, parameter indices, counters and RNG remain untouched.
    New branch gets its own group at exactly the inherited hyperparameters.
    """
    if hasattr(head, 'local_context'):
        raise ValueError('context already attached')
    before = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(SPEC['seed']).get_state())
        branch = LocalContext()
    head.local_context = branch.to(head.input_proj.weight.device)
    if optimizer is not None:
        group = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
        optimizer.add_param_group(dict(group, params=list(branch.parameters())))
        after = copy.deepcopy(optimizer.state_dict())
        after['param_groups'].pop()
        if state_digest(before) != state_digest(after):
            raise AssertionError('parent optimizer changed when attaching local branch')
    return dict(spec=SPEC, parent_optimizer_unchanged=True,
                effective_head_sha256=tensor_digest(head.state_dict()),
                extra_parameters=sum(p.numel() for p in branch.parameters()))


def make_head(cfg, num_classes, stats, context=False, seed=42, attach=True):
    from model.ocr import ChineseHandwritingOCR

    class LocalOCR(ChineseHandwritingOCR):
        def forward(self, x, padding_mask=None, attention_mask=None):
            valid = (torch.ones(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
                     if padding_mask is None else ~padding_mask)
            features = transform(x, valid, 'relative_scaled', stats, 4)
            if hasattr(self, 'local_context'):
                _, real = unpack(x, valid, 4)
                delta = self.local_context(features, valid).reshape(x.shape[0], 4, 5, x.shape[2])
                delta = delta.masked_fill(~real[:, :, None], 0.).reshape(x.shape[0], 20, x.shape[2])
                # Both arms execute the same deterministic branch; only this gate differs.
                features = torch.cat((features[:, :20] + float(context) * delta,
                                      features[:, 20:]), dim=1)
            return super().forward(features, padding_mask, attention_mask)

    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        head = LocalOCR(cfg['latent_dim'], cfg['ocr_hidden_dim'], cfg['ocr_num_heads'],
                        cfg['ocr_num_layers'], num_classes, dropout=.1)
        with torch.no_grad():
            head.output_fc.bias[0] = 0.
    head.ctc.zero_infinity = False
    if attach:
        attach_context(head)
    return head
