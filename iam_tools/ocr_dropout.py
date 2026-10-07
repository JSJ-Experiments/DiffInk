"""OCR-only dropout intervention; state tensors and trajectory decoder never change."""
import math,torch
from .ocr_context_study import tensor_digest


def set_dropout(head,rate):
    if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<=rate<1:
        raise ValueError('finite OCR dropout probability in [0,1) required')
    before=tensor_digest(head.state_dict());changed=[]
    for name,module in head.named_modules():
        if isinstance(module,torch.nn.Dropout):
            changed.append(dict(name=name,attribute='p',previous=module.p));module.p=float(rate)
        elif isinstance(module,torch.nn.MultiheadAttention):
            changed.append(dict(name=name,attribute='dropout',previous=module.dropout));module.dropout=float(rate)
    if not changed:raise ValueError('Transformer OCR dropout modules required')
    if tensor_digest(head.state_dict())!=before:raise AssertionError('dropout intervention changed tensors')
    return dict(rate=float(rate),sites=changed,state_tensors_unchanged=True)
