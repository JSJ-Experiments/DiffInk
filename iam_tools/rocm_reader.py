"""Opt-in ATen GPU inference for a frozen reader on ROCm/ROCDXG.

Packed-GRU MIOpen inference failed with miopenStatusUnknownError on this WSL
workstation. Disabling the vendor RNN path keeps the exact weights/features and
uses PyTorch's native GPU operations. No catch/retry, CPU fallback, or OCR update.
The cuDNN flags interface also controls ROCm's MIOpen dispatch. Scope is confined
to the reader forward; restore flags, including when the reader raises.
"""
import torch
from torch import nn

class ATenFrozenReader(nn.Module):
    requires_point_mask=True
    def __init__(self,reader):
        super().__init__()
        if any(p.requires_grad for p in reader.parameters()):raise ValueError('only an explicitly frozen reader can use this inference wrapper')
        self.reader=reader;self.train(False)
    def train(self,mode=True):
        super().train(False);self.reader.eval();return self
    def forward(self,*args,**kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            return self.reader(*args,**kwargs)
