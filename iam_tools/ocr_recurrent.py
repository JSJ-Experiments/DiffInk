"""Packed bidirectional GRU reader for the same initialized4-point transport fields.

A deliberately separate research architecture/checkpoint; codec never changes.
"""
import torch
from .ocr_context_features import transform
SPEC=dict(hidden_per_direction=320,layers=3,dropout=.1,bidirectional=True,
          input_projection=384,packed_sequences=True,
          position_encoding='none; acquisition order supplied by recurrence',
          features='same4-point relative_scaled, TRAIN192 moments, no unusedlatentnoise')


def make_head(cfg,num_classes,stats,seed=42):
    from model.ocr import ChineseHandwritingOCR
    class RecurrentOCR(ChineseHandwritingOCR):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.input_proj=torch.nn.Linear(cfg['latent_dim'],cfg['ocr_hidden_dim'])
            self.rnn=torch.nn.GRU(cfg['ocr_hidden_dim'],SPEC['hidden_per_direction'],
                                 num_layers=SPEC['layers'],dropout=SPEC['dropout'],
                                 bidirectional=True,batch_first=True)
            self.output_fc=torch.nn.Linear(2*SPEC['hidden_per_direction'],num_classes)
            self.ctc=torch.nn.CTCLoss(blank=0,zero_infinity=False)
            with torch.no_grad():self.output_fc.bias.zero_()

        def forward(self,x,padding_mask=None,attention_mask=None):
            if attention_mask is not None:raise ValueError('packed recurrent reader has no attention-mask contract')
            valid=(torch.ones(x.shape[0],x.shape[2],dtype=torch.bool,device=x.device)
                   if padding_mask is None else ~padding_mask)
            if valid.dtype!=torch.bool or valid.shape!=(x.shape[0],x.shape[2]) or not valid.any(1).all():
                raise ValueError('nonempty Boolean right-padded input required')
            if ((~valid[:,:-1])&valid[:,1:]).any():raise ValueError('packed reader requires right-padded prefixes')
            features=transform(x,valid,'relative_scaled',stats,4)
            return self.logits_from_features(features,valid)

        def logits_from_features(self,features,valid):
            """Already transformed B,C,T fields; shared readout, same weights."""
            features=features.transpose(1,2)
            projected=self.input_proj(features).masked_fill(~valid[:,:,None],0.)
            packed=torch.nn.utils.rnn.pack_padded_sequence(projected,valid.sum(1).cpu(),batch_first=True,enforce_sorted=False)
            encoded,_=self.rnn(packed)
            encoded,_=torch.nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=features.shape[1])
            return self.output_fc(encoded).transpose(0,1)

    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        head=RecurrentOCR()
    return head
