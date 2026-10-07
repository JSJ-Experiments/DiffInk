"""Explicit differentiable polyphase40→four-point frozen-GRU supervision.

NOT an adapter for opaque semantic latents. Actual point lengths are mandatory;
predicted pen/EOC fields remain inputs but cannot change the supervised prefix.
"""
from contextlib import contextmanager
from pathlib import Path
import torch
from .ocr_context_features import transform
from .ocr_frame_study import split_tensor
from .ocr_recurrent import make_head
from .pen_ab import file_sha

READER_REL='checkpoints/iam_ocr_recurrent_study/20261007-100011/bigru/head-best.pt'
READER_SHA='2ebac71d897920233c8399a38e1f16fce24075cfa722c7d8e588ab14e564100e'
CONTRACT='research initialized polyphase40→4point explicit-length frozen BiGRU; NOT opaque semantic z'


@contextmanager
def input_gradient_gru(head, enabled):
    """cuDNN needs training reserves for RNN input backward, even frozen weights.

    Use its training backend with dropout=0, NOT stochastic reader training.
    Restore flags even on exceptions; caller never updates reader parameters.
    GPU preflight checks logits/decoded-text parity versus normal eval forward.
    """
    training,dropout=head.rnn.training,head.rnn.dropout
    try:
        if enabled:head.rnn.train(True);head.rnn.dropout=0.
        yield
    finally:
        head.rnn.train(training);head.rnn.dropout=dropout


class PolyphaseGRUOCR(torch.nn.Module):
    requires_point_mask=True

    def __init__(self,head,stats):
        super().__init__();self.head=head;self.stats=stats
        self.head.eval().requires_grad_(False)

    def train(self,mode=True):
        # model.train() must not silently reintroduce reader dropout.
        super().train(False);self.head.eval();return self

    def framed(self,x,latent_mask,point_mask):
        if x.ndim!=3 or x.shape[1]<40 or point_mask is None:
            raise ValueError('known polyphase40 fields and explicit point_mask required')
        if point_mask.shape!=(x.shape[0],8*x.shape[2]) or point_mask.dtype!=torch.bool:
            raise ValueError('Boolean original-point mask must have shape B,8*T')
        if latent_mask is None:latent_mask=point_mask.reshape(x.shape[0],-1,8).any(-1)
        if latent_mask.shape!=(x.shape[0],x.shape[2]) or not torch.equal(latent_mask.bool(),point_mask.reshape(x.shape[0],-1,8).any(-1)):
            raise ValueError('original point/latent masks disagree')
        frames=point_mask.reshape(x.shape[0],-1,4).any(-1)
        fields=split_tensor(x.masked_fill(~latent_mask.bool()[:,None],0.),4)
        return transform(fields,frames,'relative_scaled',self.stats,4,point_mask=point_mask),frames

    def forward(self,x,padding_mask=None,point_mask=None):
        features,valid=self.framed(x,None if padding_mask is None else ~padding_mask,point_mask)
        with input_gradient_gru(self.head,torch.is_grad_enabled() and features.requires_grad):
            return self.head.logits_from_features(features,valid)

    def get_ocr_loss(self,x,labels,mask=None,point_mask=None):
        features,valid=self.framed(x,mask,point_mask)
        lengths=(labels!=-1).sum(1).long()
        repeats=((labels[:,1:]==labels[:,:-1])&(labels[:,1:]!=-1)).sum(1)
        if not (lengths>0).all() or (valid.sum(1)<lengths+repeats).any():
            raise ValueError('every joint CTC target must be nonempty and exactly feasible')
        if ((labels[:,:-1]==-1)&(labels[:,1:]!=-1)).any():raise ValueError('right-padded targets required')
        with input_gradient_gru(self.head,torch.is_grad_enabled() and features.requires_grad):
            logits=self.head.logits_from_features(features,valid)
        return self.head.ctc(logits.clamp(-30,30).log_softmax(2),labels+1,valid.sum(1).long().cpu(),lengths.cpu())


def load_reader(root,codec_cfg,device='cpu'):
    path=Path(root)/READER_REL
    if file_sha(path)!=READER_SHA:raise ValueError('canonical frozen GRU SHA changed')
    saved=torch.load(path,map_location='cpu',weights_only=True);cfg=saved['config']
    if cfg['cfg']!=codec_cfg or cfg['architecture']!='bigru' or cfg['points_per_frame']!=4 or saved['updates']!=8000:
        raise ValueError('reader architecture/codec/frame contract mismatch')
    head=make_head(codec_cfg,82,cfg['feature_stats'],seed=42)
    head.load_state_dict(saved['ocr_state_dict'],strict=True)
    return PolyphaseGRUOCR(head,cfg['feature_stats']).to(device).eval(),cfg
