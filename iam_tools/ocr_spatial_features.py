"""Optional normalized horizontal location for the4-point transport OCR only."""
import torch
from .ocr_context_features import unpack,transform
from .ocr_context_study import tensor_digest


def normalized_x(x,mask):
    """Each real phase (x-min_realX)/max(realXspan,.01); no labels/alignment.

    Spatial position, NOT acquisition time/velocity. Backward strokes stay backward.
    Synthetic/padded phases excluded from min/max and zeroed at output.
    """
    fields,real=unpack(x,mask,4);v=fields[:,:,0]
    lo=v.masked_fill(~real,float('inf')).amin((1,2));hi=v.masked_fill(~real,float('-inf')).amax((1,2))
    return ((v-lo[:,None,None])/(hi-lo).clamp_min(.01)[:,None,None]).masked_fill(~real,0.)


def features(x,mask,stats,spatial=False):
    y=transform(x,mask,'relative_scaled',stats,4)
    if x.shape[1]<24:raise ValueError('at least24 inputchannels for20local+4spatial fields required')
    if spatial:y[:,20:24]=normalized_x(x,mask)
    return y


def make_head(cfg,num_classes,stats,spatial=False,seed=42):
    from model.ocr import ChineseHandwritingOCR
    class SpatialOCR(ChineseHandwritingOCR):
        def forward(self,x,padding_mask=None,attention_mask=None):
            valid=torch.ones(x.shape[0],x.shape[2],dtype=torch.bool,device=x.device) if padding_mask is None else ~padding_mask
            return super().forward(features(x,valid,stats,spatial),padding_mask,attention_mask)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        head=SpatialOCR(cfg['latent_dim'],cfg['ocr_hidden_dim'],cfg['ocr_num_heads'],cfg['ocr_num_layers'],num_classes,dropout=.1)
        with torch.no_grad():head.output_fc.bias[0]=0.
    head.ctc.zero_infinity=False
    return head


def zero_unused_columns(head,optimizer):
    """Common intervention BOTH arms: zero previously unused input columns20:24.

    They were multiplied by zero under parent, hence no change in parent function.
    Assert existing moments exactly0, don't reset counters or any optimizer state.
    """
    weight=head.input_proj.weight;state=optimizer.state[weight]
    for k in ('exp_avg','exp_avg_sq'):
        if torch.count_nonzero(state[k][:,20:24]).item():raise ValueError('new positional columns were not unused in parent')
    before=weight.detach().clone()
    with torch.no_grad():weight[:,20:24].zero_()
    if not torch.equal(weight[:,:20],before[:,:20]) or not torch.equal(weight[:,24:],before[:,24:]):raise AssertionError('unrelated input weights changed')
    return dict(columns=[20,24],old_moments_exactly_zero=True,effective_head_sha256=tensor_digest(head.state_dict()),parent_function_unchanged='both old inputs and new-column weights zero atbaseline; verify exact step0 pairing')
