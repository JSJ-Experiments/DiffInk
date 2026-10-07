"""Research-only features for the initialized polyphase40 transport, NOT generic z.

These adapters change ONLY OCR input. Codec trajectories/data/normalization do not
change. Head-only checkpoints deliberately cannot be loaded as standard VAEs.
"""
import torch


def unpack(x, mask, points_per_frame=8):
    if type(points_per_frame) is not int or points_per_frame not in (2,4,8):raise ValueError('transport reader supports2,4 or8 points/frame')
    channels=5*points_per_frame
    if x.ndim!=3 or x.shape[1]<channels or mask.shape!=(x.shape[0],x.shape[2]) or mask.dtype!=torch.bool:
        raise ValueError('polyphase40 features and Boolean latent valid mask required')
    if not mask.any(1).all():raise ValueError('nonempty line required')
    clean=x.masked_fill(~mask[:,None],0.)
    fields=clean[:,:channels].reshape(x.shape[0],points_per_frame,5,x.shape[2])
    states=fields[:,:,2:].argmax(2).permute(0,2,1).reshape(x.shape[0],-1)
    temporal=mask.repeat_interleave(points_per_frame,dim=1)
    eos=(states==2)&temporal
    if not eos.any(1).all():raise ValueError('transport requires line-final EOC, not opaque learned latents')
    real=temporal&((eos.cumsum(1)-eos.long())==0)
    real=real.reshape(x.shape[0],x.shape[2],points_per_frame).permute(0,2,1)
    return fields,real


def relative_x(fields):
    """Invertible up to horizontal translation; all within-frame X shape preserved.

    Phase0 is current2/4/8-point block's firstX minus previous block's firstX (first=0).
    Remaining phases are offsets from current firstX. These are index displacements,
    NOT velocity. Includes pen jumps; separate pen fields tell the head about them.
    """
    anchor=fields[:,0,0];first=torch.cat((torch.zeros_like(anchor[:,:1]),anchor[:,1:]-anchor[:,:-1]),dim=1)
    rest=fields[:,1:,0]-anchor[:,None]
    return torch.cat((first[:,None],rest),dim=1)


def transform(x,mask,mode='global_raw',stats=None,points_per_frame=8):
    if mode not in ('global_raw','global_scaled','relative_scaled'):raise ValueError('unknown transport OCR mode')
    fields,real=unpack(x,mask,points_per_frame)
    xy=fields[:,:,:2].clone()
    if mode=='relative_scaled':xy[:,:,0]=relative_x(fields)
    if mode!='global_raw' and stats is not None:
        mean=x.new_tensor(stats['mean'])[None,None,:,None];std=x.new_tensor(stats['std'])[None,None,:,None]
        xy=(xy-mean)/std
    # Synthetic post-EOC phases are excluded, not turned into large -lineWidth
    # displacements by local centering. No unused posterior noise enters OCR.
    packed=torch.cat((xy,fields[:,:,2:]),dim=2).masked_fill(~real[:,:,None],0.)
    return torch.cat((packed.reshape(x.shape[0],5*points_per_frame,x.shape[2]),torch.zeros_like(x[:,5*points_per_frame:])),dim=1)


@torch.no_grad()
def fit_stats(cache,train_ids,mode,points_per_frame=8):
    if not train_ids or len(train_ids)!=len(set(train_ids)):raise ValueError('unique nonempty TRAIN IDs required')
    values=[]
    for sid in train_ids:
        c=cache[sid];f,real=unpack(c['mu'],c['mask'],points_per_frame);xy=f[:,:,:2].clone()
        if mode=='relative_scaled':xy[:,:,0]=relative_x(f)
        values.append(xy.permute(0,1,3,2)[real].double())
    values=torch.cat(values);mean=values.mean(0);std=values.std(0,unbiased=False).clamp_min(.01)
    return dict(mean=mean.cpu().tolist(),std=std.cpu().tolist(),real_points=len(values),
        source='TRAIN cached means only, valid real phases; population XY-axis moments; std floor0.01',mode=mode)


def local_attention_mask(valid,heads,radius):
    """True=blocked MHA [B*H,T,T], finite dummy queries without global leakage.

    Valid queries see ONLY +/-radius. Padded queries may see key0, which is valid
    for right-padded inputs. No real query sees padding. This avoids all-masked
    padded rows producing NaNs, and never gives real queries a global key0 escape.
    """
    if valid.ndim!=2 or valid.dtype!=torch.bool or not valid.any(1).all() or not isinstance(radius,int) or radius<0 or heads<1:
        raise ValueError('nonempty Boolean valid mask and nonnegative integer radius/heads required')
    if ((~valid[:,:-1])&valid[:,1:]).any():raise ValueError('right-padded valid prefixes required')
    b,t=valid.shape;indices=torch.arange(t,device=valid.device)
    blocked=(indices[:,None]-indices[None]).abs()>radius
    blocked=blocked[None].expand(b,-1,-1)|~valid[:,None,:]
    dummy=torch.ones_like(blocked);dummy[:,:,0]=False
    blocked=torch.where(valid[:,:,None],blocked,dummy)
    return blocked[:,None].expand(-1,heads,-1,-1).reshape(b*heads,t,t)


def make_head(cfg,num_classes,mode='global_raw',stats=None,radius=None,seed=42,points_per_frame=8):
    from model.ocr import ChineseHandwritingOCR
    class TransportOCR(ChineseHandwritingOCR):
        def forward(self,x,padding_mask=None,attention_mask=None):
            valid=torch.ones(x.shape[0],x.shape[2],dtype=torch.bool,device=x.device) if padding_mask is None else ~padding_mask
            features=transform(x,valid,mode,stats,points_per_frame)
            if radius is not None:attention_mask=local_attention_mask(valid,cfg['ocr_num_heads'],radius)
            return super().forward(features,padding_mask,attention_mask)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        head=TransportOCR(cfg['latent_dim'],cfg['ocr_hidden_dim'],cfg['ocr_num_heads'],cfg['ocr_num_layers'],num_classes,dropout=.1)
        with torch.no_grad():head.output_fc.bias[0]=0.
    head.ctc.zero_infinity=False
    return head
