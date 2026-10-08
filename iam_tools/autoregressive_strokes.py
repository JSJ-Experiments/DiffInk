"""Research autoregressive eight-point offset/pen baseline, NOT InkDiT.

Only previous trajectory blocks, text and writer reach each recurrent update.
A positive-increment Gaussian text window can adapt to generated history. Fixed
window is an otherwise same-weights control. No source length, teacher timing,
absolute query PE or target trajectory enters free generation. Character window
coordinates are token indices, not physical time or verified glyph boundaries.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MonotonicStrokeWriter(nn.Module):
    def __init__(self,vocab_size,writer_count,initial_advance,width=192,text_width=64,writer_width=16,adaptive=True):
        super().__init__()
        if type(adaptive) is not bool or min(vocab_size,writer_count,width,text_width,writer_width)<1 or not math.isfinite(initial_advance) or initial_advance<=.01:
            raise ValueError('positive dimensions/advance and explicit adaptive flag required')
        self.config=dict(vocab_size=vocab_size,writer_count=writer_count,initial_advance=initial_advance,width=width,text_width=text_width,writer_width=writer_width,adaptive=adaptive)
        self.text=nn.Embedding(vocab_size+2,text_width,padding_idx=0) # chars+1, EOS=vocab+1
        self.writer=nn.Embedding(writer_count,writer_width)
        self.history=nn.GRUCell(41+text_width+writer_width,width)
        self.clock=nn.Linear(width,2)
        self.decoder=nn.GRUCell(width+text_width+writer_width,width)
        self.readout=nn.Linear(width,40)
        nn.init.zeros_(self.clock.weight)
        with torch.no_grad():
            self.clock.bias.copy_(torch.tensor([math.log(math.expm1(initial_advance-.01)),math.log(math.expm1(.5))]))
        nn.init.normal_(self.readout.weight,std=.005);nn.init.zeros_(self.readout.bias)

    def memory(self,text,writer_ids):
        if text.ndim!=2 or text.dtype!=torch.long or writer_ids.shape!=(len(text),) or writer_ids.dtype!=torch.long:
            raise ValueError('B,S integer right-padded text and B writer IDs required')
        if (text < -1).any() or (text >= self.config['vocab_size']).any() or not (text>=0).any(1).all() or ((text[:,:-1]<0)&(text[:,1:]>=0)).any():
            raise ValueError('nonempty covered right-padded text required; NULL needs explicit token-mask intervention')
        lengths=(text>=0).sum(1);tokens=torch.zeros(len(text),text.shape[1]+1,dtype=torch.long,device=text.device)
        tokens[:,:text.shape[1]]=torch.where(text>=0,text+1,0);tokens.scatter_(1,lengths[:,None],self.config['vocab_size']+1)
        valid=torch.arange(tokens.shape[1],device=text.device)[None]<=lengths[:,None]
        return self.text(tokens),valid,self.writer(writer_ids)

    def initial_state(self,memory):
        b=len(memory);w=self.config['width'];tw=self.config['text_width']
        return (memory.new_zeros(b,w),memory.new_zeros(b,w),memory.new_full((b,),-self.config['initial_advance']/2),memory.new_zeros(b,tw))

    def step(self,previous,state,memory,valid,writer,start):
        h,d,center,context=state
        h=self.history(torch.cat((previous,start[:,None],context,writer),-1),h)
        raw=self.clock(h)
        if self.config['adaptive']:
            advance=.01+F.softplus(raw[:,0]);sigma=(.5+F.softplus(raw[:,1])).clamp_max(4.)
        else:
            advance=torch.full_like(center,self.config['initial_advance']);sigma=torch.ones_like(center)
        center=center+advance
        index=torch.arange(memory.shape[1],device=memory.device,dtype=memory.dtype)
        logits=(-.5*((index[None]-center[:,None])/sigma[:,None]).square()).masked_fill(~valid,float('-inf'))
        weights=logits.softmax(-1);context=torch.bmm(weights[:,None],memory).squeeze(1)
        d=self.decoder(torch.cat((h,context,writer),-1),d);out=self.readout(d)
        offsets=out[:,:16].reshape(len(out),8,2);pens=out[:,16:].reshape(len(out),8,3)
        return offsets,pens,(h,d,center,context),dict(center=center,advance=advance,sigma=sigma,attention=weights)

    def teacher(self,feedback,text,writer_ids):
        """Strictly shifted target history; prediction j cannot see target block j."""
        if feedback.ndim!=3 or feedback.shape[-1]!=40 or not feedback.shape[1]:raise ValueError('B,L,40 offset/pen feedback required')
        memory,valid,writer=self.memory(text,writer_ids);state=self.initial_state(memory);offsets=[];pens=[];traces=[]
        previous=feedback.new_zeros(len(feedback),40)
        for j in range(feedback.shape[1]):
            start=feedback.new_full((len(feedback),),float(j==0))
            o,p,state,trace=self.step(previous,state,memory,valid,writer,start)
            offsets.append(o);pens.append(p);traces.append(trace)
            previous=feedback[:,j]
        return torch.stack(offsets,1),torch.stack(pens,1),{key:torch.stack([t[key] for t in traces],1) for key in traces[0]}

    @torch.no_grad()
    def generate(self,text,writer_ids,stats,max_blocks=256):
        """Predicted offsets/hard pens fed back; learned firstEOC, common budget cap.

        No source trajectories/point masks/point counts/duration predictor input.
        Stop is reported separately from cap; no forced final EOC. Already-stopped
        samples are padded for batched compute but cannot change any saved prefix.
        """
        if self.training or type(max_blocks) is not int or not 1<=max_blocks<=256:raise ValueError('eval-mode bounded target-free generation required')
        memory,valid,writer=self.memory(text,writer_ids);state=self.initial_state(memory)
        previous=memory.new_zeros(len(text),40);offsets=[];pens=[];traces=[]
        stopped=torch.zeros(len(text),device=text.device,dtype=torch.bool);stops=torch.full((len(text),),max_blocks*8,device=text.device,dtype=torch.long);found=stopped.clone()
        for j in range(max_blocks):
            o,p,state,t=self.step(previous,state,memory,valid,writer,memory.new_full((len(text),),float(j==0)))
            hard=p.argmax(-1);real=decode_offsets(o,stats)
            offsets.append(real);pens.append(hard);traces.append(t)
            hits=hard==2;has=hits.any(-1)&~stopped;first=hits.long().argmax(-1)+j*8+1
            stops=torch.where(has,first,stops);found|=has;stopped|=has
            previous=torch.cat((o.reshape(len(text),16),F.one_hot(hard,3).to(o.dtype).reshape(len(text),24)),-1)
            if bool(stopped.all()):break
        delta=torch.stack(offsets,1).reshape(len(text),-1,2);states=torch.stack(pens,1).reshape(len(text),-1)
        xy=delta.cumsum(1);points=torch.cat((xy,F.one_hot(states,3).to(xy.dtype)),-1)
        return dict(points=points,stops=stops,found_eoc=found,traces={key:torch.stack([t[key] for t in traces],1) for key in traces[0]})


def offset_fields(points):
    if points.ndim!=2 or points.shape[-1]!=5 or not len(points) or not torch.isfinite(points).all():raise ValueError('nonempty finite N,5 absolute XY and one-hot pen fields required')
    states=points[:,2:].argmax(-1)
    if not torch.allclose(points[:,2:],F.one_hot(states,3).to(points.dtype)) or states[-1]!=2 or (states[:-1]==2).any():raise ValueError('English final-only EOC and one-hot pen contract required')
    delta=points[:,:2]-torch.cat((torch.zeros_like(points[:1,:2]),points[:-1,:2]),0)
    return delta,states


def fit_offset_stats(targets,ids):
    if not ids or len(set(ids))!=len(ids) or set(targets)!=set(ids):raise ValueError('exact unique TRAIN-only offset scope required')
    values=torch.cat([offset_fields(targets[s])[0] for s in ids]).double()
    mean=values.mean(0);std=values.std(0,unbiased=False).clamp_min(.01)
    return dict(train_ids=list(ids),mean=mean.tolist(),std=std.tolist(),real_points=len(values),definition='TRAIN chronological point displacements including pen-up jumps and origin-to-first-point; axis mean/populationstd floor.01; NOT physical velocity')


def encode_offsets(delta,stats):return (delta-delta.new_tensor(stats['mean']))/delta.new_tensor(stats['std'])
def decode_offsets(delta,stats):return delta*delta.new_tensor(stats['std'])+delta.new_tensor(stats['mean'])


class StrokePool:
    def __init__(self,targets,records,vocab,ids,stats,device='cpu'):
        if not ids or len(set(ids))!=len(ids):raise ValueError('unique nonempty bounded samples required')
        self.ids=list(ids);self.index={s:j for j,s in enumerate(ids)};self.lengths={s:(len(targets[s])+7)//8 for s in ids};self.characters={s:len(records[s]['text']) for s in ids}
        n=max(self.lengths.values());s=max(self.characters.values());b=len(ids)
        self.feedback=torch.zeros(b,n,40,device=device);self.offsets=torch.zeros(b,n,8,2,device=device);self.pens=torch.full((b,n,8),2,dtype=torch.long,device=device)
        self.mask=torch.zeros(b,n,8,dtype=torch.bool,device=device);self.text=torch.full((b,s),-1,dtype=torch.long,device=device)
        for j,sid in enumerate(ids):
            q=torch.as_tensor(targets[sid],device=device,dtype=torch.float32);d,p=offset_fields(q);count=len(q);length=self.lengths[sid]
            self.offsets[j,:length].reshape(-1,2)[:count]=encode_offsets(d,stats);self.pens[j,:length].reshape(-1)[:count]=p;self.mask[j,:length].reshape(-1)[:count]=True
            self.text[j,:self.characters[sid]]=torch.tensor([vocab.index(c) for c in records[sid]['text']],device=device)
        self.feedback=torch.cat((self.offsets.flatten(2),F.one_hot(self.pens,3).float().flatten(2)),-1)
        self.feedback.masked_fill_(~self.mask.any(-1)[...,None],0.)

    def select(self,ids):
        if not ids or any(s not in self.index for s in ids):raise ValueError('known nonempty sample scope required')
        index=torch.tensor([self.index[s] for s in ids],device=self.feedback.device);n=max(self.lengths[s] for s in ids);s=max(self.characters[i] for i in ids)
        return tuple(x.index_select(0,index)[:,:n] for x in [self.feedback,self.offsets,self.pens,self.mask])+(self.text.index_select(0,index)[:,:s],)


def reconstruction_terms(offsets,pens,target,states,mask,weights):
    if offsets.shape!=target.shape or offsets.shape!=(*mask.shape,2) or pens.shape!=(*mask.shape,3) or states.shape!=mask.shape or mask.dtype!=torch.bool:raise ValueError('matching B,L,8 coordinate/pen targets and real mask required')
    real=(offsets-target)[mask];mse=real.square().mean()
    ce=F.cross_entropy(pens[mask],states[mask],reduction='none');pen=(weights[states[mask]]*(1-ce.neg().exp()).square()*ce).mean()
    return dict(offset=mse,pen=pen)
