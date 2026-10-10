import pytest
import torch
from iam_tools.topology_counterfactual import boundary_candidates, flip_probe


class Reader:
    def get_ocr_loss(self,x,text,mask,point_mask):
        q=x[:,:40].transpose(1,2).reshape(len(x),-1,5)
        return ((q[...,0].square()+q[...,1].square()+.2*q[...,3]+.4*q[...,4])*point_mask).sum()/point_mask.sum()


def source():
    raw=torch.zeros(1,16,7);raw[...,0]=.1;raw[...,4]=1
    raw[0,5,4:]=torch.tensor([0.,1.,0.])
    raw[0,12:,4:]=torch.tensor([0.,0.,1.])
    return raw.reshape(1,2,56)


# Deliberately tiny test doubles, NOT another motion56 implementation.
# The diagnostic's responsibility is querying callbacks without mutating inputs.
def hard_forward(raw,joint=False):
    f=raw.reshape(1,-1,7);scores=f[...,4:];state=scores.detach().argmax(-1)
    pen=torch.nn.functional.one_hot(state,3).to(raw)
    if joint:pen=pen+scores.softmax(-1)-scores.softmax(-1).detach()
    delta=f[...,0]*(state==0).to(raw)
    xy=torch.stack((delta.cumsum(-1),f[...,1]),-1)
    return torch.cat((xy,pen),-1).reshape(1,-1,40)


def content_loss(reader,raw,stats,text,lengths,joint=False):
    absolute=hard_forward(raw,joint)
    point_mask=torch.arange(raw.shape[1]*8)[None]<lengths[:,None]
    return reader.get_ocr_loss(torch.nn.functional.pad(absolute,(0,344)).transpose(1,2),
        text,point_mask.reshape(1,-1,8).any(-1),point_mask=point_mask)


def probe(*args):
    return flip_probe(*args,content_loss=content_loss,hard_forward=hard_forward)


def test_deterministic_candidates_exclude_anchor_final_and_eoc():
    s=torch.tensor([0,0,1,0,1,0,2])
    assert boundary_candidates(s)==[1,2,4,5]
    assert boundary_candidates(torch.tensor([0,1,2]))==[1]


def test_flip_is_actual_discrete_change_and_does_not_shorten_train_ctc():
    raw=source();saved=raw.clone()
    a=probe(Reader(),raw,torch.ones(1,1,dtype=torch.long),13,5,0)
    assert a["old_state"]==1 and a["new_state"]==0
    assert a["suffix_shift_norm"]>.09
    assert a["free_first_eoc_points"]==13 and a["fixed_train_prefix_points"]==13
    b=probe(Reader(),raw,torch.ones(1,1,dtype=torch.long),13,3,2)
    assert b["free_first_eoc_points"]==4
    assert b["free_retained_fraction"]==4/13
    torch.testing.assert_close(raw,saved,rtol=0,atol=0)


def test_padding_garbage_cannot_change_probe():
    raw=source();dirty=raw.clone();dirty.reshape(1,-1,7)[:,13:]=1e3
    a=probe(Reader(),raw,torch.ones(1,1,dtype=torch.long),13,3,1)
    b=probe(Reader(),dirty,torch.ones(1,1,dtype=torch.long),13,3,1)
    for key in ("ctc_before","ctc_after","exact_ctc_delta","surrogate_directional_ctc_delta","xy_rmse"):
        assert a[key]==b[key]


@pytest.mark.parametrize("points,index,label",[(17,3,1),(13,0,1),(13,12,1),(13,13,1),(13,3,3),(13,3,0)])
def test_invalid_or_noop_flips_fail(points,index,label):
    with pytest.raises(ValueError):
        probe(Reader(),source(),torch.ones(1,1,dtype=torch.long),points,index,label)
