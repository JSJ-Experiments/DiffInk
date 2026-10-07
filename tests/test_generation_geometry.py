import unittest
import torch
from iam_tools.generation_geometry import physical_terms,coefficients

class GenerationGeometryTests(unittest.TestCase):
    def fixture(self):
        q=torch.zeros(1,2,384);fields=torch.zeros(1,16,5)
        fields[0,:,0]=torch.arange(16);fields[0,:,2]=1;fields[0,3,2]=0;fields[0,3,3]=1
        q[...,:40]=fields.reshape(1,2,40)
        mask=torch.arange(16)[None]<8
        stats={'mean':torch.zeros(384),'std':torch.ones(384)}
        return q,mask,stats
    def test_identical(self):
        q,m,s=self.fixture();r=physical_terms(q,q,s,m,codec_contract='polyphase40');self.assertEqual(float(r['xy']+r['first_difference']),0)
    def test_translation_not_smoothing(self):
        q,m,s=self.fixture();p=q.clone();p[...,0:40:5]+=3;p[...,1:40:5]+=3
        r=physical_terms(p,q,s,m,codec_contract='polyphase40');self.assertEqual(float(r['xy']),9);self.assertEqual(float(r['first_difference']),0)
    def test_pen_jump_excluded(self):
        q,m,s=self.fixture();p=q.clone();p[:,0,20:40:5]+=3;p[:,0,21:40:5]+=3
        r=physical_terms(p,q,s,m,codec_contract='polyphase40');self.assertEqual(float(r['first_difference']),0)
    def test_padding_excluded(self):
        q,m,s=self.fixture();p=q.clone();p[:,1,:40]=10000
        r=physical_terms(p,q,s,m,codec_contract='polyphase40');self.assertEqual(float(r['xy']+r['first_difference']),0)
    def test_gradient(self):
        q,m,s=self.fixture();p=(q+.1).requires_grad_();r=physical_terms(p,q,s,m,codec_contract='polyphase40');sum(r.values()).backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertEqual(float(p.grad[:,1].abs().sum()),0)
    def test_contract_guard(self):
        q,m,s=self.fixture()
        with self.assertRaises(ValueError):physical_terms(q,q,s,m,codec_contract='semantic')
    def test_gradient_fraction(self):
        c=coefficients(10,20,5);self.assertEqual(c,{'xy':.125,'first_difference':.2})
        with self.assertRaises(ValueError):coefficients(0,1,1)
if __name__=='__main__':unittest.main()
