import itertools,math,unittest
import numpy as np
from iam_tools.ctc_beam import prefix_beam

class BeamTests(unittest.TestCase):
    def test_beam_can_beat_frame_argmax_by_summing_ctc_paths(self):
        p=np.array([[.4,.35,.25],[.4,.35,.25]])
        r=prefix_beam(np.log(p),['a','b']);self.assertEqual(r['text'],'a');self.assertAlmostEqual(math.exp(r['log_probability']),.4025)
    def test_repeated_character_requires_blank_separation(self):
        p=np.array([[.05,.9,.05],[.9,.05,.05],[.05,.9,.05]])
        self.assertEqual(prefix_beam(np.log(p),['a','b'])['text'],'aa')
    def test_exhaustive_small_ctc_matches_all_path_enumeration(self):
        p=np.random.default_rng(13).dirichlet([1,2,3],size=4);sums={}
        for path in itertools.product(range(3),repeat=4):
            text=[];last=None
            for token in path:
                if token and token!=last:text.append('ab'[token-1])
                last=token
            text=''.join(text);sums[text]=sums.get(text,0)+np.prod([p[t,c] for t,c in enumerate(path)])
        best=max(sums,key=sums.get);r=prefix_beam(np.log(p),['a','b'],width=100)
        self.assertEqual(r['text'],best);self.assertAlmostEqual(math.exp(r['log_probability']),sums[best])
    def test_empty_and_all_blank(self):
        self.assertEqual(prefix_beam(np.zeros((0,2)),['a'])['text'],'')
        self.assertEqual(prefix_beam(np.array([[0.,-math.inf],[0.,-math.inf]]),['a'])['text'],'')
    def test_head_audit_uses_same_logits_and_restores_training_mode(self):
        import torch
        from iam_tools.ctc_beam import audit_head
        class FixedHead(torch.nn.Module):
            def forward(self,z,padding_mask):
                return torch.tensor([.4,.35,.25]).log()[None,None].expand(2,z.shape[0],-1)
        head=FixedHead();cache={i:dict(mu=torch.ones(1,1,2),mask=torch.ones(1,2,dtype=torch.bool),labels=torch.zeros(1,1,dtype=torch.long)) for i in 'ab'}
        state=torch.get_rng_state().clone();r=audit_head(head,cache,dict(a='a',b='a'),dict(dev=['a'],held_out=['b']),['a','b'])
        self.assertTrue(head.training);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(r['groups']['dev']['greedy_cer'],1);self.assertEqual(r['groups']['dev']['beam_cer'],0)
        self.assertTrue(r['no_checkpoint_selection_using_beam']);self.assertEqual(len(r['lines']),2)

    def test_invalid_probabilities_alphabet_and_width(self):
        for p in (np.array([[math.nan,0]]),np.array([[math.inf,0]]),np.zeros((2,2))):
            with self.assertRaises(ValueError):prefix_beam(p,['a'])
        with self.assertRaises(ValueError):prefix_beam(np.log([[.5,.5]]),['a'],width=0)
        with self.assertRaises(ValueError):prefix_beam(np.log([[.3,.3,.4]]),['a','a'])

if __name__=='__main__':unittest.main()
