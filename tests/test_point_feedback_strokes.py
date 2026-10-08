import unittest
import torch
from iam_tools.point_feedback_strokes import PointFeedbackStrokeWriter


class PointFeedbackTests(unittest.TestCase):
    def model(self, feedback=True):
        torch.manual_seed(6)
        return PointFeedbackStrokeWriter(4, 2, .5, width=16, text_width=8, writer_width=4, point_feedback=feedback)

    def inputs(self):
        return torch.randn(2, 3, 40), torch.tensor([[0, 1, 2], [1, 3, -1]]), torch.tensor([0, 1])

    def test_same_parameters_and_only_policy_differs(self):
        a=self.model();b=self.model(False)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        self.assertNotIn('readout.weight',a.state_dict())

    def test_no_current_or_future_point_leakage(self):
        for policy in [True,False]:
            m=self.model(policy);f,t,w=self.inputs();a=m.teacher(f,t,w)
            changed=f.clone();changed[:,1,6:16]+=17.;changed[:,1,25:]+=17.;changed[:,2]+=17.
            # Point3 and every future target changed, earliest affected prediction is point4.
            b=m.teacher(changed,t,w)
            torch.testing.assert_close(a[0][:,:1],b[0][:,:1],atol=0,rtol=0)
            torch.testing.assert_close(a[0][:,1,:4],b[0][:,1,:4],atol=0,rtol=0)
            if policy:self.assertGreater(float((a[0][:,1,4:]-b[0][:,1,4:]).detach().abs().max()),1e-6)
            else:torch.testing.assert_close(a[0][:,1],b[0][:,1],atol=0,rtol=0)

    def test_fused_teacher_matches_sequential_true_history(self):
        for policy in [True,False]:
            m=self.model(policy).eval();f,t,w=self.inputs();o,p,tr=m.teacher(f,t,w)
            memory,valid,writer=m.memory(t,w);state=m.initial_state(memory);previous=f.new_zeros(2,40)
            for j in range(3):
                state,trace=m.coarse_step(previous,state,memory,valid,writer,f.new_full((2,),float(j==0)))
                hidden=state[1][None];prior=torch.cat((previous[:,14:16],previous[:,37:40]),-1)
                for q in range(8):
                    ins=torch.cat((prior,f.new_full((2,1),float(j==0 and q==0))),-1)
                    decoded,hidden=m.point_gru(ins[:,None],hidden);out=m.point_readout(decoded[:,0])
                    torch.testing.assert_close(o[:,j,q],out[:,:2],atol=1e-7,rtol=1e-5)
                    torch.testing.assert_close(p[:,j,q],out[:,2:],atol=1e-7,rtol=1e-5)
                    prior=torch.cat((f[:,j,2*q:2*q+2],f[:,j,16+3*q:19+3*q]),-1) if policy else torch.zeros_like(prior)
                previous=f[:,j]

    def test_free_history_matches_teacher_of_own_outputs(self):
        # This checks the ACTUAL generate point loop and chronological field layout.
        for policy in [True,False]:
            m=self.model(policy).eval();_,t,w=self.inputs();stats=dict(mean=[0.,0.],std=[1.,1.])
            with torch.no_grad():m.point_readout.bias[2:]=torch.tensor([5.,0.,-30.])
            g=m.generate(t,w,stats,max_blocks=3);xy=g['points'][...,:2];delta=xy-torch.cat((torch.zeros_like(xy[:,:1]),xy[:,:-1]),1)
            feedback=torch.cat((delta.reshape(2,3,16),g['points'][...,2:].reshape(2,3,24)),-1)
            o,p,_=m.teacher(feedback,t,w)
            torch.testing.assert_close(o.reshape(2,24,2),delta,atol=1e-7,rtol=2e-5)
            torch.testing.assert_close(p.argmax(-1).reshape(2,24),g['points'][...,2:].argmax(-1))
            longer=m.generate(t,w,stats,max_blocks=5)
            torch.testing.assert_close(g['points'],longer['points'][:,:24],atol=0,rtol=0)
            self.assertFalse(g['found_eoc'].any());self.assertEqual(g['stops'].tolist(),[24,24])

    def test_first_learned_eoc_not_forced(self):
        m=self.model().eval();_,t,w=self.inputs()
        with torch.no_grad():m.point_readout.weight.zero_();m.point_readout.bias[2:]=torch.tensor([0.,0.,5.])
        g=m.generate(t,w,dict(mean=[0.,0.],std=[1.,1.]),max_blocks=5)
        self.assertEqual(g['stops'].tolist(),[1,1]);self.assertTrue(g['found_eoc'].all())

    def test_batch_padding_and_backward(self):
        m=self.model();f,t,w=self.inputs();o,p,tr=m.teacher(f,t,w);a,b,_=m.teacher(f[1:],t[1:,:2],w[1:])
        torch.testing.assert_close(o[1:],a,atol=1e-7,rtol=1e-5)
        (o.square().mean()+p.square().mean()).backward()
        for layer in [m.point_gru,m.point_readout,m.history,m.decoder,m.clock]:
            self.assertGreater(sum(float(q.grad.abs().sum()) for q in layer.parameters() if q.grad is not None),0.)
        self.assertTrue((tr['advance']>0).all())

    def test_reject_invalid_policy_shape_mode_budget(self):
        with self.assertRaises(ValueError):self.model(1)
        m=self.model();f,t,w=self.inputs()
        with self.assertRaises(ValueError):m.teacher(f[...,:39],t,w)
        with self.assertRaises(ValueError):m.generate(t,w,dict(mean=[0.,0.],std=[1.,1.]))
        m.eval()
        for cap in [0,257,1.2]:
            with self.assertRaises(ValueError):m.generate(t,w,dict(mean=[0.,0.],std=[1.,1.]),max_blocks=cap)

if __name__=='__main__':unittest.main()
