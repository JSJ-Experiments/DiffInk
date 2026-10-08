import unittest
import torch
from iam_tools.point_feedback_strokes import PointFeedbackStrokeWriter
from iam_tools.history_rollin import history_forward,rollin_probability,transition_mask

class HistoryRollinTests(unittest.TestCase):
    def setup_inputs(self):
        torch.manual_seed(9);model=PointFeedbackStrokeWriter(4,2,.5,width=16,text_width=8,writer_width=4,point_feedback=False)
        f=torch.randn(2,4,40);t=torch.tensor([[0,1,2],[1,3,-1]]);w=torch.tensor([0,1]);mask=torch.zeros(2,3,dtype=torch.bool)
        return model,f,t,w,mask
    def test_zero_rollin_matches_original_fused_teacher(self):
        m,f,t,w,z=self.setup_inputs();a=m.teacher(f,t,w);b=history_forward(m,f,t,w,z,z)
        for j in [0,1]:torch.testing.assert_close(a[j],b[j],atol=1e-7,rtol=1e-5)
        for k in a[2]:torch.testing.assert_close(a[2][k],b[2][k],atol=1e-7,rtol=1e-5)
    def test_full_rollin_matches_actual_free_prefix(self):
        m,f,t,w,z=self.setup_inputs();m.eval()
        with torch.no_grad():m.point_readout.bias[2:]=torch.tensor([5.,0.,-30.])
        a=history_forward(m,f,t,w,~z,~z);g=m.generate(t,w,dict(mean=[0.,0.],std=[1.,1.]),max_blocks=4)
        delta=g['points'][...,:2]-torch.cat((torch.zeros_like(g['points'][:,:1,:2]),g['points'][:,:-1,:2]),1)
        torch.testing.assert_close(a[0].reshape(2,-1,2),delta,atol=2e-7,rtol=1e-4)
        torch.testing.assert_close(a[1].argmax(-1).reshape(2,-1),g['points'][...,2:].argmax(-1))
        self.assertTrue(torch.equal(a[2]['used_history'][:,0],torch.zeros(2,40)))
    def test_current_future_feedback_never_reaches_current_block(self):
        m,f,t,w,z=self.setup_inputs()
        for ownxy,ownpen in [(z,z),(~z,z),(z,~z),(~z,~z)]:
            a=history_forward(m,f,t,w,ownxy,ownpen);changed=f.clone();changed[:,2:]+=20.;b=history_forward(m,changed,t,w,ownxy,ownpen)
            torch.testing.assert_close(a[0][:,:3],b[0][:,:3],atol=0,rtol=0)
            if ownxy.all() and ownpen.all():torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
    def test_xy_and_pen_interventions_are_independent(self):
        m,f,t,w,z=self.setup_inputs()
        for x,p in [(~z,z),(z,~z)]:
            o,pen,tr=history_forward(m,f,t,w,x,p);h=tr['used_history'][:,1]
            if x.all():
                torch.testing.assert_close(h[:,:16],o[:,0].detach().flatten(1));torch.testing.assert_close(h[:,16:],f[:,0,16:])
            else:
                torch.testing.assert_close(h[:,:16],f[:,0,:16]);torch.testing.assert_close(h[:,16:],torch.nn.functional.one_hot(pen[:,0].argmax(-1),3).float().flatten(1))
    def test_own_feedback_detached_but_current_output_and_recurrence_train(self):
        m,f,t,w,z=self.setup_inputs();f.requires_grad_(True);o,p,tr=history_forward(m,f,t,w,~z,~z)
        self.assertFalse(tr['used_history'].requires_grad)
        loss=o.square().mean()+p.square().mean();loss.backward();self.assertIsNone(f.grad)
        for layer in [m.history,m.decoder,m.point_gru,m.point_readout,m.clock]:
            self.assertGreater(sum(float(q.grad.abs().sum()) for q in layer.parameters() if q.grad is not None),0.)
    def test_mask_probability_bounded_and_padding_never_rolls_in(self):
        real=torch.tensor([[[True]*8,[True]*8,[False]*8],[[True]*8,[True]*8,[True]*8]])
        draws=torch.tensor([[.1,.1],[.3,.1]])
        self.assertEqual(transition_mask(draws,real,.2).tolist(),[[True,False],[False,True]])
        self.assertEqual(rollin_probability(250),.1);self.assertEqual(rollin_probability(1000),.2)
        for step in [0,1.2]:
            with self.assertRaises(ValueError):rollin_probability(step)
        with self.assertRaises(ValueError):transition_mask(torch.ones(2,2),real,.2)
    def test_reject_wrong_model_mask_and_shape(self):
        m,f,t,w,z=self.setup_inputs()
        for bad in [z.float(),z[:,:1]]:
            with self.assertRaises(ValueError):history_forward(m,f,t,w,bad,z)
        m.config['point_feedback']=True
        with self.assertRaises(ValueError):history_forward(m,f,t,w,z,z)

if __name__=='__main__':unittest.main()
