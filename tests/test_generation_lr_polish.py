import copy,unittest
import numpy as np
import torch
from iam_tools.generation_lr_polish import restore_adam,phase_statistics
from iam_tools.ocr_context_study import tensor_digest

class LRPolishTests(unittest.TestCase):
 def fixture(self):
  model=torch.nn.Linear(2,2);params=list(model.parameters())
  opt=torch.optim.AdamW([dict(params=params[:1],lr=3e-4),dict(params=params[1:],lr=7e-4)],betas=(.9,.99),weight_decay=.01)
  model(torch.ones(1,2)).square().sum().backward();opt.step();model.zero_grad(set_to_none=True)
  return model,opt.state_dict()
 def test_restores_all_moments_changes_every_actual_lr(self):
  model,state=self.fixture();before=tensor_digest({str(i)+':'+k:v for i,s in state['state'].items() for k,v in s.items()})
  weights=tensor_digest(model.state_dict());opt=restore_adam(model,state,1e-5);after=opt.state_dict()
  self.assertEqual(before,tensor_digest({str(i)+':'+k:v for i,s in after['state'].items() for k,v in s.items()}))
  self.assertEqual(weights,tensor_digest(model.state_dict()));self.assertEqual([g['lr'] for g in opt.param_groups],[1e-5]*2)
  self.assertEqual([g['lr'] for g in state['param_groups']],[3e-4,7e-4]);self.assertTrue(all(g['betas']==(.9,.99) for g in opt.param_groups))
  self.assertTrue(all(p.grad is None for p in model.parameters()))
 def test_no_shared_mutable_moments(self):
  model,state=self.fixture();a=restore_adam(model,state,1e-4);b=restore_adam(model,state,1e-5)
  next(iter(a.state.values()))['exp_avg'].add_(42)
  self.assertFalse(torch.equal(next(iter(a.state.values()))['exp_avg'],next(iter(b.state.values()))['exp_avg']))
 def test_bad_lr_reset_missing_duplicate_group(self):
  model,state=self.fixture()
  for lr in [0,-1,float('inf'),float('nan'),True,.1]:
   with self.assertRaises(ValueError):restore_adam(model,state,lr)
  for mutation in [lambda s:s.__setitem__('state',{}),lambda s:s['param_groups'][1].__setitem__('params',[0]),lambda s:s['param_groups'][1].__setitem__('params',[99])]:
   bad=copy.deepcopy(state);mutation(bad)
   with self.assertRaises(ValueError):restore_adam(model,bad,1e-5)
 def test_phase7_detects_seams_ignores_penup_jumps(self):
  target=np.c_[np.arange(24.),np.zeros(24)];pred=target.copy();pred[8:16,1]=1;states=np.zeros(24,int);states[-1]=2
  rows=phase_statistics([(pred,target,states)]);self.assertEqual(rows[7]['segment_error_rmse'],1.);self.assertTrue(all(r['segment_error_rmse']==0 for r in rows[:7]))
  states[7]=states[15]=1;rows=phase_statistics([(pred,target,states)]);self.assertEqual(rows[7]['links'],0);self.assertIsNone(rows[7]['segment_error_rmse'])
 def test_target_corners_and_nonuniform_spacing_not_smoothed(self):
  target=np.array([[0.,0],[.1,0],[.1,1],[.1,3],[4,3]]);states=np.array([0,0,0,0,2]);rows=phase_statistics([(target,target,states)],2)
  self.assertTrue(all(r['segment_error_rmse']==0 and r['tangent_error_p90_degrees']==0 for r in rows));self.assertNotEqual(rows[0]['target_segment_rms'],rows[1]['target_segment_rms'])
 def test_degenerate_and_bad_shapes(self):
  rows=phase_statistics([(np.zeros((1,2)),np.zeros((1,2)),[2])]);self.assertTrue(all(r['links']==0 and r['tangent_error_p90_degrees'] is None for r in rows))
  with self.assertRaises(ValueError):phase_statistics([(np.zeros((3,2)),np.zeros((2,2)),[0,2])])
  with self.assertRaises(ValueError):phase_statistics([],0)
if __name__=='__main__':unittest.main()
