import unittest
from unittest.mock import patch
import torch
from iam_tools.rocm_reader import ATenFrozenReader
class DummyReader(torch.nn.Module):
 def __init__(self):
  super().__init__();self.linear=torch.nn.Linear(3,2);self.dropout=torch.nn.Dropout(.9);self.calls=[];self.requires_grad_(False)
 def forward(self,x,**kwargs):
  self.calls.append((torch.backends.cudnn.enabled,kwargs));return self.linear(x)
class NativeReaderTests(unittest.TestCase):
 def test_opt_in_preserves_weights_and_masks(self):
  base=DummyReader().eval();before={k:v.clone() for k,v in base.state_dict().items()};wrapper=ATenFrozenReader(base);x=torch.randn(2,3);pm=torch.ones(2,8,dtype=torch.bool)
  torch.testing.assert_close(wrapper(x,point_mask=pm),base.linear(x));self.assertFalse(base.calls[0][0]);self.assertIs(base.calls[0][1]['point_mask'],pm)
  for k,v in base.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
  self.assertTrue(wrapper.requires_point_mask)
 def test_vendor_flag_is_restored(self):
  base=DummyReader();before=torch.backends.cudnn.enabled;ATenFrozenReader(base)(torch.randn(2,3));self.assertEqual(torch.backends.cudnn.enabled,before)
 def test_flag_restored_on_exception_without_retry(self):
  base=DummyReader();before=torch.backends.cudnn.enabled
  with patch.object(base,'forward',side_effect=RuntimeError('reader failure')) as forward:
   with self.assertRaisesRegex(RuntimeError,'reader failure'):ATenFrozenReader(base)(torch.randn(2,3))
   self.assertEqual(forward.call_count,1)
  self.assertEqual(torch.backends.cudnn.enabled,before)
 def test_training_cannot_reenable_frozen_dropout(self):
  base=DummyReader();wrapper=ATenFrozenReader(base).train(True);self.assertFalse(wrapper.training);self.assertFalse(base.training);self.assertFalse(base.dropout.training)
 def test_trainable_reader_rejected(self):
  with self.assertRaisesRegex(ValueError,'frozen'):ATenFrozenReader(torch.nn.Linear(3,2))
if __name__=='__main__':unittest.main()
