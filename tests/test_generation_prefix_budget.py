import unittest
from iam_tools.generation_prefix_budget import budget,aggregate


class PrefixBudgetTests(unittest.TestCase):
    def test_constant_budget_does_not_read_targets_or_duration_model(self):
        self.assertEqual(budget('generous256',{},None),256)

    def test_native_is_explicit_diagnostic(self):
        r={'points':581}
        self.assertEqual(budget('native',r,None),73)
        self.assertEqual(budget('shorter_one',r,None),72)
        self.assertEqual(budget('longer_one',r,None),74)
        with self.assertRaises(ValueError):budget('implicit',r,None)

    def test_aggregation_does_not_hide_failures(self):
        rows=[dict(free_errors=1,characters=2,first_eoc_point=None,latent_prefix_max_abs=.1,xy_prefix_max_abs=.2,
             x_prefix_rmse=.1,y_prefix_rmse=.2,pen_changes=2,reader_equal_to_native=False),
             dict(free_errors=0,characters=8,first_eoc_point=5,latent_prefix_max_abs=0,xy_prefix_max_abs=0,
             x_prefix_rmse=0,y_prefix_rmse=0,pen_changes=0,reader_equal_to_native=True)]
        q=aggregate(rows);self.assertEqual(q['cer'],.1);self.assertEqual(q['missing_eoc'],1)
        self.assertEqual(q['pen_changes'],2);self.assertEqual(q['reader_changes'],1)

    def test_unpaired_generous_eval_has_no_fake_reference_metrics(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        import torch
        from iam_tools.generation_prefix_budget import evaluate_generous
        class FakeModel(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            def forward(self,x,time,text,mask,**kwargs):
                self.assertion=(x.shape[1]==256 and bool(mask.all()))
                return x
        m=FakeModel();records={'a':dict(text='abc',writer_id='w')}
        metrics=dict(free_errors=1,characters=3,first_eoc_point=None,oracle_window_decoded='fake',window_errors=1,oracle_points=2048,internal_eoc_count=2)
        with tempfile.TemporaryDirectory() as d,patch('iam_tools.generation_prefix_budget.decode_sample',return_value=(np.zeros((2048,5)),metrics)):
            result=evaluate_generous(m,None,None,records,list('abc'),dict(mean=torch.zeros(384),std=torch.ones(384)),['a'],d,1,['w'],controls=False)
            self.assertTrue(m.assertion);self.assertTrue(m.training)
            row=result['lines'][0];self.assertTrue(row['no_paired_target']);self.assertEqual(row['budget_blocks'],256)
            self.assertTrue(all(key not in row for key in ['oracle_window_decoded','window_errors','oracle_points','internal_eoc_count']))
            self.assertEqual(result['aggregate']['generous_correct']['missing_eoc'],1)

    def test_bad_device_rejected_before_loading_or_creating_artifacts(self):
        from iam_tools.generation_prefix_budget import audit
        with self.assertRaisesRegex(ValueError,'available explicit'):audit('missing','missing',device='tpu')
