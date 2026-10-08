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
