import unittest
import numpy as np
from iam_tools.character_progress import fit_clock
from iam_tools.character_progress_study import fold_partition,compare_progress,bootstrap_form_delta,predictions
from iam_tools.generation_composition import fit_duration

class CharacterProgressStudyTests(unittest.TestCase):
    def fixture(self):
        records={f's{j}':dict(text=['ab','ba'][j%2]+str(j),writer_id='w',points=24,prompt_family=f'f{j}') for j in range(6)}
        labels={s:np.array([0,0,1,1,2,2]) for s in records}
        return records,labels,list(records)
    def test_fold_leakage_guard_for_both_text_and_form(self):
        r,l,ids=self.fixture();a={s:j%2 for j,s in enumerate(ids)}
        tr,te=fold_partition(r,ids,a,0);self.assertEqual(set(tr)|set(te),set(ids))
        r['s1']['prompt_family']=r['s0']['prompt_family']
        with self.assertRaises(ValueError):fold_partition(r,ids,a,0)
        r['s1']['prompt_family']='f1';r['s1']['text']=r['s0']['text']
        with self.assertRaises(ValueError):fold_partition(r,ids,a,0)
    def test_primary_progress_independent_of_oracle_metadata(self):
        r,l,ids=self.fixture();models={m:fit_clock(r,l,ids,mode=m) for m in ['neighbor','character','no_glyph']};d=fit_duration(r,ids,['w']);q=np.arange(6)/2
        record=dict(r['s0']);pred,_=predictions(models,d,d['blocks_per_character'],record,np.array([.5,1.5,2.5]),q)
        # Actual point budget is not consumed by primary inference branches.
        record['points']=9999;changed,_=predictions(models,d,d['blocks_per_character'],record,np.array([1.,4.,10.]),q)
        for mode in ['static_ratio','duration_uniform','neighbor','character','no_glyph','same_endpoints_uniform']:
            np.testing.assert_array_equal(pred[mode],changed[mode])
        self.assertFalse(np.array_equal(pred['oracle_endpoints_uniform'],changed['oracle_endpoints_uniform']))
    def test_progress_metric_is_index_error_not_cer(self):
        m=compare_progress(np.array([.1,1.4,1.]),np.array([0,1,2]))
        self.assertAlmostEqual(m['nearest_token_accuracy'],2/3);self.assertAlmostEqual(m['index_error']['mean'],.5)
    def test_form_bootstrap_paired_identical_and_constant_differences(self):
        rows=[dict(prompt_family=f'f{j//2}',metrics={'a':{'index_error':{'p90':1.}},'b':{'index_error':{'p90':2.}}}) for j in range(6)]
        b=bootstrap_form_delta(rows,'a','b',repetitions=100);self.assertEqual(b['observed'],-1.);self.assertEqual(b['descriptive_percentile95'],[-1.,-1.]);self.assertEqual(b['form_clusters'],3)
        self.assertEqual(bootstrap_form_delta(rows,'a','a',repetitions=100)['descriptive_percentile95'],[0.,0.])

if __name__=='__main__':unittest.main()
