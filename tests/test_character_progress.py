import copy
import unittest
import numpy as np
from iam_tools.character_progress import emission_centers,form_folds,fit_clock,predict_clock,monotonic_progress,MODES

class CharacterProgressTests(unittest.TestCase):
    def fixture(self):
        records={};labels={}
        for j in range(12):
            sid=f's{j}';text=['ab','ba','aba'][j%3]
            records[sid]=dict(text=text,writer_id=str(j%2),points=len(text)*8,prompt_family=f'f{j}')
            labels[sid]=np.repeat(np.arange(len(text)),2)
        return records,labels,list(records)
    def test_clock_units_and_no_blank_interpolation_in_teacher(self):
        centers,gaps=emission_centers(np.array([-1,0,0,-1,1,1,-1]),2,27)
        np.testing.assert_allclose(centers,[1.,2.5]);np.testing.assert_allclose(gaps,[1.,1.5,1.5])
    def test_requires_complete_monotonic_integer_path_and_actual_frame_count(self):
        for a,n,p in [(np.array([0,1,0]),2,12),(np.array([0,-1]),2,8),(np.array([0.,1.]),2,8),(np.array([0,2]),2,8),(np.array([0,1]),2,9),(np.array([-2,0]),1,8)]:
            with self.assertRaises(ValueError):emission_centers(a,n,p)
    def test_form_and_duplicate_text_connected_components(self):
        r={f's{j}':dict(text=f'text{j}',writer_id='w',prompt_family=f'f{j}') for j in range(8)}
        r['s1']['text']=r['s0']['text'];r['s2']['prompt_family']='f1'
        folds=form_folds(r,list(r),folds=3)
        self.assertEqual(folds['s0'],folds['s1']);self.assertEqual(folds['s1'],folds['s2'])
        self.assertEqual(folds,form_folds(r,list(r),folds=3))
        for k in range(3):
            tr=[s for s in r if folds[s]!=k];te=[s for s in r if folds[s]==k]
            self.assertFalse({r[s]['prompt_family'] for s in tr}&{r[s]['prompt_family'] for s in te})
            self.assertFalse({r[s]['text'] for s in tr}&{r[s]['text'] for s in te})
    def test_positive_monotonic_all_modes_unknown_fallback(self):
        records,labels,ids=self.fixture()
        for mode in MODES:
            m=fit_clock(records,labels,ids,mode=mode);p=predict_clock(m,'abba','0')
            self.assertTrue((p['intervals']>0).all());self.assertTrue((np.diff(p['centers'])>0).all())
            self.assertAlmostEqual(p['total_blocks'],p['intervals'].sum())
            q=predict_clock(m,'az !','new');self.assertFalse(q['known_writer']);self.assertEqual(q['unknown_characters'],[' ','!','z'])
    def test_reject_extra_held_targets(self):
        r,l,ids=self.fixture();l['held']=np.array([0])
        with self.assertRaises(ValueError):fit_clock(r,l,ids)
    def test_sample_ids_points_form_metadata_cannot_influence_inference(self):
        r,l,ids=self.fixture();m=fit_clock(r,l,ids);p=predict_clock(m,'abba','0')
        mutated=copy.deepcopy(m);mutated['train_ids']=['arbitrary'];mutated['clock']='unused description'
        np.testing.assert_array_equal(p['centers'],predict_clock(mutated,'abba','0')['centers'])
    def test_progress_monotonic_and_prefix_budget_invariant(self):
        c=np.array([.5,2.,5.]);p=monotonic_progress(c,np.arange(20)/2)
        self.assertTrue((np.diff(p)>=0).all());self.assertEqual(p[0],0.);self.assertEqual(p[-1],2.)
        np.testing.assert_array_equal(p[:8],monotonic_progress(c,np.arange(8)/2))
        np.testing.assert_allclose(monotonic_progress(c,c),[0,1,2])
    def test_single_token_and_bad_centers(self):
        np.testing.assert_array_equal(monotonic_progress([1.],[-1,0,1,2]),[0,0,0,0])
        for c in [[],[0],[1,1],[2,1],[np.nan]]:
            with self.assertRaises(ValueError):monotonic_progress(c,[0,1])
    def test_prediction_rejects_invalid_model_or_text(self):
        r,l,ids=self.fixture();m=fit_clock(r,l,ids)
        for text,w in [('', '0'),('a',2)]:
            with self.assertRaises(ValueError):predict_clock(m,text,w)
        bad=copy.deepcopy(m);bad['coefficients'][0]=np.nan
        with self.assertRaises(ValueError):predict_clock(bad,'ab','0')
        bad=copy.deepcopy(m);bad['smearing_factor']=-1
        with self.assertRaises(ValueError):predict_clock(bad,'ab','0')
        for ridge in [0,-1,np.nan]:
            with self.assertRaises(ValueError):fit_clock(r,l,ids,ridge=ridge)
    def test_fits_reusable_synthetic_character_intervals(self):
        records={};labels={}
        for j,text in enumerate(['ab','ba','aa','bb','aba','bab']*5):
            # a occupies two4-index frames; b six. Complete monotonic weak clock.
            sid=str(j);labels[sid]=np.concatenate([np.repeat(i,2 if c=='a' else 6) for i,c in enumerate(text)])
            records[sid]=dict(text=text,writer_id='w',points=len(labels[sid])*4,prompt_family=sid)
        m=fit_clock(records,labels,list(records),ridge=.01)
        pa=predict_clock(m,'aaaa','w');pb=predict_clock(m,'bbbb','w')
        self.assertGreater(pb['total_blocks'],pa['total_blocks']*2)

if __name__=='__main__':unittest.main()
