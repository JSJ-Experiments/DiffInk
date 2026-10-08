import unittest
from iam_tools.generation_prefix_confirmation import reserve_synthetic, fit_gate


class PrefixConfirmationTests(unittest.TestCase):
    def fixture(self):
        text='I We She They left found kept moved the blue notebook a small letter old book cup of tea on desk by window near door in office.'
        return {'train':dict(text=text)},['train'],[str(i) for i in range(8)]

    def test_deterministic_novel_prompts_known_writers_and_no_fake_points(self):
        records,ids,writers=self.fixture();a=reserve_synthetic(records,ids,writers);b=reserve_synthetic(records,ids,writers)
        self.assertEqual(a,b);self.assertEqual(len(a['records']),16)
        self.assertEqual(set(r['writer_id'] for r in a['records'].values()),set(writers))
        self.assertTrue(all(set(r)=={'text','writer_id'} for r in a['records'].values()))

    def test_exposed_prompts_excluded_not_just_assigned_new_writers(self):
        records,ids,writers=self.fixture();a=reserve_synthetic(records,ids,writers)
        records.update(a['records']);b=reserve_synthetic(records,ids,writers)
        self.assertFalse({r['text'] for r in a['records'].values()}&{r['text'] for r in b['records'].values()})

    def test_insufficient_character_or_writer_coverage_fails(self):
        records,ids,writers=self.fixture()
        with self.assertRaises(ValueError):reserve_synthetic(records,ids,writers[:7])
        with self.assertRaises(ValueError):reserve_synthetic({'train':dict(text='a')},ids,writers)

    def test_both_selected_candidates_must_be_frozen_and_fit_before_opening(self):
        import copy
        r={a:dict(stop='budget_completed',last_step=48000,best_step=36000,history=[dict(step=36000,aggregate={'all_train256':{'correct':{'free_cer':.04}}})]) for a in ['noncausal','causal']}
        fit_gate({'max_updates':48000},r)
        for key,value in [('last_step',36000),('stop','wall_limit'),('best_step',8000)]:
            q=copy.deepcopy(r);q['causal'][key]=value
            with self.assertRaises(ValueError):fit_gate({'max_updates':48000},q)
        for cer in [.051,float('nan'),float('inf'),-.01]:
            q=copy.deepcopy(r);q['causal']['history'][0]['aggregate']['all_train256']['correct']['free_cer']=cer
            with self.assertRaises(ValueError):fit_gate({'max_updates':48000},q)

    def test_actual_source_fingerprint_guard(self):
        import hashlib,numpy as np
        from iam_tools.generation_prefix_confirmation import verify_source_points
        x=np.zeros((8,5),dtype=np.float32);r=dict(points=8,points_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        verify_source_points(x,r)
        with self.assertRaises(ValueError):verify_source_points(x,dict(r,points=7))
        y=x.copy();y[1,0]=1
        with self.assertRaises(ValueError):verify_source_points(y,r)
