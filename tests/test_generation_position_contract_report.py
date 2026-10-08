import unittest
from iam_tools.report_generation_position_contract import verify


class PositionContractReportTests(unittest.TestCase):
    def fixture(self):
        arms=['relative100','absolute']
        cfg=dict(fresh=True,parent_step=0,neural_checkpoint_initialization=False,max_updates=48000,
                 models={a:dict(position_policy=a,alignment=True,width=16) for a in arms})
        data={'training_ids':{a:['x'] for a in arms}}
        results={a:dict(last_step=1,stop='wall_limit',codec_reader_unchanged=True) for a in arms}
        return cfg,data,results

    def test_exact_paired_arms(self):
        with self.assertRaisesRegex(ValueError,'exact relative100'):verify('.',{}, {}, {})

    def test_not_a_memorized_checkpoint_continuation(self):
        for key,value in [('fresh',False),('parent_step',48000),('neural_checkpoint_initialization',True),('max_updates',8000)]:
            cfg,data,results=self.fixture();cfg[key]=value
            with self.assertRaisesRegex(ValueError,'fresh empty-Adam'):verify('.',cfg,data,results)

    def test_scope_must_match(self):
        cfg,data,results=self.fixture();data['training_ids']['absolute']=['y']
        with self.assertRaisesRegex(ValueError,'actual TRAIN'):verify('.',cfg,data,results)

    def test_architecture_must_not_change(self):
        cfg,data,results=self.fixture();cfg['models']['absolute']['width']=32
        with self.assertRaisesRegex(ValueError,'only declared'):verify('.',cfg,data,results)

    def test_cannot_mislabel_policy_or_remove_prior(self):
        for key,value in [('position_policy','relative100'),('alignment',False)]:
            cfg,data,results=self.fixture();cfg['models']['absolute'][key]=value
            with self.assertRaisesRegex(ValueError,'only declared'):verify('.',cfg,data,results)

    def test_incomplete_experiment_not_a_matched_result(self):
        cfg,data,results=self.fixture()
        with self.assertRaisesRegex(ValueError,'matched complete'):verify('.',cfg,data,results)
