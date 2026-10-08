import unittest
from iam_tools.report_generation_prefix_contract import verify


class PrefixContractReportTests(unittest.TestCase):
    def fixture(self):
        arms=['noncausal','causal']
        cfg=dict(fresh=True,parent_step=0,neural_checkpoint_initialization=False,max_updates=48000,
                 models={a:dict(position_policy='absolute',causal_queries=a=='causal',alignment=True,width=16) for a in arms})
        data={'training_ids':{a:['x'] for a in arms}}
        results={a:dict(last_step=1,stop='wall_limit',codec_reader_unchanged=True) for a in arms}
        return cfg,data,results

    def test_exact_arms(self):
        with self.assertRaisesRegex(ValueError,'exact noncausal'):verify('.',{}, {}, {})

    def test_only_attention_changes(self):
        for key,value in [('width',32),('causal_queries',False),('position_policy','relative100'),('alignment',False)]:
            cfg,data,results=self.fixture();cfg['models']['causal'][key]=value
            with self.assertRaisesRegex(ValueError,'only declared'):verify('.',cfg,data,results)

    def test_fresh_complete_paired_protocol(self):
        cfg,data,results=self.fixture()
        with self.assertRaisesRegex(ValueError,'matched complete'):verify('.',cfg,data,results)
        cfg['fresh']=False
        with self.assertRaisesRegex(ValueError,'fresh empty-Adam'):verify('.',cfg,data,results)
        cfg,data,results=self.fixture();data['training_ids']['causal']=['y']
        with self.assertRaisesRegex(ValueError,'actual TRAIN'):verify('.',cfg,data,results)
