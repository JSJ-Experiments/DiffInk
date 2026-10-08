import copy,unittest
from iam_tools.report_corpus_dit import aggregate,verify_evaluation,verify_log,trajectory_section
class CorpusDiTReportTests(unittest.TestCase):
    def fixture(self):
        cfg=dict(eval_noise_seeds=[11,12],eval_guidance=[1.,2.]);data=dict(records=dict(a=dict(text='ab')),duration={});rows=[]
        from unittest.mock import patch
        for seed in cfg['eval_noise_seeds']:
            for g in cfg['eval_guidance']:
                for p in (['correct','swapped','null'] if seed==11 and g==1 else ['correct']):
                    supplied='' if p=='null' else 'ab';rows.append(dict(split='fresh_confirmation',sample_id='a',seed=seed,guidance=g,policy=p,text='ab',decoded='ab',characters=2,errors=0,conditioning_text=supplied,errors_against_supplied_text=None if p=='null' else 0,no_source_trajectory=True,no_source_length=True,no_writer_id=True,estimated_blocks=3,generated_points=8,found_eoc=True))
        splits=dict(fresh_confirmation=['a']);ev=dict(rows=rows,aggregate=aggregate(rows,splits,cfg['eval_guidance']));return cfg,data,ev,splits
    def verify(self,cfg,data,ev,splits):
        from unittest.mock import patch
        with patch('iam_tools.report_corpus_dit.duration',return_value=3):verify_evaluation(cfg,data,ev,splits,True)
    def test_all_actual_seeds_guidances_and_controls_reproduce(self):self.verify(*self.fixture())
    def test_aggregate_tamper_rejected(self):
        c,d,e,s=self.fixture();e['aggregate']['fresh_confirmation']['1.0']['cer']=.5
        with self.assertRaisesRegex(ValueError,'aggregate'):self.verify(c,d,e,s)
    def test_missing_or_duplicate_row_rejected(self):
        for mode in ('missing','duplicate'):
            c,d,e,s=self.fixture()
            if mode=='missing':e['rows'].pop()
            else:e['rows'].append(copy.deepcopy(e['rows'][0]))
            with self.assertRaises(ValueError):self.verify(c,d,e,s)
    def test_reader_errors_independently_recomputed(self):
        c,d,e,s=self.fixture();e['rows'][0]['decoded']='wrong'
        with self.assertRaisesRegex(ValueError,'CER'):self.verify(c,d,e,s)
    def test_oracle_length_flag_or_cap_mismatch_rejected(self):
        for key,value in [('no_source_length',False),('estimated_blocks',10),('generated_points',50),('found_eoc',False)]:
            c,d,e,s=self.fixture();e['rows'][0][key]=value
            with self.assertRaises(ValueError):self.verify(c,d,e,s)
    def test_correct_and_null_supplied_text_contract(self):
        for policy in ('correct','null'):
            c,d,e,s=self.fixture();next(r for r in e['rows'] if r['policy']==policy)['conditioning_text']='other'
            with self.assertRaises(ValueError):self.verify(c,d,e,s)
class CorpusDiTReportLogTests(unittest.TestCase):
    def fixture(self):
        from iam_tools.corpus_dit_study import schedule
        from iam_tools.corpus_dit import learning_rate
        import json,hashlib
        cfg=dict(max_updates=3,batch=2,schedule_seed=5,clip=1.,warmup_updates=1,lr=5e-5,min_lr=1e-6,max_train_wall_seconds=1800)
        ids=['a','b','c'];data=dict(scope=dict(splits=dict(train=ids)),records={i:dict(points=32) for i in ids});batches=list(schedule(ids,dict.fromkeys(ids,4),3,2,5))
        rows=[dict(step=j,sample_ids=b,x0_loss=1.,active40_loss=1.,unused344_loss=1.,raw_grad_norm=.5,train_seconds=j,clipped=False,lr=learning_rate(j,cfg),text_dropped=False,prefix_retained=True) for j,b in enumerate(batches,1)]
        result=dict(last_step=3,stop='budget_completed',clip_fraction=0.,training_order_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),history=[dict(step=0,aggregate=dict(dev_unseen_text={'1.0':dict(cer=1.)})),dict(step=3,aggregate=dict(dev_unseen_text={'1.0':dict(cer=.75)}))],best_step=3,best_dev_cer=.75,codec_reader_unchanged=True,confirmation_not_encoded_during_training=True,not_promoted=True)
        return cfg,data,result,rows
    def test_complete_train_log_and_dev_selection_reproduce(self):verify_log(*self.fixture())
    def test_unused_fields_cannot_hide_geometry_loss(self):
        c,d,r,rows=self.fixture();rows[0]['active40_loss']=3.
        with self.assertRaisesRegex(ValueError,'objective'):verify_log(c,d,r,rows)
    def test_cannot_reselect_on_confirmation(self):
        c,d,r,rows=self.fixture();r['best_step']=0
        with self.assertRaisesRegex(ValueError,'DEV'):verify_log(c,d,r,rows)
    def test_order_or_cfg_drop_or_clipping_corruption_rejected(self):
        for key,value in [('sample_ids',['held']),('text_dropped',True),('clipped',True)]:
            c,d,r,rows=self.fixture();rows[0][key]=value
            with self.assertRaises(ValueError):verify_log(c,d,r,rows)
    def test_missing_update_or_false_budget_completion_rejected(self):
        c,d,r,rows=self.fixture();rows.pop()
        with self.assertRaises(ValueError):verify_log(c,d,r,rows)
        c,d,r,rows=self.fixture();c['max_updates']=5
        with self.assertRaises(ValueError):verify_log(c,d,r,rows)
class CorpusInlineRenderTests(unittest.TestCase):
    def test_inline_svg_keeps_fixed_scale_and_escapes_user_transcript(self):
        import numpy as np
        r=dict(split='dev',sample_id='id',policy='correct',seed=1,guidance=1.,text='<script>x</script>',conditioning_text='<script>x</script>',decoded='<b>',errors=1,characters=2,found_eoc=True)
        section=trajectory_section(r,np.array([[0.,0.,1,0,0],[2.,1.,0,0,1]]))
        self.assertIn('<svg ',section);self.assertNotIn('<img ',section);self.assertNotIn('<script>',section);self.assertIn('&lt;script&gt;',section);self.assertIn('width="212.000" height="112.000"',section);self.assertNotIn('<circle',section)
    def test_inline_render_still_rejects_invalid_geometry(self):
        import numpy as np
        r=dict(split='dev',sample_id='id',policy='correct',seed=1,guidance=1.,text='a',conditioning_text='a',decoded='',errors=1,characters=1,found_eoc=True)
        with self.assertRaises(ValueError):trajectory_section(r,np.array([[float('nan'),0.,0.,0.,1.]]))
if __name__=='__main__':unittest.main()
