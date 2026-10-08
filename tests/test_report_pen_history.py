import copy,unittest
from iam_tools.report_pen_history import validate_diagnostic
class PenHistoryReportTests(unittest.TestCase):
    def fixture(self):
        ids=list('abcdefgh');modes=['true_history','own_xy_true_pen_history','true_xy_own_pen_history','own_both_history']
        rows=[dict(sample_id=s,mode=m,source_length=True,not_free_generation=True,characters=3,true_pen_errors=0,predicted_pen_errors=1) for m in modes for s in ids]
        r=dict(model_reader_unchanged=True,summary={m:dict(true_pen_cer=0.,predicted_pen_cer=1/3) for m in modes},rows=rows,actual_free_prefix_parity=[dict(sample_id=s,max_xy_drift=0.,pen_mismatches=0) for s in ids]);return r,ids
    def test_all8_exact_metrics_parity_scope(self):
        r,ids=self.fixture();validate_diagnostic(r,ids)
        for change in ['missing','oracle','cer','parity']:
            bad=copy.deepcopy(r)
            if change=='missing':bad['rows'].pop()
            if change=='oracle':bad['rows'][0]['not_free_generation']=False
            if change=='cer':bad['summary']['true_history']['true_pen_cer']=.1
            if change=='parity':bad['actual_free_prefix_parity'][0]['max_xy_drift']=1e-6
            with self.assertRaises(ValueError):validate_diagnostic(bad,ids)
if __name__=='__main__':unittest.main()
