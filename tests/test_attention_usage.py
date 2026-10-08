import unittest,torch
from iam_tools.attention_usage import rotated_character_delta,last_layer_intervention,MODES
import tests.test_weak_alignment as fixtures

class AttentionUsageTests(unittest.TestCase):
    def test_manual_control_and_state_rng_unchanged(self):
        m,args,wi=fixtures.WeakAlignmentTests().fixture();a=m(*args,writer_ids=wi);rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in m.state_dict().items()}
        with last_layer_intervention(m,args[2],'manual_control'):b=m(*args,writer_ids=wi)
        torch.testing.assert_close(a,b,atol=3e-7,rtol=1e-5)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()))
        self.assertFalse(m.blocks[-1].cross_attention._forward_hooks)

    def test_token_delta_preserves_bos_padding_and_no_position_rotation(self):
        m,args,_=fixtures.WeakAlignmentTests().fixture();text=args[2];delta=rotated_character_delta(m,text)
        self.assertTrue((delta[:,0]==0).all());self.assertTrue((delta[1,-1]==0).all())
        tokens=text[0]+2;expected=m.text(tokens.roll(1))-m.text(tokens)
        torch.testing.assert_close(delta[0,1:],expected)

    def test_interventions_nonvacuous_and_cleaned_up(self):
        m,args,wi=fixtures.WeakAlignmentTests().fixture();a=m(*args,writer_ids=wi)
        for mode in MODES[1:]:
            with last_layer_intervention(m,args[2],mode):b=m(*args,writer_ids=wi)
            self.assertGreater(float((a-b).detach().abs().max()),1e-5,mode)
            self.assertFalse(m.blocks[-1].cross_attention._forward_hooks)
        with self.assertRaises(ValueError):
            with last_layer_intervention(m,args[2],'invalid'):pass
