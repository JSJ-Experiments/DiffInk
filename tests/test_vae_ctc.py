"""Regression checks on the actual VAE.forward → VAE.get_ocr_loss path."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from model.vae import VAE


def small_config():
    return SimpleNamespace(in_channels=5,hidden_dims=[8,8,8],latent_dim=8,
        decoder_dims=[8,8,128],trans_hidden_dim=8,decoder_output_dim=123,
        trans_num_layers=1,trans_num_heads=2,ocr_hidden_dim=8,ocr_num_heads=2,
        ocr_num_layers=1,num_text_embedding=4,style_classifier_dim=8,num_writer=2)


class VAETrainingPathCTCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model=VAE(small_config()).eval()

    def forward_loss(self, labels, points):
        data=torch.randn(1,5,points)
        pad_mask=torch.ones(1,points//8,dtype=torch.bool)
        with patch.object(self.model,'get_ocr_loss',wraps=self.model.get_ocr_loss) as actual:
            with torch.no_grad():
                output,ctc,kl,style=self.model(data,pad_mask,labels,torch.tensor([0]))
            self.assertEqual(actual.call_count,1)
        self.assertEqual(output.shape,(1,123,points))
        for value in (output,ctc,kl,style):self.assertTrue(torch.isfinite(value).all())
        return ctc

    def test_distinct_three_labels_three_steps_through_forward(self):
        # Old VAE filter requires five timesteps and silently returns zero.
        loss=self.forward_loss(torch.tensor([[0,1,2]]),24)
        self.assertGreater(float(loss),0)

    def test_padding_does_not_count_as_target_or_repeat(self):
        loss=self.forward_loss(torch.tensor([[0,1,-1,-1]]),16)
        self.assertGreater(float(loss),0)

    def test_repeated_labels_need_blank_step(self):
        invalid=self.forward_loss(torch.tensor([[0,0]]),16)
        valid=self.forward_loss(torch.tensor([[0,0]]),24)
        self.assertEqual(float(invalid),0)
        self.assertGreater(float(valid),0)

    def test_vae_and_ocr_helper_share_loss(self):
        features=torch.randn(1,8,3);labels=torch.tensor([[0,1,2]])
        valid=torch.ones(1,3,dtype=torch.bool)
        with torch.no_grad():
            a=self.model.get_ocr_loss(features,labels,valid)
            b=self.model.ocr_model.get_ocr_loss(features,labels,valid)
        torch.testing.assert_close(a,b)
        self.assertGreater(float(a),0)

if __name__=='__main__':unittest.main()
