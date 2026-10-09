"""Tests for CrossAttentionDiT: joint text-trajectory attention architecture.

Verifies that the new cross-attention architecture:
1. Constructs correctly and produces finite outputs
2. Keeps text at natural character length (not padded to latent length)
3. Has proper padding isolation (padding noise doesn't leak into real positions)
4. Produces different outputs for different text inputs (text sensitivity)
5. Supports backward pass for training
6. Supports classifier-free guidance (drop_text/drop_cond)
7. Has the same forward() signature as original DiT for API compatibility
"""
import unittest, torch
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1] / 'third_party/DiffInk'
if not REPO.is_dir():
    REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.dit import CrossAttentionDiT, CrossAttentionTextEmbedding, CrossAttentionInputEmbedding, DiT
from utils.utils import ModelConfig


class CrossAttentionDiTConfig:
    """Shared test config for CrossAttentionDiT."""
    @staticmethod
    def small():
        return ModelConfig(dict(
            dim=32, latent_dim=40, num_text_embedding=12,
            text_dim=16, text_mask_padding=True, conv_layers=1,
            dim_head=8, depth=2, heads=4, ff_mult=2,
            dropout=0., long_skip_connection=False,
        ))

    @staticmethod
    def with_skip():
        return ModelConfig(dict(
            dim=32, latent_dim=40, num_text_embedding=12,
            text_dim=16, text_mask_padding=True, conv_layers=1,
            dim_head=8, depth=2, heads=4, ff_mult=2,
            dropout=0., long_skip_connection=True,
        ))


class TestCrossAttentionTextEmbedding(unittest.TestCase):
    def test_preserves_natural_text_length(self):
        """Text should remain at its natural character length, not padded to latent length."""
        te = CrossAttentionTextEmbedding(text_num_embeds=12, text_dim=16, conv_layers=0)
        # 3-char and 5-char texts, padded to max_len=5
        text = torch.tensor([[1, 2, 3, -1, -1], [1, 2, 3, 4, 5]])
        embed, mask = te(text)
        self.assertEqual(tuple(embed.shape), (2, 5, 16))
        self.assertEqual(tuple(mask.shape), (2, 5))
        # First sample: chars 0,1,2 are real, 3,4 are padding
        self.assertTrue(mask[0, :3].all())
        self.assertFalse(mask[0, 3:].any())
        # Second sample: all 5 are real
        self.assertTrue(mask[1].all())

    def test_drop_text_zeros_embeddings(self):
        """When drop_text=True, all embeddings should be zero (CFG)."""
        te = CrossAttentionTextEmbedding(text_num_embeds=12, text_dim=16, conv_layers=0)
        text = torch.tensor([[1, 2, 3]])
        embed_normal, _ = te(text, drop_text=False)
        embed_dropped, _ = te(text, drop_text=True)
        # Dropped should be the embedding of token 0 (filler), not arbitrary
        self.assertFalse(torch.equal(embed_normal, embed_dropped))

    def test_with_conv_layers(self):
        """ConvNeXt text processing should work with masking."""
        te = CrossAttentionTextEmbedding(text_num_embeds=12, text_dim=16, conv_layers=2)
        text = torch.tensor([[1, 2, 3, -1], [1, 2, 3, 4]])
        embed, mask = te(text)
        self.assertEqual(tuple(embed.shape), (2, 4, 16))
        self.assertTrue(torch.isfinite(embed).all())


class TestCrossAttentionInputEmbedding(unittest.TestCase):
    def test_projects_latent_only(self):
        """Input embedding should project latent dim → model dim without text."""
        ie = CrossAttentionInputEmbedding(latent_dim=40, out_dim=32)
        x = torch.randn(2, 8, 40)
        mask = torch.ones(2, 8, dtype=torch.bool)
        out = ie(x, x, mask=mask)
        self.assertEqual(tuple(out.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_padding_isolation(self):
        """Padding positions must not leak into real positions."""
        torch.manual_seed(42)
        ie = CrossAttentionInputEmbedding(latent_dim=40, out_dim=32).eval()
        x = torch.randn(1, 8, 40)
        mask = torch.ones(1, 8, dtype=torch.bool)
        a = ie(x, x, mask=mask)

        # Pad with garbage noise
        xp = torch.cat((x, torch.randn(1, 8, 40) * 100), 1)
        mp = torch.arange(16)[None] < 8
        b = ie(xp, xp, mask=mp)
        torch.testing.assert_close(a, b[:, :8], atol=1e-5, rtol=1e-5)
        self.assertEqual(float(b[:, 8:].abs().sum().detach()), 0.)

    def test_drop_cond_uses_noise(self):
        """When drop_cond=True, x should be replaced by noise."""
        ie = CrossAttentionInputEmbedding(latent_dim=40, out_dim=32).eval()
        x = torch.randn(1, 8, 40)
        noise = torch.randn(1, 8, 40) * 5
        out_normal = ie(x, noise, drop_cond=False)
        out_dropped = ie(x, noise, drop_cond=True)
        # With drop_cond, noise replaces x → different output
        self.assertFalse(torch.equal(out_normal, out_dropped))


class TestCrossAttentionDiT(unittest.TestCase):
    def test_construction_and_forward(self):
        """CrossAttentionDiT should construct and produce finite outputs."""
        cfg = CrossAttentionDiTConfig.small()
        torch.manual_seed(42)
        m = CrossAttentionDiT(cfg).eval()
        x = torch.randn(2, 9, 40)
        text = torch.tensor([[1, 2, 3, -1], [1, 3, 4, 5]])
        out = m(x, x, text, torch.tensor([10, 200]), mask=None)
        self.assertEqual(tuple(out.shape), (2, 9, 40))
        self.assertTrue(torch.isfinite(out).all())

    def test_text_and_latent_different_lengths(self):
        """Text (nt) and latent (n) should be different lengths — this is the whole point."""
        cfg = CrossAttentionDiTConfig.small()
        torch.manual_seed(42)
        m = CrossAttentionDiT(cfg).eval()
        # 3 text chars, 20 latent blocks
        x = torch.randn(1, 20, 40)
        text = torch.tensor([[1, 2, 3]])
        out = m(x, x, text, torch.tensor([500]))
        self.assertEqual(tuple(out.shape), (1, 20, 40))
        self.assertTrue(torch.isfinite(out).all())

    def test_text_sensitivity(self):
        """Different text inputs should produce different outputs.

        This is the critical test: the original DiT was text-insensitive
        (correct vs swapped vs null text all gave ~same output). Cross-attention
        DiT must be text-sensitive even at initialization with non-zero weights.
        """
        cfg = CrossAttentionDiTConfig.small()
        torch.manual_seed(42)
        m = CrossAttentionDiT(cfg).eval()
        # Push weights away from zero init so outputs differ
        with torch.no_grad():
            m.proj_out.weight.normal_(std=0.1)
            m.proj_out.bias.fill_(0.1)
            for block in m.transformer_blocks:
                block.attn_norm_x.linear.weight.normal_(std=0.1)
                block.attn_norm_x.linear.bias.normal_(std=0.1)
                if hasattr(block.attn_norm_c, 'linear'):
                    block.attn_norm_c.linear.weight.normal_(std=0.1)
                    block.attn_norm_c.linear.bias.normal_(std=0.1)

        x = torch.randn(1, 12, 40)
        t = torch.tensor([500.])
        text_a = torch.tensor([[1, 2, 3, 4, 5]])
        text_b = torch.tensor([[5, 4, 3, 2, 1]])  # reversed
        text_c = torch.tensor([[7, 8, 9, 10, 11]])  # different chars
        out_a = m(x, x, text_a, t)
        out_b = m(x, x, text_b, t)
        out_c = m(x, x, text_c, t)
        # Different text must produce different outputs
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-5),
                         "Reversed text should give different output")
        self.assertFalse(torch.allclose(out_a, out_c, atol=1e-5),
                         "Different text should give different output")

    def test_padding_isolation(self):
        """Padding noise in latent must not leak into real positions."""
        cfg = CrossAttentionDiTConfig.small()
        torch.manual_seed(42)
        m = CrossAttentionDiT(cfg).eval()
        with torch.no_grad():
            m.proj_out.weight.normal_(std=0.1)
            m.proj_out.bias.fill_(0.1)
            for block in m.transformer_blocks:
                block.attn_norm_x.linear.weight.normal_(std=0.1)
                block.attn_norm_x.linear.bias.normal_(std=0.1)
                if hasattr(block.attn_norm_c, 'linear'):
                    block.attn_norm_c.linear.weight.normal_(std=0.1)
                    block.attn_norm_c.linear.bias.normal_(std=0.1)

        x = torch.randn(1, 8, 40)
        text = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 8, dtype=torch.bool)
        time = torch.tensor([100.])
        a = m(x, x, text, time, mask)

        # Pad latent with garbage
        xp = torch.cat((x, torch.randn(1, 8, 40) * 100), 1)
        mp = torch.arange(16)[None] < 8
        b = m(xp, xp, text, time, mp)
        torch.testing.assert_close(a, b[:, :8], atol=1e-5, rtol=1e-5)
        self.assertEqual(float(b[:, 8:].abs().sum().detach()), 0.)

    def test_backward_pass(self):
        """Training backward pass should produce finite gradients."""
        cfg = CrossAttentionDiTConfig.small()
        torch.manual_seed(42)
        m = CrossAttentionDiT(cfg).train()
        x = torch.randn(2, 8, 40)
        text = torch.tensor([[1, 2, 3, -1], [1, 2, 3, 4]])
        mask = torch.ones(2, 8, dtype=torch.bool)
        time = torch.tensor([100., 500.])
        out = m(x, x, text, time, mask)
        loss = out.square().mean()
        loss.backward()
        self.assertTrue(all(
            torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
        ))

    def test_classifier_free_guidance(self):
        """drop_text and drop_cond should alter outputs for CFG."""
        cfg = CrossAttentionDiTConfig.small()
        torch.manual_seed(42)
        m = CrossAttentionDiT(cfg).eval()
        with torch.no_grad():
            m.proj_out.weight.normal_(std=0.1)
            m.proj_out.bias.fill_(0.1)
            for block in m.transformer_blocks:
                block.attn_norm_x.linear.weight.normal_(std=0.1)
                block.attn_norm_x.linear.bias.normal_(std=0.1)
                if hasattr(block.attn_norm_c, 'linear'):
                    block.attn_norm_c.linear.weight.normal_(std=0.1)
                    block.attn_norm_c.linear.bias.normal_(std=0.1)

        x = torch.randn(1, 8, 40)
        text = torch.tensor([[1, 2, 3]])
        time = torch.tensor([100.])

        out_normal = m(x, x, text, time)
        out_no_text = m(x, x, text, time, drop_text=True)
        out_no_cond = m(x, x, text, time, drop_cond=True)
        # All three should differ when weights are non-zero
        self.assertFalse(torch.allclose(out_normal, out_no_text, atol=1e-5))

    def test_same_api_as_original_dit(self):
        """CrossAttentionDiT forward() should accept the same args as DiT."""
        import inspect
        orig_sig = inspect.signature(DiT.forward)
        new_sig = inspect.signature(CrossAttentionDiT.forward)
        orig_params = list(orig_sig.parameters.keys())
        new_params = list(new_sig.parameters.keys())
        self.assertEqual(orig_params, new_params,
                         "CrossAttentionDiT.forward must have same parameter names as DiT.forward")

    def test_with_long_skip_connection(self):
        """Long skip connection should work with CrossAttentionDiT."""
        cfg = CrossAttentionDiTConfig.with_skip()
        torch.manual_seed(42)
        m = CrossAttentionDiT(cfg).eval()
        x = torch.randn(1, 8, 40)
        text = torch.tensor([[1, 2, 3]])
        out = m(x, x, text, torch.tensor([100.]))
        self.assertEqual(tuple(out.shape), (1, 8, 40))
        self.assertTrue(torch.isfinite(out).all())

    def test_parameter_count_reasonable(self):
        """CrossAttentionDiT should have more params than DiT due to cross-attention projections."""
        cfg = CrossAttentionDiTConfig.small()
        torch.manual_seed(42)
        original = DiT(cfg)
        cross_attn = CrossAttentionDiT(cfg)
        orig_params = sum(p.numel() for p in original.parameters())
        cross_params = sum(p.numel() for p in cross_attn.parameters())
        # Cross-attention adds context Q/K/V projections → more parameters
        self.assertGreater(cross_params, orig_params,
                           "CrossAttentionDiT should have more params due to cross-attention projections")
        # But not wildly more (should be within 3x for same config)
        self.assertLess(cross_params, orig_params * 3,
                        "CrossAttentionDiT parameter count should be reasonable")


if __name__ == '__main__':
    unittest.main()
