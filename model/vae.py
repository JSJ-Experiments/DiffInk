import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import Encoder, Decoder, TransformerDecoder
from .ocr import ChineseHandwritingOCR
from model.writer import WriterStyleClassifier
from utils.utils import ModelConfig

class VAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(in_channels=config.in_channels, hidden_dims=config.hidden_dims)
        self.conv_mu = nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=1)

        self.decoder = Decoder(hidden_dims=config.decoder_dims)
        self.transformer_decoder = TransformerDecoder(
            input_dim=config.decoder_dims[-1],
            hidden_dim=config.trans_hidden_dim,
            output_dim=config.decoder_output_dim,
            num_layers=config.trans_num_layers,
            num_heads=config.trans_num_heads,
            dropout=getattr(config, "trans_dropout", 0.1)
        )
        self.ocr_model = ChineseHandwritingOCR(
            input_dim=config.latent_dim,
            hidden_dim=config.ocr_hidden_dim,
            num_heads=config.ocr_num_heads,
            num_layers=config.ocr_num_layers,
            num_classes=config.num_text_embedding
        )
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

        self.style_classifier = WriterStyleClassifier(
            input_dim=config.style_classifier_dim,
            num_writers=config.num_writer
        )
        
    def apply_checkpoint_contract(self, checkpoint):
        saved = checkpoint.get('config')
        if saved is None:
            if getattr(self.config, 'language', None) == 'en':
                raise ValueError('English VAE checkpoint missing scale/config contract')
            return
        for key in ('model_input_scale', 'trans_dropout', 'use_decoder_padding_mask', 'pen_policy'):
            if key in saved: setattr(self.config, key, saved[key])
        dropout = float(getattr(self.config, 'trans_dropout', .1))
        for layer in self.transformer_decoder.transformer.layers:
            layer.self_attn.dropout = dropout
            for module in layer.modules():
                if isinstance(module, nn.Dropout): module.p = dropout

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def to_model_space(self, x):
        """Public encode/forward accept raw HDF5 XY; decoder emits MODEL-space XY."""
        if x.ndim != 3 or x.shape[1] != 5:
            raise ValueError('expected B x 5 x T trajectory')
        scale = float(getattr(self.config, 'model_input_scale', 1.0))
        if not 0 < scale <= 1:
            raise ValueError('model_input_scale must be in (0,1]')
        result = x.clone(); result[:, :2] *= scale
        return result

    def to_data_space(self, sequence):
        """Inverse adapter for N x 5 rendered trajectories; pen states unchanged."""
        result = sequence.clone() if torch.is_tensor(sequence) else sequence.copy()
        result[..., :2] /= float(getattr(self.config, 'model_input_scale', 1.0))
        return result

    def encode(self, x, input_is_model_space=False):
        x = x if input_is_model_space else self.to_model_space(x)
        features = self.encoder(x)
        mu = self.conv_mu(features)
        logvar = self.conv_logvar(features)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, x, padding_mask=None):
        decoded = self.decoder(x)
        output = self.transformer_decoder(decoded, padding_mask=padding_mask)
        return output

    def forward(self, data, pad_mask, labels, writer_labels, get_ctc_loss=True, get_style_loss=True, point_mask=None, input_is_model_space=False):
        # encoder and decode
        z, mu, logvar = self.encode(data, input_is_model_space=input_is_model_space)
        if getattr(self.config, 'use_decoder_padding_mask', False):
            if point_mask is None:
                sentinel = data.new_tensor([0, 0, 0, 0, 1])[None, :, None]
                point_mask = ~(data == sentinel).all(dim=1)
            padding_mask = ~point_mask.bool()
        else:
            padding_mask = None
        output = self.decode(z, padding_mask=padding_mask)

        # ocr loss and kl loss
        kl_loss = self.kl_divergence_new(mu, logvar, pad_mask)
        if get_ctc_loss:
            ctc_loss = self.get_ocr_loss(z, labels, pad_mask)
        else:
            ctc_loss = torch.tensor(0.0, requires_grad=False, device=data.device)
        
        # style loss
        if get_style_loss:
            style_loss = self.get_style_loss(z, writer_labels, pad_mask)
        else:
            style_loss = torch.tensor(0.0, requires_grad=False, device=data.device)

        return output, ctc_loss, kl_loss, style_loss
    
    def kl_divergence(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = torch.clamp(kl_loss, min=0.0, max=10.0)
        return kl_loss
    
    def kl_divergence_new(self, mu, logvar, mask=None):
        # B,C,T: average over actual valid latent ELEMENTS, not C-summed / T.
        if mu.shape != logvar.shape:
            raise ValueError('mu/logvar shape mismatch')
        if mask is None:
            selected_mu, selected_logvar = mu, logvar
        else:
            if mask.shape != (mu.shape[0], mu.shape[2]) or not mask.bool().any():
                raise ValueError('nonempty matching latent mask required')
            valid = mask.bool().unsqueeze(1).expand_as(mu)
            selected_mu, selected_logvar = mu[valid], logvar[valid]
        loss = (-0.5 * (1 + selected_logvar - selected_mu.square() - selected_logvar.exp())).mean()
        return loss.clamp(max=1e4)

    def get_style_loss(self, z, writer_labels, mask):
        writer_logits = self.style_classifier(z, mask)  # [B, num_writers]
        loss = F.cross_entropy(writer_logits, writer_labels)
        return loss

    def get_ocr_loss(self, features, labels, mask=None):
        # VAE.forward uses this method, not a standalone OCR helper test.
        # Share the implementation so the two paths cannot drift again.
        return self.ocr_model.get_ocr_loss(features, labels, mask)

    @torch.no_grad()
    def val(self, data, point_mask=None, latent_mean=True, input_is_model_space=False):
        z, mu, _ = self.encode(data, input_is_model_space=input_is_model_space)
        padding_mask = ~point_mask.bool() if point_mask is not None and getattr(self.config, 'use_decoder_padding_mask', False) else None
        return self.decode(mu if latent_mean else z, padding_mask=padding_mask)
