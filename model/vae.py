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
            num_heads=config.trans_num_heads
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
        
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        features = self.encoder(x)
        mu = self.conv_mu(features)
        logvar = self.conv_logvar(features)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, x):
        decoded = self.decoder(x)
        output = self.transformer_decoder(decoded)
        return output

    def forward(self, data, pad_mask, labels, writer_labels, get_ctc_loss=True, get_style_loss=True):
        # encoder and decode
        z, mu, logvar = self.encode(data)
        decoded = self.decoder(z)
        output = self.transformer_decoder(decoded)

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
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, T, C]
        kl = kl.sum(dim=1)  # [B, T]

        if mask is not None:
            kl = kl * mask  # apply mask to ignore padded regions
            loss = kl.sum() / (mask.sum() * kl.size(-1))
        else:
            loss = kl.mean()
        
        # 4. Final loss clamp to avoid explosion
        loss = torch.clamp(loss, max=1e4)

        return loss

    
    def get_style_loss(self, z, writer_labels, mask):
        writer_logits = self.style_classifier(z, mask)  # [B, num_writers]
        loss = F.cross_entropy(writer_logits, writer_labels)
        return loss

    def get_ocr_loss(self, features, labels, mask=None):
        # VAE.forward uses this method, not a standalone OCR helper test.
        # Share the implementation so the two paths cannot drift again.
        return self.ocr_model.get_ocr_loss(features, labels, mask)

    @torch.no_grad()
    def val(self, data):
        z, mu, logvar = self.encode(data)
        decoded = self.decoder(z)
        output = self.transformer_decoder(decoded)

        return output









