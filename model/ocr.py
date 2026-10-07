import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index
        pe = pe.unsqueeze(0)  # [1, T, D] for broadcasting

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return x


class ChineseHandwritingOCR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        # self.input_norm = nn.LayerNorm(hidden_dim)
        # self.output_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_fc = nn.Linear(hidden_dim, num_classes)

        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

        self.run()

    # 抑制模型输出空白-0
    def run(self):
        with torch.no_grad():
            self.output_fc.bias.data.zero_()
            self.output_fc.bias[0].copy_(-5.0)

    def forward(self, x, padding_mask=None):
        """
        x: Tensor [B, C, T] — from frozen VAE encoder
        returns: [T, B, num_classes] — for CTCLoss
        """
        if padding_mask is not None:
            if padding_mask.shape != (x.shape[0], x.shape[2]) or padding_mask.dtype != torch.bool:
                raise ValueError("OCR padding_mask must be Boolean [B,T], True = ignored")
            if padding_mask.all(dim=1).any():
                raise ValueError("OCR attention requires at least one real timestep per line")
            # An attention mask alone cannot prevent padded NaNs leaking through
            # Q/K/V matrix products. Remove invalid features before projection.
            x = x.masked_fill(padding_mask[:, None, :], 0.)
        x = x.permute(0, 2, 1)          # [B, T, C]
        x = self.input_proj(x)          # [B, T, H]
        # x = self.input_norm(x)          # 输入 LayerNorm
        x = self.pos_encoder(x)         # [B, T, H]
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, T, H]
        # x = self.output_norm(x)         # 输出 LayerNorm
        x = self.output_fc(x)           # [B, T, num_classes]
        return x.permute(1, 0, 2)       # [T, B, num_classes]
    
    def get_ocr_loss(self, features, labels, mask=None):
        if mask is None:
            mask = torch.ones(features.shape[0], features.shape[2], dtype=torch.bool, device=features.device)
        if mask.shape != (features.shape[0], features.shape[2]):
            raise ValueError("OCR valid mask must have shape [B,T]")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("OCR valid mask must contain only 0/1")
        mask = mask.bool()  # Released downsample_mask returns binary floats.
        # CTC assumes a contiguous real prefix, unlike general attention masks.
        if ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError("CTC valid mask must be a right-padded prefix")
        outputs = self.forward(features, padding_mask=~mask)  # [T, B, C]
        outputs = torch.clamp(outputs, -30.0, 30.0)
        log_probs = outputs.log_softmax(2)

        # 排除padding和前缀label（如果有）
        labels = labels + 1
        labels[labels == 0] = -100

        input_lengths = mask.sum(dim=1).to(torch.long)
        target_lengths = (labels != -100).sum(dim=1).to(torch.long)

        # CTC needs one extra timestep only for adjacent repeated labels.
        # 2 * target_len - 1 is sufficient but NOT necessary, and incorrectly
        # discards many cursive English lines after the VAE's 8x compression.
        repeats = ((labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] != -100)).sum(dim=1)
        required_lengths = target_lengths + repeats
        valid_mask = (target_lengths > 0) & (input_lengths >= required_lengths)

        if valid_mask.any():
            input_lengths = input_lengths[valid_mask]
            target_lengths = target_lengths[valid_mask]
            log_probs = log_probs[:, valid_mask, :]
            labels = labels[valid_mask]

            try:
                loss = self.ctc(log_probs, labels, input_lengths.cpu(), target_lengths.cpu())
            except Exception as e:
                print("🔥 CTC.backward() failed")
                print("input_lengths:", input_lengths)
                print("target_lengths:", target_lengths)
                print("log_probs shape:", log_probs.shape)
                print("labels shape:", labels.shape)
                raise e
        else:
            # Preserve graph connectivity without a standalone leaf constant.
            loss = log_probs.sum() * 0.0

        return loss
    