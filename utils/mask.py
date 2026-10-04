import torch

def build_prefix_mask_from_char_points(
    char_points_idx,
    mask,  # [B, T]
    compression_factor=4,
    prefix_ratio=0.3,
    max_label_len=None,
    point_seq=None,  # Optional real trajectory for unaligned IAM lines.
    ):
    if len(char_points_idx) != mask.shape[0]:
        raise ValueError("one alignment array is required per sample")
    if any(len(x) == 0 for x in char_points_idx):
        if not all(len(x) == 0 for x in char_points_idx):
            raise ValueError("do not mix character-aligned and stroke-only samples")
        if point_seq is None:
            raise ValueError("unaligned IAM needs point_seq; refusing fake char alignment")
        return build_prefix_mask_from_strokes(point_seq, mask, compression_factor, prefix_ratio)

    B, T = mask.shape
    device = mask.device

    total_lengths = mask.sum(dim=1).tolist()

    full_mask = torch.zeros(B, T, dtype=torch.float32, device=device)

    # 用于生成 prefix_label_mask（字符级别的）
    max_num_chars = max(len(x) for x in char_points_idx) if max_label_len is None else max_label_len
    prefix_label_mask = torch.zeros(B, max_num_chars, dtype=torch.bool, device=device)

    for b in range(B):
        idx_list = char_points_idx[b]
        total_len = total_lengths[b]

        num_chars = len(idx_list)
        num_prefix = max(1, round(num_chars * prefix_ratio))

        # 1. 前缀 mask（特征级别）
        if num_prefix >= num_chars:
            prefix_end = total_len
        else:
            prefix_end = idx_list[num_prefix - 1]

        prefix_end = min(prefix_end, total_len)
        full_mask[b, :prefix_end] = 1.0  # 前缀区域设为 1

        # 2. 前缀标签 mask（字符级别）
        prefix_label_mask[b, :num_prefix] = 1  # 前缀字符设为 1（应忽略）

    latent_mask = downsample_mask(full_mask, compression_factor)
    latent_mask = 1.0 - latent_mask  # 前缀为 0，后缀为 1

    pad_mask = downsample_mask(mask, compression_factor)  # 1=有效，0=padding

    return latent_mask, pad_mask, 1.0 - full_mask  # [B, T_latent], [B, T_latent], [B, L]

def downsample_mask(mask, compression_factor):
    B, T = mask.shape
    valid_T = (T // compression_factor) * compression_factor
    mask = mask[:, :valid_T]  # 保证整除
    downsampled = mask.reshape(B, -1, compression_factor).float().mean(dim=2)
    return (downsampled > 0.0).float()  # 二值化

def build_prefix_mask_from_strokes(point_seq, mask, compression_factor=8, prefix_ratio=0.3):
    """IAM fallback: exclusive stroke end nearest target fraction of real points.

    Returns the same THREE temporal masks as the released char-prefix helper.
    No character boundaries or labels are fabricated. No full-line reference
    when prefix_ratio < 1; a one-stroke line gets an empty reference.
    """
    if not 0 <= prefix_ratio <= 1:
        raise ValueError("prefix_ratio must lie in [0, 1]")
    if compression_factor < 1 or mask.shape[1] % compression_factor:
        raise ValueError("sequence must be padded to the compression multiple")
    B, T = mask.shape
    if point_seq.shape == (B, 5, T):
        point_seq = point_seq.transpose(1, 2)
    if point_seq.shape != (B, T, 5):
        raise ValueError("expected B×T×5 or B×5×T trajectory")
    full_mask = torch.zeros(B, T, dtype=torch.float32, device=mask.device)
    for b in range(B):
        n = int(mask[b].sum().item())
        if not bool(mask[b, :n].all()) or bool(mask[b, n:].any()):
            raise ValueError("valid mask must be a contiguous prefix")
        ends = torch.nonzero(point_seq[b, :n, 2] == 0, as_tuple=False).flatten() + 1
        if n and (not len(ends) or int(ends[-1]) != n):
            raise ValueError("trajectory lacks final stroke boundary")
        if prefix_ratio == 0 or n == 0:
            cutoff = 0
        elif prefix_ratio == 1:
            cutoff = n
        else:
            candidates = ends[ends < n]
            cutoff = (int(candidates[torch.argmin(torch.abs(candidates.float() - n * prefix_ratio))])
                      if len(candidates) else 0)
        full_mask[b, :cutoff] = 1
    return (1.0 - downsample_mask(full_mask, compression_factor),
            downsample_mask(mask, compression_factor), 1.0 - full_mask)
