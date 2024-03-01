import torch
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    '''计算旋转位置编码矩阵
        形成一个[end, dim/2]的矩阵，每一行是一个旋转位置向量，每一列是一个维度的旋转位置向量
    '''
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # length = 64
    t = torch.arange(end, device=freqs.device)  # length = 128_000
    freqs = torch.outer(t, freqs).float()  # torch.Size([128000, 64])
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)                    # 按照给定xq的类型转换xq_out的类型
