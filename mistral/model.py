import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from simple_parsing.helpers import Serializable

from mistral.rope import precompute_freqs_cis, apply_rotary_emb
from mistral.cache import CacheView, RotatingBufferCache
from mistral.moe import MoeArgs, MoeLayer

from xformers.ops.fmha import memory_efficient_attention

from spikingjelly.activation_based import surrogate, neuron, functional

import numpy

# torch.set_printoptions(threshold=numpy.inf)


@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: Optional[float] = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: Optional[int] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":   #seqlens -> 整数列表； -> "SimpleInputMetadata" 函数的类型提示   
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)                 # 重复张量的元素: 输入keys, 重复repeats次数, 需要重复的维度dim
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)             # y = w*x + b的线性变换
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)


        self.wq_IF = neuron.IFNode(step_mode='m')
        self.wk_IF = neuron.IFNode(step_mode='m')
        self.wv_IF = neuron.IFNode(step_mode='m')
        self.wo_IF = neuron.IFNode(step_mode='m')



    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:                               # 改成SSA+正负矩阵
        N, C = x.shape                               # x.shape: torch.Size([13, 4096]), torch.Size([2, 4096])  N = 2， C = 409
        # print("N, C ", N, C)

        # 设置scale
        scale = 0.125

        # 设定时间步
        T_pos = 10
        T_neg = 10

        # 得到+/-矩阵 
        x_pos = nn.functional.relu(x)
        x_neg = nn.functional.relu(-x)

        # 根据时间步T,得到时间步数据
        x_pos_ = (x_pos.unsqueeze(0)).repeat(T_pos, 1, 1)
        x_neg_ = (x_neg.unsqueeze(0)).repeat(T_neg, 1, 1)

        # 计算xq， xk， xv
        self.wq_IF.reset()
        self.wk_IF.reset()
        self.wv_IF.reset()
        # print(self.wq(x_pos_).shape, self.wk(x_pos_).shape)
        xq_pos_, xk_pos_, xv_pos_ = self.wq_IF(self.wq(x_pos_)), self.wk_IF(self.wk(x_pos_)), self.wv_IF(self.wv(x_pos_))
        self.wq_IF.reset()
        self.wk_IF.reset()
        self.wv_IF.reset()
        xq_neg_, xk_neg_, xv_neg_ = self.wq_IF(self.wq(x_neg_)), self.wk_IF(self.wk(x_neg_)), self.wv_IF(self.wv(x_neg_))
        xq = xq_pos_ - xq_neg_
        xk = xk_pos_ - xk_neg_
        xv = xv_pos_ - xv_neg_

        xq = xq.view(T_pos, N, self.n_heads, self.head_dim)         # tensor进行reshape torch.Size([33, 32, 128]), torch.Size([2, 32, 128])
        xk = xk.view(T_pos, N, self.n_kv_heads, self.head_dim)      # torch.Size([2, 8, 128])
        xv = xv.view(T_pos, N, self.n_kv_heads, self.head_dim)

        xk, xv = repeat_kv(xk, xv, self.repeats, dim=1)             #  
        xq = xq.reshape(T_pos, N, self.n_heads, C // self.n_heads)
        xk = xk.reshape(T_pos, N, self.n_heads, C // self.n_heads)
        xv = xv.reshape(T_pos, N, self.n_heads, C // self.n_heads)

        # SSA
        attn = (xq @ xk.transpose(-2, -1)) * scale
        x_attn = attn @ xv
        x_attn = x_attn.transpose(-1, -2).reshape(T_pos, N, C)
        xo = self.wo(x_attn)
        
        return xo.mean(0)    # torch.Size([33, 4096]);  torch.Size([2, 4096])  


    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     freqs_cis: torch.Tensor,
    #     cache: Optional[CacheView],
    # ) -> torch.Tensor:
    #     seqlen_sum, _ = x.shape

    #     # 设定时间步
    #     T_pos = 10
    #     T_neg = 10

    #     # 得到+/-矩阵 
    #     x_pos = nn.functional.relu(x)
    #     x_neg = nn.functional.relu(-x)

    #     # 根据时间步T,得到时间步数据
    #     x_pos_ = (x_pos.unsqueeze(0)).repeat(T_pos, 1, 1)
    #     x_neg_ = (x_neg.unsqueeze(0)).repeat(T_neg, 1, 1)

    #     # 计算xq， xk， xv
    #     self.wq_IF.reset()
    #     self.wk_IF.reset()
    #     self.wv_IF.reset()
    #     xq_pos_, xk_pos_, xv_pos_ = self.wq_IF(self.wq(x_pos_)), self.wk_IF(self.wk(x_pos_)), self.wv_IF(self.wv(x_pos_))
    #     self.wq_IF.reset()
    #     self.wk_IF.reset()
    #     self.wv_IF.reset()
    #     xq_neg_, xk_neg_, xv_neg_ = self.wq_IF(self.wq(x_neg_)), self.wk_IF(self.wk(x_neg_)), self.wv_IF(self.wv(x_neg_))
    #     xq = xq_pos_ - xq_neg_
    #     xk = xk_pos_ - xk_neg_
    #     xv = xv_pos_ - xv_neg_
    #     # print("xq: ", xq.shape, "xk: ", xk.shape, "xv: ", xv.shape)
    #     # print("xq: ", xq, "xk: ", xk, "xv: ", xv)


    #     xq = xq.view(T_pos, seqlen_sum, self.n_heads, self.head_dim)         # tensor进行reshape torch.Size([33, 32, 128]), torch.Size([2, 32, 128])
    #     xk = xk.view(T_pos, seqlen_sum, self.n_kv_heads, self.head_dim)      # torch.Size([2, 8, 128])
    #     xv = xv.view(T_pos, seqlen_sum, self.n_kv_heads, self.head_dim)

    #     print(xq.shape, xk.shape, xv.shape)

    #     xq = xq.mean(0)
    #     xk = xk.mean(0)
    #     xv = xv.mean(0)
    #     xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)    # 线性变换 xq:  torch.Size([2, 32, 128])   xq:  torch.Size([2, 8, 128]
        

    #     if cache is None:
    #         key, val = xk, xv
    #     elif cache.prefill:
    #         key, val = cache.interleave_kv(xk, xv)
    #         cache.update(xk, xv)
    #     else:
    #         cache.update(xk, xv)
    #         key, val = cache.key, cache.value
    #         key = key.view(
    #             seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
    #         )
    #         val = val.view(
    #             seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
    #         )

    #     # Repeat keys and values to match number of query heads
    #     key, val = repeat_kv(key, val, self.repeats, dim=1)

    #     # xformers requires (B=1, S, H, D)
    #     xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
    #     # xq_1 = xq - torch.ones_like(xq, device='cuda:0')
    #     # xq_2 = xq - xq_1 
    #     output = memory_efficient_attention(                                      # torch.Size([1, 2, 32, 128])
    #         xq, key, val, None if cache is None else cache.mask
    #     )

    #     print('output.view(seqlen_sum, self.n_heads * self.head_dim): ', output.view(seqlen_sum, self.n_heads * self.head_dim).shape, self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim)).shape)
    #     return self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim))    # torch.Size([33, 4096]);  torch.Size([2, 4096])  


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)


        self.w1_IF = neuron.IFNode(step_mode='m')
        self.w2_IF = neuron.IFNode(step_mode='m')
        self.w3_IF = neuron.IFNode(step_mode='m')



    # def forward(self, x) -> torch.Tensor:
    #     return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))       # silu = x * sigmoid(x)   relu替换silu
    


    def forward(self, x) -> torch.Tensor:
        # 设定时间步
        T_pos = 10
        T_neg = 10

        # 得到+/-矩阵 
        x_pos = nn.functional.relu(x)
        x_neg = nn.functional.relu(-x)

        # 根据时间步T,得到时间步数据
        x_pos_ = (x_pos.unsqueeze(0)).repeat(T_pos, 1, 1)
        x_neg_ = (x_neg.unsqueeze(0)).repeat(T_neg, 1, 1)

        # # 转化为脉冲
        # 计算w3
        self.w3_IF.reset()
        x_w3_pos_ = self.w3_IF(self.w3(x_pos_))
        # self.w3_IF.reset()
        self.w3_IF.reset()
        x_w3_neg_ = self.w3_IF(self.w3(x_neg_))
        x_w3 = x_w3_pos_ - x_w3_neg_
        # 计算w1
        self.w1_IF.reset()
        x_w1_pos_ = self.w1_IF(self.w1(x_pos_))
        # 计算w1 * w3
        # print(x_w1_pos_.mean(0).shape, x_w3.mean(0).shape)

        # return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))         # silu = x * sigmoid(x)   relu替换silu
        return self.w2((x_w1_pos_.mean(0)) * (x_w3.mean(0)))


class RMSNorm(torch.nn.Module):                             # Root Mean Square Layer Normalization(RMSnorm) 
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        # print("self.weight = nn.Parameter(torch.ones(dim)): \n", self.weight)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

        self.feed_forward: nn.Module
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        assert self.vocab_size > 0
        assert pipeline_rank < num_pipeline_ranks, (pipeline_rank, num_pipeline_ranks)
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        # Modules specific to some ranks:
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        if pipeline_rank == 0:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
            x = torch.LongTensor([1]) # 2个句子，每个句子3个词
            # print('self.tok_embeddings: 11111 ', self.tok_embeddings(x))
            # with open('Embedding_weight_1.txt', 'w') as f:
            #     f.writelines(str(self.tok_embeddings.weight))
            # torch.savetxt('Embedding_weight_1.txt', self.tok_embeddings.weight)
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # Initialize all layers but slice off those not of this rank.
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

    @property      # 装饰器，只读属性，防止修改
    def dtype(self) -> torch.dtype:           
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # If no sliding window, assume a larger seqlen
            theta = self.args.rope_theta
            if theta is None:
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            # theta = 10000.
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        # print('self._precomputed_freqs_cis: \n', self._precomputed_freqs_cis.size())
        return self._precomputed_freqs_cis

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        # print("input_ids pre: ", input_ids)
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
            # print("cache: yes", input_metadata)  
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)
            # print("cache: no", input_metadata)    
 
        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)           # torch.Size([26, 4096])
            # print("h = self.tok_embeddings(input_ids): ", h.size(), '\n', h) 
        else:
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)
            # print("h = torch.empty: ", h.size(), '\n', h)  
        # print("h: ", h.shape)

        freqs_cis = self.freqs_cis[input_metadata.positions]

        for local_layer_id, layer in enumerate(self.layers.values()):
            # print("local_layer_id: \n", local_layer_id, "\n layer: \n", layer)
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
                # print("cache_view: ", cache_view)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)    # layer: transformerblock 
            # print("local_layer_id: ", local_layer_id, "\n", "layer: ", layer, "\n", "layer | h: ", h.shape, "\n", h)
            # print(("h layer in : ", h.size()))
            # print("h layer_id: ", local_layer_id, 'layer:', layer, h.size(), "\n", h)
        # print("h layer all: ", h.size(), '\n', h)   

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            # print('torch.distributed.send(h, dst=self.pipeline_rank + 1): ', h.shape, "\n", h)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            # print('self.norm(h): ', self.norm(h).size(), '\n', self.norm(h))
            return self.norm(h)



    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        # print("input_ids: ", input_ids,"\n", 'seqlens:', seqlens)
        h = self.forward_partial(input_ids, seqlens, cache=cache)   # 第一次  torch.Size([26, 4096])； 后面：  torch.Size([1, 4096])  
        # print("h = self.forward_partial(input_ids, seqlens, cache=cache) : ", h.size(), h)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            outs = self.output(h)
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        torch.cuda.empty_cache()
        return outs.float()

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("layers"):
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            else:
                raise ValueError(f"Unexpected key {k}")
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
        super().load_state_dict(state_to_load, *args, **kwargs)

    @staticmethod
    def from_folder(
        folder: Path,
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device="cuda",
        dtype=torch.float16,
    ) -> "Transformer":
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        if num_pipeline_ranks > 1:
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
        with torch.device("meta"):
            model = Transformer(
                model_args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
            )
        loaded = torch.load(str(folder / "consolidated.00.pth"), mmap=True)
        # 查看模型参数
        # print("loaded: ",type(loaded), '\n', loaded)    
        # with open('model_loaded.txt', 'w') as f:
        #     for key in loaded:
        #         f.write('\n\n')
        #         f.writelines(str(key) + ': ' + str(loaded[key].size()) + '\n' + str(loaded[key]))

        model.load_state_dict(loaded, assign=True)
        return model.to(device=device, dtype=dtype)
