from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from gated_delta_rule_ops import chunk_gated_delta_rule
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math

if TYPE_CHECKING:
    from torch import Tensor

class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

'''
class FusedRMSNorm(torch.nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.dim = dim
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)
'''

class ShortConvolution(nn.Module):
    """Реализация короткой свёртки на PyTorch."""
    def __init__(self, dim: int, kernel_size: int, activation: Optional[str] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if self.activation == 'silu':
            x = F.silu(x)
        return x

class CustomGate(nn.Module):
    def __init__(self, head_v_dim, norm_eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(head_v_dim, eps=norm_eps, elementwise_affine=elementwise_affine)
        self.activation = nn.SiLU()

    def forward(self, o, g):
        o = self.norm(o)
        o = self.activation(o)
        # print(g.shape, o.shape)
        if g.shape != o.shape:
            g = F.pad(g, (0, 0, 0, 0, 0, 3))
        o = o * g  # Комбинируем o и g
        return o


ACT2FN = {
    "swish": nn.SiLU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "silu": nn.SiLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    # Добавьте другие функции активации по необходимости
}

class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.75,
        expand_v: float = 1.5,
        num_heads: int = 9,
        num_kv_heads: Optional[int] = None,
        qk_norm: str = 'l2',
        conv_size: int = 4,
        conv_bias: bool = False,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        fuse_norm: bool = True,
        layer_idx: int = None,
        use_mamba_gate: bool = True,
        use_mva: bool = False,
        use_residual: bool = False,
        use_input_gate: bool = False,
    ) -> GatedDeltaNet:
        super().__init__()
        self.qk_norm = qk_norm
        assert self.qk_norm in ['l2', 'longhorn', 'softmax']

        self.use_mva = use_mva

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.layer_idx = layer_idx

        assert mode in ['chunk'], f"Not supported mode `{mode}`."

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, padding=conv_size-1, bias=conv_bias)
        self.k_conv1d = nn.Conv1d(self.key_dim_per_group, self.key_dim_per_group, conv_size, padding=conv_size-1, bias=conv_bias)
        self.v_conv1d = nn.Conv1d(self.value_dim_per_group, self.value_dim_per_group, conv_size, padding=conv_size-1, bias=conv_bias)
        self.gk_proj = nn.Linear(hidden_size, self.num_heads, bias=not use_mamba_gate)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        if gate_fn == 'swish' and fuse_norm:
            '''
            self.g_norm_swish_gate = nn.Sequential(
                nn.LayerNorm(self.head_v_dim, eps=norm_eps, elementwise_affine=elementwise_affine),
                nn.SiLU()
            )
            '''
            self.g_norm_swish_gate = CustomGate(head_v_dim=self.head_v_dim, norm_eps=norm_eps,
                                                elementwise_affine=elementwise_affine)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = nn.LayerNorm(self.head_v_dim, eps=norm_eps, elementwise_affine=elementwise_affine)
            self.gate_fn = ACT2FN[gate_fn]
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_mamba_gate = use_mamba_gate
        if use_mamba_gate:
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            self.D = nn.Parameter(torch.ones(self.num_heads))
            self.D._no_weight_decay = True
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

        self.use_residual = use_residual
        if self.use_residual:
            self.D = nn.Parameter(torch.ones(self.num_heads))
            self.D._no_weight_decay = True
        self.use_input_gate = use_input_gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        conv_state_q = last_state[0] if use_cache else None
        conv_state_k = last_state[1] if use_cache else None
        conv_state_v = last_state[2] if use_cache else None

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self.q_conv1d(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_conv1d(k.transpose(1, 2)).transpose(1, 2)
        v = self.v_conv1d(v.transpose(1, 2)).transpose(1, 2)

        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        gk = self.gk_proj(hidden_states).float()
        if self.use_mamba_gate:
            gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)
        else:
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        gk = gk.transpose(1, 2)

        beta = self.b_proj(hidden_states).float().sigmoid()
        beta = beta.transpose(1, 2)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v = (repeat(x, 'b l (h d) -> b (h g) l d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v))
        else:
            k, v = (rearrange(x, 'b l (h d) -> b h l d', h=self.num_kv_heads) for x in (k, v))

        assert self.qk_norm is not None
        if self.qk_norm == 'l2':
            q = F.normalize(q, p=2, dim=-1).to(v)
            k = F.normalize(k, p=2, dim=-1).to(v)
        elif self.qk_norm == 'softmax':
            k = k.softmax(dim=-1).to(v)
            q = q.softmax(dim=-1).to(v)
        elif self.qk_norm == 'longhorn':
            beta = beta / (1 + beta * (k * k).sum(-1))
        else:
            raise KeyError

        if self.use_input_gate:
            original_v_dtype = v.dtype
            v = (v * (1 - gk.float().exp())[..., None]).to(original_v_dtype)

        recurrent_state = last_state[-1] if use_cache else None

        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(q, k, v, beta, gk, initial_state=recurrent_state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if self.use_short_conv:
                last_state = (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        if self.use_residual:
            o = o + self.D[None, :, None, None] * v
        o = rearrange(o, 'b h l d -> b l h d')

        g = self.g_proj(hidden_states)
        if self.fuse_norm_and_gate:
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()

        state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                  param.new_zeros(batch_size, self.key_dim, self.conv_size),
                  param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        for module in self.children():
            if isinstance(module, nn.Conv1d):
                state_size += module.kernel_size[0] * module.in_channels
        return state_size