from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from gated_delta_rule_ops import chunk_gated_delta_rule
from rmsnorms import RMSNorm, FusedRMSNormSwishGate
import math

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

if TYPE_CHECKING:
    from torch import Tensor


class ShortConvolution(nn.Conv1d):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    This implementation is designed to work exclusively on CPU without CUDA dependencies.
    """

    def __init__(
            self,
            hidden_size: int,
            kernel_size: int,
            bias: bool = False,
            activation: Optional[str] = 'silu',
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        return s.format(**self.__dict__)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            cache: Optional[torch.Tensor] = None,
            output_final_state: bool = False,
            seq_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[batch_size, seq_len, hidden_size]`
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[batch_size, hidden_size, kernel_size]`.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[batch_size, hidden_size, kernel_size]`. Default: `False`.
            seq_idx (Optional[torch.Tensor]):
                Sequence index for each token. Used for varlen. Default: `None`.
                Shape: [batch_size, seq_len]
        Returns:
            Tensor of shape `[batch_size, seq_len, hidden_size]`.
        """
        # print('x.shape in ShortConvolution forward', x.shape)
        batch_size, _, hidden_size = x.shape
        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))
        if output_final_state and cache is None:
            cache = x.new_zeros(batch_size, hidden_size, self.kernel_size[0])
        if cache is not None and x.shape[1] == 1:
            return self.step(x, cache)

        x = rearrange(x, "b t d -> b d t")
        # Update state (B D W)
        if cache is not None:
            cache.copy_(F.pad(x, (self.kernel_size[0] - x.shape[-1], 0)))

        # Perform convolution using PyTorch's native implementation
        x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
        if self.activation is not None:
            x = ACT2FN[self.activation](x)

        return rearrange(x, "b d t -> b t d"), cache

    def step(
            self,
            x: torch.Tensor,
            cache: torch.Tensor
    ):
        assert x.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        x = x.squeeze(1)
        dtype = x.dtype
        cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
        cache[:, :, -1] = x
        x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
        if self.bias is not None:
            x = x + self.bias
        if self.activation is not None:
            x = ACT2FN[self.activation](x).to(dtype=dtype)
        return x.unsqueeze(1), cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]


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
        mode: str = 'chunk',    # Режим работы модели
        hidden_size: int = 1024,
        expand_k: float = 0.75,  # Коэффициент расширения для ключей
        expand_v: float = 1.5,   # Коэффициент расширения для значений
        num_heads: int = 9,  # Количество голов внимания
        num_kv_heads: Optional[int] = None,
        qk_norm: str = 'l2',
        conv_size: int = 4,
        conv_bias: bool = False,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,  # Для стабилизации нормализации
        gate_logit_normalizer: int = 16,
        fuse_norm: bool = True,
        layer_idx: int = None,
        use_mamba_gate: bool = True,
        use_mva: bool = False,
        use_residual: bool = False,
        use_input_gate: bool = False,
        vocab_size: int = 50257, # Добавлено - размер словаря
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

        # Эмбеддинг-слой для входа и слой для выхода
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu' if self.qk_norm != 'softmax' else None)
        self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu' if self.qk_norm != 'softmax' else None)
        self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        self.gk_proj = nn.Linear(hidden_size, self.num_heads, bias=not use_mamba_gate)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        if gate_fn == 'swish' and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
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

        # print(hidden_states.shape)

        hidden_states = self.embedding(hidden_states)  # (batch_size, sequence_length, hidden_size)

        # Применяем линейные проекции для получения запросов, ключей и значений
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Применяем свертки к запросам, ключам и значениям
        q = self.q_conv1d(q, attention_mask, conv_state_q)[0]
        k = self.k_conv1d(k, attention_mask, conv_state_k)[0]
        v = self.v_conv1d(v, attention_mask, conv_state_v)[0]

        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        # Вычисляем гейты ("ворота")
        gk = self.gk_proj(hidden_states).float()

        if self.use_mamba_gate:
            gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)
        else:
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        gk = gk.transpose(1, 2)

        # Вычисляем коэффициент beta
        beta = self.b_proj(hidden_states).float().sigmoid()
        beta = beta.transpose(1, 2)

        # Реорганизуем запросы, ключи и значения для многоголового внимания
        # (batch_size, sequence_length, hidden_size) -> (batch_size, num_heads, sequence_length, head_dim)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        if self.num_kv_groups > 1:
            # Если количество групп ключей и значений больше 1, повторяем их для каждой группы
            k, v = (repeat(x, 'b l (h d) -> b (h g) l d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v))
        else:
            # Иначе просто реорганизуем
            k, v = (rearrange(x, 'b l (h d) -> b h l d', h=self.num_kv_heads) for x in (k, v))

        # Применяем нормализацию к запросам и ключам
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

        # Если используется входной гейт, применяем его к значениям
        if self.use_input_gate:
            original_v_dtype = v.dtype
            v = (v * (1 - gk.float().exp())[..., None]).to(original_v_dtype)

        recurrent_state = last_state[-1] if use_cache else None

        # Применяем соответствующий механизм внимания
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

        # Если используются остаточные связи, добавляем их к выходным значениям
        if self.use_residual:
            o = o + self.D[None, :, None, None] * v
        o = rearrange(o, 'b h l d -> b l h d')

        # Применяем проекцию для гейтов
        g = self.g_proj(hidden_states)
        if self.fuse_norm_and_gate:
            # Если нормализация и гейт объединены, применяем их вместе
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)

        # Применяем финальную проекцию и преобразуем выходные значения в логиты
        o = self.o_proj(o)
        o = self.lm_head(o)
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