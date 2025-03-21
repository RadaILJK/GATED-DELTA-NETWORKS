from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


def layer_norm_fwd(
    x: torch.Tensor,
    o: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False
):
    """
    Прямой проход для Layer Normalization или RMS Normalization на CPU.

    Аргументы:
        x (torch.Tensor): Входной тензор размера (M, N).
        o (torch.Tensor): Тензор "гейта" размера (M, N).
        weight (torch.Tensor): Тензор весов размера (N,). Опционально.
        bias (torch.Tensor): Тензор смещений размера (N,). Опционально.
        eps (float): Эпсилон для численной стабильности.
        residual (torch.Tensor): Тензор остаточного соединения размера (M, N). Опционально.
        out_dtype (torch.dtype): Тип данных для выходного тензора. По умолчанию совпадает с типом входа.
        residual_dtype (torch.dtype): Тип данных для остаточного тензора. Опционально.
        is_rms_norm (bool): Использовать ли RMS Normalization вместо Layer Normalization.

    Возвращает:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        - y: Выходной тензор после нормализации и применения гейта.
        - mean: Среднее значение входа (None для RMSNorm).
        - rstd: Обратное стандартное отклонение.
        - residual_out: Обновленный тензор остаточного соединения (или входной тензор, если остаток не задан).
    """
    # Проверяем корректность размеров тензоров
    M, N = x.shape
    if residual is not None:
        assert residual.shape == (M, N), "Тензор остаточного соединения должен иметь тот же размер, что и вход."
    if weight is not None:
        assert weight.shape == (N,), "Тензор весов должен иметь размер (N,)."
    if bias is not None:
        assert bias.shape == (N,), "Тензор смещений должен иметь размер (N,)."

    # Создаем выходные тензоры
    y = torch.empty_like(x, dtype=out_dtype if out_dtype is not None else x.dtype)
    residual_out = residual if residual is not None else x
    mean = None if is_rms_norm else torch.zeros((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    # Обрабатываем каждую строку входного тензора
    for row in range(M):
        # Извлекаем текущую строку
        x_row = x[row]
        o_row = o[row]

        # Добавляем остаточное соединение, если оно предоставлено
        if residual is not None:
            x_row += residual[row]

        # Вычисляем среднее и дисперсию
        if not is_rms_norm:
            mean[row] = x_row.mean()  # Среднее значение
            x_centered = x_row - mean[row]  # Центрируем данные
            var = (x_centered ** 2).mean()  # Дисперсия
        else:
            var = (x_row ** 2).mean()  # Для RMSNorm вычисляем только дисперсию

        # Вычисляем обратное стандартное отклонение
        rstd[row] = 1 / torch.sqrt(var + eps)

        # Нормализуем данные
        x_hat = x_row * rstd[row] if is_rms_norm else (x_row - mean[row]) * rstd[row]

        # Применяем веса и смещения
        if weight is not None:
            x_hat = x_hat * weight
        if bias is not None:
            x_hat = x_hat + bias

        # Применяем механизм гейтинга
        y_row = x_hat * o_row * torch.sigmoid(o_row)

        # Сохраняем результат
        y[row] = y_row

    return y, mean, rstd, residual_out


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    o: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
):
    """
    Обратный проход для Layer Normalization или RMS Normalization на CPU.

    Args:
        dy (torch.Tensor): Градиент по выходу размера (M, N).
        x (torch.Tensor): Входной тензор размера (M, N).
        o (torch.Tensor): Тензор "гейта" размера (M, N).
        weight (torch.Tensor): Веса для нормализации размера (N,). Опционально.
        bias (torch.Tensor): Смещения для нормализации размера (N,). Опционально.
        eps (float): Эпсилон для численной стабильности.
        mean (torch.Tensor): Среднее значение входа размера (M,). Опционально.
        rstd (torch.Tensor): Обратное стандартное отклонение размера (M,). Опционально.
        dresidual (torch.Tensor): Градиент по остаточному соединению размера (M, N). Опционально.
        has_residual (bool): Флаг наличия остаточного соединения.
        is_rms_norm (bool): Использовать ли RMS Normalization вместо Layer Normalization.
        x_dtype (torch.dtype): Тип данных для входного тензора. Опционально.
        recompute_output (bool): Флаг для пересчета выхода.

    Returns:
        Tuple[torch.Tensor]: Градиенты по входным данным, весам, смещениям и другим параметрам.
    """
    # Проверяем корректность размеров
    M, N = x.shape
    assert dy.shape == (M, N), "Градиент по выходу должен иметь размер (M, N)."
    if dresidual is not None:
        assert dresidual.shape == (M, N), "Градиент по остатку должен иметь размер (M, N)."
    if weight is not None:
        assert weight.shape == (N,), "Веса должны иметь размер (N,)."
    if bias is not None:
        assert bias.shape == (N,), "Смещения должны иметь размер (N,)."

    # Выделяем память для градиентов
    dx = torch.zeros_like(x, dtype=x_dtype if x_dtype is not None else x.dtype)
    do = torch.zeros_like(o, dtype=x_dtype if x_dtype is not None else o.dtype)
    dw = torch.zeros_like(weight) if weight is not None else None
    db = torch.zeros_like(bias) if bias is not None else None
    dresidual_in = torch.zeros_like(x) if has_residual and dx.dtype != x.dtype else None

    # Пересчитываем выход, если требуется
    y = torch.zeros((M, N), dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Обрабатываем каждую строку
    for row in range(M):
        x_row = x[row]
        o_row = o[row]
        dy_row = dy[row]

        # Нормализация
        if not is_rms_norm:
            xhat = (x_row - mean[row]) * rstd[row]
        else:
            xhat = x_row * rstd[row]

        # Применяем веса и смещения
        y_row = xhat * weight if weight is not None else xhat
        if bias is not None:
            y_row += bias

        # Пересчитываем выход, если требуется
        if recompute_output:
            y[row] = y_row

        # Градиент по гейту
        sigmoid_o = torch.sigmoid(o_row)
        do_row = dy_row * y_row * (sigmoid_o + o_row * sigmoid_o * (1 - sigmoid_o))
        do[row] = do_row

        # Градиент по выходу
        wdy = dy_row * o_row * sigmoid_o
        if weight is not None:
            wdy *= weight

        # Градиент по весам и смещениям
        if weight is not None:
            dw += wdy * xhat
        if bias is not None:
            db += wdy

        # Градиент по входу
        if not is_rms_norm:
            c1 = (xhat * wdy).sum() / N
            c2 = wdy.sum() / N
            dx_row = (wdy - (xhat * c1 + c2)) * rstd[row]
        else:
            c1 = (xhat * wdy).sum() / N
            dx_row = (wdy - xhat * c1) * rstd[row]

        # Добавляем градиент по остаточному соединению
        if has_residual:
            dx_row += dresidual[row]

        dx[row] = dx_row

    # Обновляем градиенты по остаточному соединению
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx

    return (dx, do, dw, db, dresidual_in) if not recompute_output else (dx, do, dw, db, dresidual_in, y)


class LayerNormSwishGateFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        o,
        weight,
        bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        """
        Прямой проход для Layer Normalization с механизмом гейтинга на CPU.

        Args:
            x (torch.Tensor): Входной тензор.
            o (torch.Tensor): Тензор "гейта".
            weight (torch.Tensor): Веса для нормализации.
            bias (torch.Tensor): Смещения для нормализации.
            residual (torch.Tensor): Остаточное соединение. Опционально.
            eps (float): Эпсилон для численной стабильности.
            prenorm (bool): Флаг для режима pre-norm.
            residual_in_fp32 (bool): Использовать ли FP32 для остаточного соединения.
            is_rms_norm (bool): Использовать ли RMS Normalization.

        Returns:
            torch.Tensor: Выходной тензор.
        """
        # Сохраняем исходную форму входных данных
        x_shape_og = x.shape
        o_shape_og = o.shape

        # Преобразуем входные данные в 2D тензоры
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])

        # Обрабатываем остаточное соединение
        if residual is not None:
            assert residual.shape == x_shape_og, "Форма остаточного тензора должна совпадать с формой входа."
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )

        # Вызываем прямой проход нормализации
        y, mean, rstd, residual_out = layer_norm_fwd(
            x, o, weight, bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm
        )

        # Сохраняем контекст для обратного прохода
        ctx.save_for_backward(residual_out, o, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype

        # Возвращаем результат в исходной форме
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    def backward(ctx, dy, *args):
        """
        Обратный проход для Layer Normalization с механизмом гейтинга на CPU.

        Args:
            ctx: Контекст, сохраненный из прямого прохода.
            dy (torch.Tensor): Градиент по выходу.
            args: Дополнительные аргументы (например, градиент по остатку).

        Returns:
            Tuple[torch.Tensor]: Градиенты по входным данным, весам и другим параметрам.
        """
        # Извлекаем сохраненные значения из контекста
        residual_out, o, weight, bias, mean, rstd = ctx.saved_tensors

        # Преобразуем градиенты в 2D тензоры
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == residual_out.shape

        # Обрабатываем градиент по остаточному соединению
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == residual_out.shape
        else:
            dresidual = None

        # Вычисляем градиенты
        dx, do, dw, db, dresidual_in = layer_norm_bwd(
            dy,
            residual_out,
            o,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )

        # Возвращаем градиенты в исходной форме
        return (
            dx.reshape(ctx.x_shape_og),
            do.reshape(ctx.o_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def rms_norm_swish_gate_fn(
    x,
    o,
    weight,
    bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6
):
    return LayerNormSwishGateFn.apply(
        x,
        o,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True
    )


class FusedRMSNormSwishGate(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> FusedRMSNormSwishGate:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_swish_gate_fn(
            x,
            o,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )


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