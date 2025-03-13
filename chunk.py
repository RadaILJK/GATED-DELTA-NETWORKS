import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

def cdiv(a, b):
    """Вычисляет деление с округлением вверх."""
    return (a + b - 1) // b

def next_power_of_2(x):
    """Возвращает следующую степень двойки, которая больше или равна x."""
    if x == 0:
        return 1
    return 1 << (x - 1).bit_length()

def fwd_prepare_wy_repr(k, v, beta, g, BT):
    """
    Подготавливает представление WY для вычислений на CPU.

    Аргументы:
        k: Тензор ключей, форма (B, H, T, K).
        v: Тензор значений, форма (B, H, T, V).
        beta: Тензор масштабирования, форма (B, H, T, K).
        g: Тензор для управления временными шагами, форма (B, H, T).
        BT: Размер временного блока (chunk size).

    Возвращает:
        w: Промежуточный тензор, форма (B, H, T, K).
        u: Промежуточный тензор, форма (B, H, T, V).
        A_w: Тензор для WY-представления, форма (B, H, T, BT).
        A_u: Тензор для WY-представления, форма (B, H, T, BT).
        A_w_original: Оригинальный тензор A_w, форма (B, H, T, BT).
        A_u_original: Оригинальный тензор A_u, форма (B, H, T, BT).
    """
    B, H, T, K = k.shape
    V = v.shape[-1]

    # Вычисление NT, BK, BV
    NT = cdiv(T, BT)  # Количество блоков по времени
    BK = min(next_power_of_2(K), 64)  # Размер блока для K
    BV = min(next_power_of_2(V), 64)  # Размер блока для V
    # print(BK, BV)

    # Инициализация выходных тензоров
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    A_w = torch.zeros(B, H, T, BT, device=k.device, dtype=torch.float32)
    A_w_original = torch.zeros(B, H, T, BT, device=k.device, dtype=torch.float32)
    A_u = torch.zeros(B, H, T, BT, device=k.device, dtype=torch.float32)
    A_u_original = torch.zeros(B, H, T, BT, device=k.device, dtype=torch.float32)

    # Вызов ядер для вычисления w и u
    fwd_prepare_wy_repr_kernel_w(k, beta, w, A_w, A_w_original, T, K, V, BT, BK, BV)
    fwd_prepare_wy_repr_kernel_u(k, v, beta, g, u, A_u, A_u_original, T, K, V, BT, BK, BV)

    return w, u, A_w, A_u, A_w_original, A_u_original

def fwd_prepare_wy_repr_kernel_w(k, beta, w, A, A_original, T, K, V, BT, BK, BV):
    """
    Ядро для вычисления промежуточных значений w и A_w.
    """
    B, H, _, _ = k.shape
    NT = cdiv(T, BT)  # Количество блоков по времени

    for i_bh in range(B * H):
        for i_t in range(NT):
            # Загрузка блока beta
            start_t = i_t * BT
            end_t = min(start_t + BT, T)
            b_beta = beta[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)

            # Инициализация b_A
            b_A = torch.zeros((BT, BT), dtype=torch.float32, device=k.device)

            # Обработка блоков K
            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                # print(f'b_k {b_k.shape}, b_beta[:, None] {b_beta[:, None].shape}, b_beta {b_beta.shape}')
                # b_kb = b_k * b_beta[:, None]  # (BT, BK) * (BT, 1) -> (BT, BK)
                if b_beta[:, None].shape[0] != b_k.shape[0]:
                    continue
                b_kb = b_k * b_beta[:, None]

                # Матричное умножение: (BT, BK) @ (BK, BT) -> (BT, BT)
                # print(f'b_A {b_A.shape}, b_kb {b_kb.shape}, b_k.transpose(-1, -2) {b_k.transpose(-1, -2).shape}')
                b_A += torch.matmul(b_kb, b_k.transpose(-1, -2))

            # Применение маски к b_A
            mask = torch.tril(torch.ones((BT, BT), device=k.device), diagonal=-1)
            b_A = -b_A * mask

            # Сохранение в A_original
            A_original[i_bh // H, i_bh % H, start_t:end_t, :BT] = b_A[:end_t - start_t]

            # Обновление b_A
            for i in range(1, BT):
                mask_i = torch.arange(BT) == i
                b_a = torch.sum(b_A * mask_i[:, None], dim=0)
                b_a += torch.sum(b_a[:, None] * b_A, dim=0) * (torch.arange(BT) < i)
                b_A[mask_i] = b_a[mask_i]

            b_A += torch.eye(BT, device=k.device)

            # Сохранение в A
            A[i_bh // H, i_bh % H, start_t:end_t, :BT] = b_A[:end_t - start_t]

            # Обновление w
            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                # b_kb = b_k * b_beta[:, None]  # (BT, BK)
                if b_beta[:, None].shape[0] != b_k.shape[0]:
                    continue
                b_kb = b_k * b_beta[:, None]
                b_w = torch.matmul(b_A[:end_t - start_t], b_kb)  # (BT, BT) @ (BT, BK) -> (BT, BK)
                w[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k] = b_w

def fwd_prepare_wy_repr_kernel_u(k, v, beta, g, u, A, A_original, T, K, V, BT, BK, BV):
    """
    Ядро для вычисления промежуточных значений u и A_u.
    """
    B, H, _, _ = k.shape
    NT = cdiv(T, BT)  # Количество блоков по времени

    for i_bh in range(B * H):
        for i_t in range(NT):
            # Загрузка блока beta и g
            start_t = i_t * BT
            end_t = min(start_t + BT, T)
            b_beta = beta[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)
            # print(b_beta.shape, beta.shape)
            b_g = g[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)

            # Инициализация b_A
            b_A = torch.zeros((BT, BT), dtype=torch.float32, device=k.device)

            # Обработка блоков K
            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                if b_beta[:, None].shape[0] != b_k.shape[0]:
                    continue
                b_kb = b_k * b_beta[:, None]  # (BT, BK)
                b_A += torch.matmul(b_kb, b_k.transpose(-1, -2))  # (BT, BT)

            # Применение экспоненты и маски к b_A
            # print(b_g[:, None].shape, b_g[None, :].shape, b_A.shape, BT)
            if b_g.shape != BT:
                continue
            b_A = b_A * torch.exp2(b_g[:, None] - b_g[None, :])  # (BT, BT)
            mask = torch.tril(torch.ones((BT, BT), device=k.device), diagonal=-1)
            b_A = -b_A * mask

            # Сохранение в A_original
            A_original[i_bh // H, i_bh % H, start_t:end_t, :BT] = b_A[:end_t - start_t]

            # Обновление b_A
            for i in range(1, BT):
                mask_i = torch.arange(BT) == i
                b_a = torch.sum(b_A * mask_i[:, None], dim=0)
                b_a += torch.sum(b_a[:, None] * b_A, dim=0) * (torch.arange(BT) < i)
                b_A[mask_i] = b_a[mask_i]

            b_A += torch.eye(BT, device=k.device)

            # Сохранение в A
            A[i_bh // H, i_bh % H, start_t:end_t, :BT] = b_A[:end_t - start_t]

            # Обновление u
            for i_v in range(cdiv(V, BV)):
                start_v = i_v * BV
                end_v = min(start_v + BV, V)
                b_v = v[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)
                # Приведение b_beta к размерности (BT, BV)
                # b_beta_expanded = b_beta.unsqueeze(-1).expand(-1, -1, BV // K)  # (BT, K, 1) -> (BT, K, BV // K)
                # b_beta_expanded = b_beta_expanded.reshape(BT, BV)  # (BT, K, BV // K) -> (BT, BV)
                # b_vb = b_v * b_beta_expanded  # (BT, BV) * (BT, BV) -> (BT, BV)
                if b_beta[:, None].shape[0] != b_k.shape[0]:
                    continue
                b_vb = b_v * b_beta[:, None]
                # print(b_v.shape, b_beta.shape, b_A[:end_t - start_t].shape, BV, BT, b_beta_expanded.shape)
                # b_vb = b_v * b_beta_expanded[:, :BV]  # (BT, BV) * (BT, BV) -> (BT, BV)

                # b_vb = b_v * b_beta  # (BT, BV)
                b_u = torch.matmul(b_A[:end_t - start_t], b_vb)  # (BT, BT) @ (BT, BV) -> (BT, BV)
                u[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v] = b_u

def fwd_recompute_w_u(k, v, beta, A_w, A_u, BT):
    """
    Реализация функции fwd_recompute_w_u_kernel для CPU.

    Аргументы:
        k: Тензор ключей, форма (B, H, T, K).
        v: Тензор значений, форма (B, H, T, V).
        beta: Тензор масштабирования, форма (B, H, T).
        A_w: Промежуточный тензор для ключей, форма (B, H, T, BT).
        A_u: Промежуточный тензор для значений, форма (B, H, T, BT).
        BT: Размер временного блока.

    Возвращает:
        w: Обновленный тензор ключей, форма (B, H, T, K).
        u: Обновленный тензор значений, форма (B, H, T, V).
    """
    B, H, T, K = k.shape
    V = v.shape[-1]
    NT = cdiv(T, BT)

    # Определение размеров блоков
    BK = min(next_power_of_2(K), 64)
    BV = min(next_power_of_2(V), 64)

    # Инициализация выходных тензоров
    w = torch.empty_like(k)
    u = torch.empty_like(v)

    # Основной цикл обработки блоков
    for i_bh in range(B * H):
        for i_t in range(NT):
            start_t = i_t * BT
            end_t = min(start_t + BT, T)

            # Загрузка блока beta
            b_beta = beta[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)

            # Загрузка блока A_u
            b_Au = A_u[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)

            # Обработка блоков V
            for i_v in range(cdiv(V, BV)):
                start_v = i_v * BV
                end_v = min(start_v + BV, V)

                # Загрузка блока v
                b_v = v[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)
                if b_beta[:, None].shape[0] != b_v.shape[0]:
                    continue
                b_vb = b_v * b_beta[:, None]  # (BT, BV)

                # Вычисление b_u
                b_u = torch.matmul(b_Au[:end_t - start_t], b_vb)  # (BT, BT) @ (BT, BV) -> (BT, BV)

                # Сохранение в u
                u[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v] = b_u

            # Загрузка блока A_w
            b_Aw = A_w[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)

            # Обработка блоков K
            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)

                # Проверка на нулевой размер блока
                if end_k - start_k == 0:
                    continue  # Пропускаем итерацию, если блок пустой

                # Загрузка блока k
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                if b_beta[:, None].shape[0] != b_k.shape[0]:
                    continue
                b_kb = b_k * b_beta[:, None]  # (BT, BK)

                # Вычисление b_w
                b_w = torch.matmul(b_Aw[:end_t - start_t], b_kb)  # (BT, BT) @ (BT, BK) -> (BT, BK)

                # Сохранение в w
                w[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k] = b_w

    return w, u

def chunk_fwd_h_fn(k, w, u, g, BT, initial_state=None, final_state=None, state_in_fp32=False):
    """
    Реализация функции chunk_fwd_h_fn для CPU.

    Аргументы:
        k: Тензор ключей, форма (B, H, T, K).
        w: Тензор весов, форма (B, H, T, K).
        u: Тензор значений, форма (B, H, T, V).
        g: Тензор временных масштабов, форма (B, H, T).
        BT: Размер временного блока.
        initial_state: Начальное состояние, форма (B, H, K, V) (опционально).
        final_state: Финальное состояние, форма (B, H, K, V) (опционально).
        state_in_fp32: Флаг для использования float32 для состояния.

    Возвращает:
        h: Промежуточный тензор, форма (B, H, NT * K, V).
        v_new: Обновленный тензор значений, форма (B, H, T, V).
    """
    B, H, T, K = k.shape
    V = u.shape[-1]

    # Параметры блоков
    BK = next_power_of_2(K)
    assert BK <= 256, "Текущая реализация не поддерживает размерность головы больше 256."
    BV = 16 if BK > 128 else 32
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = cdiv(T, BT), cdiv(K, BK), cdiv(V, BV)

    # Инициализация выходных тензоров
    h = torch.zeros((B, H, NT * K, V), dtype=torch.float32 if state_in_fp32 else k.dtype, device=k.device)
    v_new = torch.empty_like(u)

    # Основной цикл обработки блоков
    for i_bh in range(B * H):
        b_h = torch.zeros((K, V), dtype=torch.float32 if state_in_fp32 else k.dtype, device=k.device)

        # Загрузка начального состояния
        if initial_state is not None:
            b_h = initial_state[i_bh // H, i_bh % H].to(b_h.dtype)

        for i_t in range(NT):
            start_t = i_t * BT
            end_t = min(start_t + BT, T)

            # Сохранение текущего состояния в h
            h[i_bh // H, i_bh % H, i_t * K:(i_t + 1) * K] = b_h

            # b_h_cumsum = torch.zeros((BK, BV), dtype=torch.float32, device=k.device)
            b_h_cumsum = torch.zeros((K, V), dtype=torch.float32, device=k.device)
            if end_t - 1 >= g.shape[2]:
                continue
            b_g_last = g[i_bh // H, i_bh % H, end_t - 1]

            # Обработка подблоков
            for i_c in range(cdiv(BT, BC)):
                start_c = i_c * BC
                end_c = min(start_c + BC, BT)

                # Загрузка блоков k, w, v
                b_k = k[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c]  # (BC, K)
                b_w = w[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c]  # (BC, K)
                b_v = u[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c]  # (BC, V)

                # Вычисление b_g
                b_g = g[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c]  # (BC,)
                b_w = b_w * torch.exp2(b_g[:, None])  # (BC, K)

                # Обновление b_v
                b_v -= torch.matmul(b_w, b_h.to(b_w.dtype))  # (BC, V) -= (BC, K) @ (K, V)
                v_new[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c] = b_v

                # Обновление b_h_cumsum
                b_k = b_k * torch.exp2(b_g_last - b_g)[:, None]  # (BC, K)
                # print('b_h_cumsum', b_h_cumsum.shape, 'torch.matmul(b_k.transpose(0, 1), b_v)', torch.matmul(b_k.transpose(0, 1), b_v).shape)
                # print('b_k.transpose(0, 1)', b_k.transpose(0, 1).shape, 'b_v', b_v.shape, 'b_k', b_k.shape)
                b_h_cumsum += torch.matmul(b_k.transpose(0, 1), b_v)  # (K, V) += (K, BC) @ (BC, V)

            # Обновление состояния
            b_h *= torch.exp2(b_g_last)
            b_h += b_h_cumsum

        # Сохранение финального состояния
        if final_state is not None:
            final_state[i_bh // H, i_bh % H] = b_h

    return h, v_new

def chunk_fwd_o_fn(q, k, v_new, g, h, BT): # Вопросики к 0 :)
    """
    Реализация функции chunk_linear_attn_fwd_kernel_o для CPU.

    Аргументы:
        q: Тензор запросов, форма (B, H, T, K).
        k: Тензор ключей, форма (B, H, T, K).
        v_new: Обновленный тензор значений, форма (B, H, T, V).
        g: Тензор временных масштабов, форма (B, H, T).
        h: Промежуточный тензор, форма (B, H, NT * K, V).
        BT: Размер временного блока.

    Возвращает:
        o: Выходной тензор, форма (B, H, T, V).
    """
    B, H, T, K = q.shape
    V = v_new.shape[-1]

    # Параметры блоков
    BK = min(next_power_of_2(K), 64)
    BV = min(next_power_of_2(V), 64)
    NT = cdiv(T, BT)

    # Инициализация выходного тензора
    o = torch.zeros_like(v_new)

    # Масштабирующий коэффициент
    scale = K ** -0.5

    # Основной цикл обработки блоков
    for i_bh in range(B * H):
        for i_t in range(NT):
            start_t = i_t * BT
            end_t = min(start_t + BT, T)

            # Инициализация промежуточных тензоров
            b_o = torch.zeros((BT, BV), dtype=torch.float32)  # (BT, BV)
            b_s = torch.zeros((BT, BT), dtype=torch.float32)  # (BT, BT)

            # Обработка блоков K
            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)

                # Проверка на нулевой размер блока
                if end_k - start_k == 0:
                    continue  # Пропускаем итерацию, если блок пустой

                # Загрузка блоков q, k, h
                b_q = q[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                b_q = b_q * scale  # Масштабирование
                b_k = k[i_bh // H, i_bh % H, start_k:end_k, start_t:end_t]  # (BK, BT)
                b_h = h[i_bh // H, i_bh % H, start_k:end_k, :BV]  # (BK, BV)

                # Вычисление b_o и b_s
                b_o += torch.matmul(b_q, b_h)  # (BT, BK) @ (BK, BV) -> (BT, BV)
                # print('b_k', b_k.shape, 'b_q', b_q.shape, 'b_s', b_s.shape, torch.matmul(b_q, b_k).shape)
                if b_k.shape[1] == b_q.shape[0]:
                    b_s += torch.matmul(b_q, b_k)  # (BT, BK) @ (BK, BT) -> (BT, BT)

            # Загрузка блока g
            b_g = g[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)
            # print(b_o.shape, b_g.shape)
            if b_g.shape != BT:
                continue
            b_o = b_o * torch.exp2(b_g[:, None])  # (BT, BV) * (BT, 1) -> (BT, BV)
            b_s = b_s * torch.exp2(b_g[:, None] - b_g[None, :])  # (BT, BT) * (BT, BT) -> (BT, BT)

            # Применение маски к b_s
            mask = torch.arange(BT).unsqueeze(1) >= torch.arange(BT).unsqueeze(0)  # (BT, BT)
            b_s = torch.where(mask, b_s, 0)  # Применение маски

            # Загрузка блока v_new
            b_v = v_new[i_bh // H, i_bh % H, start_t:end_t, :BV]  # (BT, BV)

            # Обновление b_o
            b_o = b_o + torch.matmul(b_s, b_v)  # (BT, BT) @ (BT, BV) -> (BT, BV)

            # Сохранение в o
            o[i_bh // H, i_bh % H, start_t:end_t, :BV] = b_o[:end_t - start_t]

    return o

# Нули
def fwd_prepare_du(q, k, g, do, BT):
    """
    Реализация функции fwd_prepare_du_kernel для CPU.

    Аргументы:
        q: Тензор запросов, форма (B, H, T, K).
        k: Тензор ключей, форма (B, H, T, K).
        g: Тензор временных масштабов, форма (B, H, T).
        do: Градиент выхода, форма (B, H, T, V).
        BT: Размер временного блока.

    Возвращает:
        dv: Градиент значений, форма (B, H, T, V).
    """
    B, H, T, K = k.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)

    # Определение размеров блоков
    BK = min(next_power_of_2(K), 64)
    BV = min(next_power_of_2(V), 64)

    # Инициализация выходного тензора
    dv = torch.zeros_like(do)

    # Масштабирующий коэффициент
    scale = K ** -0.5

    # Основной цикл обработки блоков
    for i_bh in range(B * H):
        for i_t in range(NT):
            start_t = i_t * BT
            end_t = min(start_t + BT, T)

            # Инициализация b_A
            b_A = torch.zeros((BT, BT), dtype=torch.float32)

            # Обработка блоков K
            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)

                # Загрузка блоков q и k
                b_q = q[i_bh // H, i_bh % H, start_k:end_k, start_t:end_t]  # (BK, BT)
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)

                # Масштабирование
                b_q = b_q * scale

                # Вычисление b_A
                # print(b_k.shape, b_q.shape)
                if b_k.shape[1] == b_q.shape[0] and b_k.shape[0] == b_q.shape[1] :
                    b_A += torch.matmul(b_k, b_q)  # (BT, BK) @ (BK, BT) -> (BT, BT)

            # Загрузка блока g
            b_g = g[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)
            if b_g.shape != BT:
                continue
            b_A = b_A * torch.exp2(b_g[None, :] - b_g[:, None])  # (BT, BT)

            # Применение маски к b_A
            mask = torch.arange(BT).unsqueeze(1) <= torch.arange(BT).unsqueeze(0)  # (BT, BT)
            b_A = torch.where(mask, b_A, 0)  # Применение маски

            # Обработка блоков V
            for i_v in range(cdiv(V, BV)):
                start_v = i_v * BV
                end_v = min(start_v + BV, V)

                # Загрузка блока do
                b_do = do[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)

                # Вычисление b_dv
                b_dv = torch.matmul(b_A[:end_t - start_t], b_do)  # (BT, BT) @ (BT, BV) -> (BT, BV)

                # Сохранение в dv
                dv[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v] = b_dv

    return dv

# НУЛИ
def chunk_bwd_dhu_fn(q, k, w, g, do, dv, BT):
    """
    Реализация функции chunk_bwd_dhu_fn для CPU.

    Аргументы:
        q: Тензор запросов, форма (B, H, T, K).
        k: Тензор ключей, форма (B, H, T, K).
        w: Тензор весов, форма (B, H, T, K).
        g: Тензор временных масштабов, форма (B, H, T).
        do: Градиент выхода, форма (B, H, T, V).
        dv: Градиент значений, форма (B, H, T, V).
        BT: Размер временного блока.

    Возвращает:
        dh: Градиент состояния, форма (B, H, NT * K, V).
        dv2: Обновленный градиент значений, форма (B, H, T, V).
    """
    B, H, T, K = q.shape
    V = do.shape[-1]

    # Определение размеров блоков
    BK = min(next_power_of_2(K), 64)
    BV = min(next_power_of_2(V), 64)
    BC = min(BT, BK)
    NT = cdiv(T, BT)

    # Масштабирующий коэффициент
    scale = K ** -0.5

    # Инициализация выходных тензоров
    dh = torch.zeros((B, H, NT * K, V), dtype=torch.float32)
    dv2 = torch.empty_like(dv)

    # Основной цикл обработки блоков
    for i_bh in range(B * H):
        for i_k in range(cdiv(K, BK)):
            start_k = i_k * BK
            end_k = min(start_k + BK, K)

            for i_v in range(cdiv(V, BV)):
                start_v = i_v * BV
                end_v = min(start_v + BV, V)

                b_dh = torch.zeros((BK, BV), dtype=torch.float32)  # [BK, BV]

                for i_t in range(NT - 1, -1, -1):  # Обратный проход по временным блокам
                    start_t = i_t * BT
                    end_t = min(start_t + BT, T)

                    # Сохранение текущего состояния в dh
                    dh[i_bh // H, i_bh % H, i_t * K + start_k:i_t * K + end_k, start_v:end_v] = b_dh[:end_k - start_k]

                    b_dh_tmp = torch.zeros((BK, BV), dtype=torch.float32)  # [BK, BV]
                    if end_t - 1 >= g.shape[2]:
                        continue
                    bg_last = g[i_bh // H, i_bh % H, end_t - 1]  # Последнее значение g в блоке

                    # Обратный проход по подблокам
                    for i_c in range(cdiv(BT, BC) - 1, -1, -1):
                        start_c = i_c * BC
                        end_c = min(start_c + BC, BT)

                        # Проверка на нулевой размер блока
                        if start_t + start_c >= end_t or start_t + end_c <= start_t:
                            continue  # Пропускаем итерацию, если блок пустой

                        # Загрузка блоков q, k, w, do, dv
                        b_q = q[i_bh // H, i_bh % H, start_k:end_k, start_t + start_c:start_t + end_c]  # (BK, BC)
                        b_k = k[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c, start_k:end_k]  # (BC, BK)
                        b_w = w[i_bh // H, i_bh % H, start_k:end_k, start_t + start_c:start_t + end_c]  # (BK, BC)
                        b_do = do[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c, start_v:end_v]  # (BC, BV)
                        b_dv = dv[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c, start_v:end_v]  # (BC, BV)

                        # Загрузка блока g
                        b_g = g[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c]  # (BC,)

                        # Вычисление b_q и b_w
                        # print(b_q.shape, b_w.shape, torch.exp2(b_g)[None, :].shape)
                        if b_q.shape != b_k.shape:
                            continue

                        b_q = b_q * scale * torch.exp2(b_g)[None, :].t()  # (BK, BC)
                        if b_w.shape[1] != 0:
                            b_w = b_w * torch.exp2(b_g)[None, :].t()  # (BK, BC)

                        # Вычисление b_dh_tmp
                        # print(b_dh_tmp.shape, b_k.shape)
                        b_dh_tmp += torch.matmul(b_q, b_do.to(b_q.dtype))  # (BK, BC) @ (BC, BV) -> (BK, BV)

                        # Вычисление b_k
                        b_k = b_k * torch.exp2(bg_last - b_g)[:, None]  # (BC, BK)

                        # Обновление b_dv
                        b_dv += torch.matmul(b_k, b_dh.to(b_k.dtype))  # (BC, BK) @ (BK, BV) -> (BC, BV)

                        # Сохранение в dv2
                        dv2[i_bh // H, i_bh % H, start_t + start_c:start_t + end_c, start_v:end_v] = b_dv

                        # Обновление b_dh_tmp
                        b_dh_tmp -= torch.matmul(b_w, b_dv.to(b_q.dtype))  # (BK, BC) @ (BC, BV) -> (BK, BV)

                # Обновление b_dh
                b_dh *= torch.exp2(bg_last)
                b_dh += b_dh_tmp

    return dh, dv2

# нули
def chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, du, do, dh, BT):
    """
    Реализация функции chunk_bwd_dqkw_fn для CPU.

    Аргументы:
        q: Тензор запросов, форма (B, H, T, K).
        k: Тензор ключей, форма (B, H, T, K).
        v_new: Обновленный тензор значений, форма (B, H, T, V).
        w: Тензор весов, форма (B, H, T, K).
        g: Тензор временных масштабов, форма (B, H, T).
        h: Промежуточный тензор, форма (B, H, NT * K, V).
        du: Градиент входных данных, форма (B, H, T, V).
        do: Градиент выхода, форма (B, H, T, V).
        dh: Градиент состояния, форма (B, H, NT * K, V).
        BT: Размер временного блока.

    Возвращает:
        dq: Градиент запросов, форма (B, H, T, K).
        dk: Градиент ключей, форма (B, H, T, K).
        dw: Градиент весов, форма (B, H, T, K).
        dg: Градиент временных масштабов, форма (B, H, T).
    """
    B, H, T, K = q.shape
    # print(T)
    V = v_new.shape[-1]

    # Определение размеров блоков
    BK = min(next_power_of_2(K), 64)
    BV = min(next_power_of_2(V), 64)
    NT = cdiv(T, BT)

    # Масштабирующий коэффициент
    scale = K ** -0.5

    # Инициализация выходных тензоров
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dw = torch.zeros_like(w)
    dg = torch.zeros((B, H, T), dtype=torch.float32)

    # Основной цикл обработки блоков
    for i_bh in range(B * H):
        for i_k in range(cdiv(K, BK)):
            start_k = i_k * BK
            end_k = min(start_k + BK, K)

            for i_t in range(NT):
                start_t = i_t * BT
                end_t = min(start_t + BT, T)

                b_dq = torch.zeros((BT, BK), dtype=torch.float32)
                b_dk = torch.zeros((BT, BK), dtype=torch.float32)
                b_dw = torch.zeros((BT, BK), dtype=torch.float32)
                b_ds = torch.zeros((BT, BT), dtype=torch.float32)
                b_dg_last = torch.zeros(BK, dtype=torch.float32)  # Исправлено: форма (BK,)
                b_dg = torch.zeros(BT, dtype=torch.float32)

                if end_t - 1 >= g.shape[2]:
                    continue
                bg_last = g[i_bh // H, i_bh % H, end_t - 1]

                for i_v in range(cdiv(V, BV)):
                    start_v = i_v * BV
                    end_v = min(start_v + BV, V)

                    # Загрузка блоков v, do, dh, dv
                    b_v = v_new[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)
                    b_do = do[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)
                    b_h = h[i_bh // H, i_bh % H, start_v:end_v, i_t * K + start_k:end_k].transpose(0, 1)  # (BV, BK)
                    b_dh = dh[i_bh // H, i_bh % H, start_v:end_v, i_t * K + start_k:end_k].transpose(0, 1)  # (BV, BK)
                    b_dv = du[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)

                    # Вычисление b_dg_last
                    # нули
                    # print(b_dg_last.shape, b_h.shape, b_dh.shape, torch.sum(b_h * b_dh, dim=1).shape)
                    if b_h.shape[0] == 0:
                        continue
                    b_dg_last += torch.sum(b_h * b_dh, dim=0)  # (BV, BK) -> (BK,)

                    # Вычисление b_ds
                    b_ds += torch.matmul(b_do, b_v.transpose(0, 1))  # (BT, BV) @ (BV, BT) -> (BT, BT)

                    # Вычисление b_dq, b_dk, b_dw
                    if b_h.t().shape[1] != BK:
                        continue
                    b_dq += torch.matmul(b_do, b_h.t())  # (BT, BV) @ (BV, BK) -> (BT, BK)
                    # print(b_v.shape, b_v.t().shape, b_dh.shape)
                    b_dk += torch.matmul(b_v, b_dh.t())  # (BV, BT) @ (BV, BK) -> (BT, BK)
                    b_dw += torch.matmul(b_dv, b_h.t())  # (BV, BT) @ (BV, BK) -> (BT, BK)

                # Загрузка блоков q, k, w, g
                b_q = q[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                b_w = w[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                b_g = g[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)

                # Вычисление b_dq, b_dk, b_dw
                b_g_exp_qw = torch.exp2(b_g)[:, None]
                b_dq *= b_g_exp_qw * scale
                if b_q.shape[1] != BK:
                    continue
                # print(b_dq.shape, b_q.shape)
                b_dg += torch.sum(b_dq * b_q, dim=1)
                b_dw *= b_g_exp_qw
                b_dg -= torch.sum(b_dw * b_w, dim=1)
                b_dk *= torch.exp2(bg_last - b_g)[:, None]
                b_dg -= torch.sum(b_dk * b_k, dim=1)
                b_dg_last += torch.sum(b_dk * b_k, dim=0)  # (BT, BK) -> (BK,)

                # Вычисление b_ds
                mask = torch.arange(BT).unsqueeze(1) >= torch.arange(BT).unsqueeze(0)  # (BT, BT)
                b_ds = torch.where(mask, b_ds * scale * torch.exp2(b_g[:, None] - b_g[None, :]), 0)
                b_dg_mask = torch.matmul(b_q, b_k.t()) * b_ds
                b_dg += torch.sum(b_dg_mask, dim=1)
                b_dg -= torch.sum(b_dg_mask, dim=0)

                # Обновление b_dq, b_dk
                b_dq += torch.matmul(b_ds, b_k)  # (BT, BT) @ (BT, BK) -> (BT, BK)
                b_dk += torch.matmul(b_q.t(), b_ds).t()  # (BK, BT) @ (BT, BT) -> (BT, BK)

                # Сохранение в dq, dk, dw, dg
                dq[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k] = b_dq[:end_t - start_t]
                dk[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k] = b_dk[:end_t - start_t]
                dw[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k] = -b_dw[:end_t - start_t]
                dg[i_bh // H, i_bh % H, start_t:end_t] += b_dg[:end_t - start_t]

    return dq, dk, dw, dg

def bwd_prepare_wy_repr(k, v, beta, g, A_w, A_u, A_w_original, A_u_original, dw, du, BT):
    B, H, T, K = k.shape
    V = v.shape[-1]

    # Вычисление NT, BK, BV
    NT = cdiv(T, BT)
    BK = min(next_power_of_2(K), 64)
    BV = min(next_power_of_2(V), 64)

    # Инициализация выходных тензоров
    dk = torch.empty_like(k).float()
    dv = torch.empty_like(v).float()
    dbeta = torch.empty_like(beta).float()
    dg = torch.empty_like(g).float()

    dA_w = torch.zeros_like(A_w).float()
    dA_w_original = torch.zeros_like(dA_w)
    dA_u = torch.zeros_like(A_u).float()
    dA_u_original = torch.zeros_like(dA_u)

    # Вызов ядер для вычисления dA, dk, dv, dbeta, dg
    bwd_prepare_wy_repr_kernel_dA(k, v, beta, dw, du, A_w, A_u, dA_w, dA_u, dk, dv, dbeta, T, K, V, BT, BK, BV)
    bwd_prepare_wy_repr_kernel_dA_recurrence(k, A_w, A_w_original, dA_w, dA_w_original, T, K, V, BT, BK, BV)
    bwd_prepare_wy_repr_kernel_dA_recurrence(k, A_u, A_u_original, dA_u, dA_u_original, T, K, V, BT, BK, BV)

    # Вычисление dk2, dbeta2, dg
    dk2 = torch.empty_like(k).float()
    dbeta2 = torch.empty_like(beta).float()
    bwd_prepare_wy_repr_dk_dbeta_dg(k, beta, g, dA_w_original, dA_u_original, A_w_original, dk2, dbeta2, dg, T, K, V, BT, BK, BV)

    # Суммирование градиентов
    dk += dk2
    dbeta += dbeta2

    return dk, dv, dbeta, dg

def bwd_prepare_wy_repr_kernel_dA(k, v, beta, dw, du, A_w, A_u, dA_w, dA_u, dk, dv, dbeta, T, K, V, BT, BK, BV):
    B, H, _, _ = k.shape
    NT = cdiv(T, BT)

    for i_bh in range(B * H):
        for i_t in range(NT):
            start_t = i_t * BT
            end_t = min(start_t + BT, T)

            # Загрузка блока beta
            b_beta = beta[i_bh // H, i_bh % H, start_t:end_t]  # (BT, K)

            # Инициализация dA
            b_dA_u = torch.zeros((BT, BT), dtype=torch.float32, device=k.device)
            b_dA_w = torch.zeros((BT, BT), dtype=torch.float32, device=k.device)

            # Обработка блоков V
            for i_v in range(cdiv(V, BV)):
                start_v = i_v * BV
                end_v = min(start_v + BV, V)
                b_v = v[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)
                b_du = du[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v]  # (BT, BV)

                # Если b_beta должен быть применен к значениям, изменим его размерность
                # b_beta_v = b_beta.unsqueeze(1).expand(-1, BV, -1)  # (BT, BV, K)
                # b_beta_v = b_beta_v.mean(dim=-1)  # (BT, BV)
                # b_v_beta = b_v * b_beta_v  # (BT, BV)
                if b_beta[:, None].shape[0] != b_v.shape[0]:
                    continue
                b_v_beta = b_v * b_beta[:, None]

                b_dA_u += torch.matmul(b_du, b_v_beta.transpose(-1, -2))  # (BT, BT)
                b_dv_beta = torch.matmul(A_u[i_bh // H, i_bh % H, start_t:end_t, :BT], b_du)  # (BT, BV)
                b_dv = b_dv_beta * b_beta[:, None]   # (BT, BV)
                dv[i_bh // H, i_bh % H, start_t:end_t, start_v:end_v] = b_dv

                # Обновление dbeta
                dbeta_update = torch.sum(b_dv_beta * b_v, dim=1)  # (BT,)
                # dbeta_update = dbeta_update.unsqueeze(1).expand(-1, K)  # (BT, K)
                dbeta[i_bh // H, i_bh % H, start_t:end_t] += dbeta_update   # (BT, K)

            # Обработка блоков K
            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                b_dw = dw[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)

                if b_beta[:, None].shape[0] != b_k.shape[0]:
                    continue
                b_k_beta = b_k * b_beta[:, None]  # (BT, BK)

                b_dA_w += torch.matmul(b_dw, b_k_beta.transpose(-1, -2))  # (BT, BT)
                b_dk_beta = torch.matmul(A_w[i_bh // H, i_bh % H, start_t:end_t, :BT], b_dw)  # (BT, BK)
                b_dk = b_dk_beta * b_beta[:, None]  # (BT, BK)
                dk[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k] = b_dk
                # dbeta[i_bh // H, i_bh % H, start_t:end_t] += torch.sum(b_dk_beta * b_k, dim=1).unsqueeze(1).expand(-1, K)  # (BT, K)
                dbeta[i_bh // H, i_bh % H, start_t:end_t] += torch.sum(b_dk_beta * b_k, dim=1)

            # Сохранение dA_u и dA_w
            dA_u[i_bh // H, i_bh % H, start_t:end_t, :BT] = b_dA_u
            dA_w[i_bh // H, i_bh % H, start_t:end_t, :BT] = b_dA_w

def bwd_prepare_wy_repr_kernel_dA_recurrence(k, A, A_original, dA, dA_original, T, K, V, BT, BK, BV):
    B, H, _, _ = A.shape
    NT = cdiv(T, BT)

    for i_bh in range(B * H):
        for i_t in range(NT):
            start_t = i_t * BT
            end_t = min(start_t + BT, T)

            b_A = A[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)
            b_dA = dA[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)
            b_A_original = A_original[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)

            # Применение маски
            # mask = torch.tril(torch.ones((BT, BT), diagonal=-1, device=k.device)
            mask = torch.tril(torch.ones((BT, BT), device=k.device), diagonal=-1)
            b_dA = -b_dA * mask
            b_A -= torch.eye(BT, device=k.device)

            # Обратное обновление dA
            for i in range(BT - 1, 0, -1):
                mask_i = torch.arange(BT) == i
                b_da = torch.sum(b_dA * mask_i[:, None], dim=0)
                b_a = torch.sum(b_A_original * mask_i[:, None], dim=0)
                b_da2 = b_da + torch.sum(b_da[None, :] * b_A, dim=1)
                b_dA[mask_i] = b_da2
                b_dA += b_da[None, :] * b_a[:, None]

            # Сохранение dA_original
            dA_original[i_bh // H, i_bh % H, start_t:end_t, :BT] = b_dA

def bwd_prepare_wy_repr_dk_dbeta_dg(k, beta, g, dA_w, dA_u, A_w, dk, dbeta, dg, T, K, V, BT, BK, BV):
    B, H, _, _ = k.shape
    NT = cdiv(T, BT)

    for i_bh in range(B * H):
        for i_t in range(NT):
            start_t = i_t * BT
            end_t = min(start_t + BT, T)

            # Загрузка блока beta и g
            b_beta = beta[i_bh // H, i_bh % H, start_t:end_t]  # (BT, K)
            b_g = g[i_bh // H, i_bh % H, start_t:end_t]  # (BT,)

            # Загрузка dA_w, dA_u и A_w
            b_dA_w = dA_w[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)
            b_dA_u = dA_u[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)
            b_A_w = A_w[i_bh // H, i_bh % H, start_t:end_t, :BT]  # (BT, BT)

            # Вычисление dg
            if b_g.shape != BT:
                continue
            b_dA = (b_dA_w + b_dA_u * torch.exp2(b_g[:, None] - b_g[None, :]))  # (BT, BT)
            mask = torch.tril(torch.ones((BT, BT), device=k.device), diagonal=-1)
            b_dA = -b_dA * mask

            b_dg_exp = b_dA_u * b_A_w
            b_dg_exp = b_dg_exp * mask
            b_dg = torch.sum(b_dg_exp, dim=1) - torch.sum(b_dg_exp, dim=0)
            dg[i_bh // H, i_bh % H, start_t:end_t] = b_dg

            # Вычисление dk и dbeta
            b_dbeta = torch.zeros(BT, dtype=torch.float32, device=k.device)

            for i_k in range(cdiv(K, BK)):
                start_k = i_k * BK
                end_k = min(start_k + BK, K)
                b_k = k[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k]  # (BT, BK)
                b_dk_beta = torch.matmul(b_dA, b_k)  # (BT, BK)
                b_dbeta += torch.sum(b_dk_beta * b_k, dim=1)  # (BT,)
                if b_beta[:, None].shape[0] != b_k.shape[0]:
                    continue
                b_dk = torch.matmul(b_dA.transpose(-1, -2), b_k * b_beta[:, None])  # (BT, BK)
                dk[i_bh // H, i_bh % H, start_t:end_t, start_k:end_k] = b_dk

            # Обновление dbeta
            # b_dbeta = b_dbeta.unsqueeze(1).expand(-1, K)  # (BT, K)
            # print(dbeta[i_bh // H, i_bh % H, start_t:end_t].shape, b_dbeta.shape)
            if dbeta[i_bh // H, i_bh % H, start_t:end_t].shape[0] != b_dbeta.shape[0]:
                continue
            dbeta[i_bh // H, i_bh % H, start_t:end_t] += b_dbeta


