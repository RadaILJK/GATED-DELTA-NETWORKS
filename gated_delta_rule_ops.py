import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from chunk import fwd_prepare_wy_repr, chunk_fwd_h_fn, chunk_fwd_o_fn, fwd_recompute_w_u, fwd_prepare_du, chunk_bwd_dhu_fn, chunk_bwd_dqkw_fn, bwd_prepare_wy_repr

class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    # На CPU mixed precision обычно не используется, поэтому эти декораторы можно просто убрать.
    # @custom_fwd
    # @contiguous
    def forward(ctx, q, k, v, beta, g, BT, initial_state, output_final_state):
        g = g.float()
        #currently we force the length to be multiple of BT

        g = F.pad(g, (0, 3))
        print('g.shape[-1]', g.shape[-1], g.shape)
        assert g.shape[-1] % BT == 0
        g = rearrange(g, 'b h (n c) -> b h n c', c=BT)
        # change the base of log from e to 2, i.e., ln->log2. To use tl.math.exp2 inside the kernel.
        g = g.cumsum(-1) * 1.44269504
        g = rearrange(g, 'b h n c -> b h (n c)')

        ### obtain WY representation. u is actually the new v.
        w, u, A_w, A_u, A_w_original, A_u_original = fwd_prepare_wy_repr(k, v, beta, g, BT)
        ### forward_h
        final_state = None
        # state will convert to bf16 to do matmul anyway so we don't need fp32 state in the forward pass.
        h, v_new = chunk_fwd_h_fn(k, w, u, g, BT, initial_state, final_state, state_in_fp32=False)
        ## obtain output
        o = chunk_fwd_o_fn(q, k, v_new, g, h, BT)
        # save memory
        # if checkpoint_level == 1:
        # always save memory
        h, v_new = None, None
        ctx.save_for_backward(q, k, v, beta, g, A_w, A_u, A_w_original, A_u_original, h, v_new, initial_state)
        ctx.BT = BT
        return o.to(q.dtype), final_state

    @staticmethod
    # На CPU mixed precision обычно не используется, поэтому эти декораторы можно просто убрать.
    # @custom_bwd
    # @contiguous
    def backward(ctx, do, d_ht=None):
        q, k, v, beta, g, A_w, A_u, A_w_original, A_u_original, h, v_new, initial_state = ctx.saved_tensors
        BT = ctx.BT
        w, u = fwd_recompute_w_u(k, v, beta, A_w, A_u, BT)
        # checkpont_level=1, recomputation.
        # we need fp32 state to compute gradient.
        if h is None:
            h, v_new = chunk_fwd_h_fn(k, w, u, g, BT, initial_state, None, state_in_fp32=True)
        du = fwd_prepare_du(q, k, g, do, BT)
        dh, du = chunk_bwd_dhu_fn(q, k, w, g, do, du, BT)
        dq, dk, dw, dg = chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, du, do, dh, BT)
        dk2, dv, dbeta, dg2 = bwd_prepare_wy_repr(k, v, beta, g, A_w, A_u, A_w_original, A_u_original, dw, du, BT)
        dk.add_(dk2)
        dg.add_(dg2)

        dg = rearrange(dg, 'b h (n c) -> b h n c', c=BT)
        # mask = (torch.arange(0, BT)[:, None] >= torch.arange(0, BT)[None, :]).to(dg)
        assert dg.dtype == torch.float32, "dg should be fp32"
        # print(dg.abs().max())
        # dg = dg @ mask
        # dg = dg * 1.44269504
        def rev_cumsum(x):
            cumsum_x = x.cumsum(-1)
            rev_cumsum_x = cumsum_x[..., -1, None] - cumsum_x
            return rev_cumsum_x + x

        dg = rev_cumsum(dg)
        dg = rearrange(dg, 'b h n c -> b h (n c)')
        # print(dg.abs().max(), dq.abs().max(), dk.abs().max(), dv.abs().max(), dbeta.abs().max())
        # if dg.isnan().any():
        # breakpoint()

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype), dg.to(g.dtype), None, None, None, None

def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64, #chunk size
    initial_state: torch.Tensor = None,
    output_final_state: bool = False):
    assert q.dtype == k.dtype == v.dtype
    L = q.shape[-2]
    if L % BT != 0:
        q, k, v, beta, g = map(lambda x: F.pad(x, (0, 0, 0, BT - L % BT)), [q, k, v, beta.unsqueeze(-1), g.unsqueeze(-1)])
    g = g.squeeze(-1)
    beta = beta.squeeze(-1)

    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkGatedDeltaRuleFunction.apply(q, k, v, beta, g, BT,  initial_state, output_final_state)
    return o[:, :, :L, :], final_state