"""Block verification (Sun et al., 2024) of a DSpark draft chain.

Instead of testing each draft token against the target on its own, the whole
block is accepted jointly: with ``p_i = min(1, p_{i-1} * P(x_i) / Q(x_i))`` and
``h_i = R_i / (R_i + 1 - p_i)`` where ``R_i = sum_x max(p_i * P(x | x_<=i) -
Q(x | x_<=i), 0)``, the longest prefix whose own uniform ``u_i <= h_i`` is
accepted, and the next token is drawn from ``max(p * P - Q, 0)`` (``P`` after a
full block). It is lossless and accepts at least as many tokens as the
token-wise test. Reference: vLLM ``rejection_sampler_utils.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DSparkBlockVerifyState:
    """Draft-side tensors the verifier needs to rebuild every step's Q."""

    # [pool_slots, steps, local_vocab] base logits of the last drafted block.
    pool: torch.Tensor
    embedding: torch.Tensor  # [vocab, rank] replicated Markov bigram table
    projection: torch.Tensor  # [local_vocab, rank] this rank's shard
    num_valid: int  # real columns of this rank's shard
    tp_size: int
    tp_group: object


def draft_logits(
    state: DSparkBlockVerifyState, slots: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Full-vocab FP32 draft logits ``[bs, steps, tp * local_vocab]``.

    Step ``s`` conditions on ``candidates[:, s]`` (the anchor, then draft
    ``s - 1``): base logits plus the Markov bigram bias of that token.
    """
    from tokenspeed.runtime.distributed.comm_ops import all_gather_single

    base = state.pool.index_select(0, slots)  # [bs, steps, local]
    bs, steps, local = base.shape
    prev = candidates[:, :steps].long().clamp_(0, state.embedding.shape[0] - 1)
    bigram = state.embedding[prev].float()  # [bs, steps, rank]
    bias = bigram @ state.projection.float().T  # [bs, steps, local]
    mine = base + bias
    mine[..., state.num_valid :] = float("-inf")
    if state.tp_size == 1:
        return mine
    out = torch.empty(
        (state.tp_size, bs, steps, local), dtype=mine.dtype, device=mine.device
    )
    all_gather_single(out.view(-1), mine.contiguous().view(-1), state.tp_group)
    return out.permute(1, 2, 0, 3).reshape(bs, steps, state.tp_size * local)


def block_verify_chain(
    target_logits: torch.Tensor,
    draft_logits_full: torch.Tensor,
    candidates: torch.Tensor,
    temperature: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
) -> None:
    """Fill ``predict``/``accept_index``/``accept_length`` like
    ``verify_chain_target_sampled`` (``accept_length`` excludes the bonus).

    Args:
        target_logits: ``[bs * n, V]`` FP32 target logits, ``n = steps + 1``.
        draft_logits_full: ``[bs, steps, >= V]`` FP32 draft logits.
        candidates: ``[bs, n]`` int32 anchor + drafts.
        temperature: ``[bs]`` sampling temperature (> 0).
    """
    bs, n = candidates.shape
    steps = n - 1
    vocab = target_logits.shape[-1]
    t = temperature.view(bs, 1, 1).clamp_min(1.0e-6)
    lp = torch.log_softmax(target_logits.view(bs, n, vocab) / t, dim=-1)
    lq = torch.log_softmax(draft_logits_full[..., :vocab] / t, dim=-1)
    drafts = candidates[:, 1:].long()
    ratio = (
        lp[:, :steps].gather(-1, drafts[..., None])[..., 0]
        - lq.gather(-1, drafts[..., None])[..., 0]
    )
    cum = []
    log_p = torch.zeros(bs, device=ratio.device)
    for i in range(steps):
        log_p = torch.clamp(log_p + ratio[:, i], max=0.0)
        cum.append(log_p)
    cum = torch.stack(cum, dim=1)  # [bs, steps]
    p = cum.exp()
    # Residual mass at positions 0 .. steps-2 looks one row ahead.
    resid = torch.clamp(
        p[:, : steps - 1, None] * lp[:, 1:steps].exp() - lq[:, 1:].exp(), min=0.0
    ).sum(-1)
    denom = resid + 1.0 - p[:, : steps - 1]
    h = torch.where(denom > 0, resid / denom.clamp_min(1.0e-30), torch.ones_like(resid))
    h = torch.cat([h, p[:, steps - 1 :]], dim=1)  # last position: h = p
    u = torch.rand(bs, steps, device=h.device).clamp_min(1.0e-7)
    ok = u <= h
    idx = torch.arange(1, steps + 1, device=h.device)
    acc = torch.where(ok, idx, torch.zeros_like(idx)).amax(dim=1)  # [bs]
    # Next token: residual after acc accepted tokens, or P after a full block.
    rows = torch.arange(bs, device=h.device)
    pa = torch.where(
        acc > 0,
        p.gather(1, (acc - 1).clamp_min(0)[:, None])[:, 0],
        torch.ones_like(acc, dtype=p.dtype),
    )
    p_row = lp[rows, acc].exp()  # [bs, V]
    q_row = lq[rows, acc.clamp_max(steps - 1)].exp()
    res = torch.clamp(pa[:, None] * p_row - q_row, min=0.0)
    res = torch.where(
        (acc == steps)[:, None] | (res.sum(-1, keepdim=True) <= 0), p_row, res
    )
    g = -torch.log(-torch.log(torch.rand_like(res).clamp(1.0e-7, 1.0 - 1.0e-7)))
    final = (torch.log(res.clamp_min(1.0e-38)) + g).argmax(-1)  # [bs]
    # Pack like verify_chain_target_sampled.
    pos = torch.arange(n, device=h.device)[None, :]
    toks = torch.cat([drafts, drafts[:, :1]], dim=1)  # predict[j] = draft j+1
    out = torch.where(pos < acc[:, None], toks, torch.zeros_like(toks))
    out = torch.where(pos == acc[:, None], final[:, None], out)
    predict.view(bs, n).copy_(out.to(predict.dtype))
    accept_index.copy_(
        torch.where(
            pos <= acc[:, None],
            rows[:, None] * n + pos,
            torch.full_like(pos, -1) + rows[:, None] * 0,
        ).to(accept_index.dtype)
    )
    accept_length.copy_(acc.to(accept_length.dtype))
