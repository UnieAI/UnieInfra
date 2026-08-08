from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
    SimpleMambaPool,
    _prepare_gdn_decode_state_path,
)
from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (
    fused_mamba_state_copy,
)


def _new_backend(page_size: int = 64) -> MambaAttnBackend:
    pool = SimpleMambaPool(
        size=8,
        num_mamba_layers=1,
        conv_state_shape=(4,),
        temporal_state_shape=(2, 2),
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        mamba_layer_ids=[0],
        device="cpu",
        page_size=page_size,
    )

    backend = object.__new__(MambaAttnBackend)
    backend.pool = pool
    backend.device = "cpu"
    backend.is_draft = False
    backend.spec_num_tokens = 1
    backend.speculative_num_draft_tokens = 0
    backend.flat_state_active = False
    return backend


@pytest.mark.parametrize(
    ("state_dtype", "expected_solution"),
    [
        (torch.float32, None),
        (torch.bfloat16, "triton"),
    ],
)
def test_gdn_decode_state_path_preserves_padding_indices(
    state_dtype: torch.dtype, expected_solution: str | None
):
    states = torch.empty(1, dtype=state_dtype)
    initial = torch.tensor([3, -1], dtype=torch.int32)
    output = torch.tensor([[4, 5], [-1, -1]], dtype=torch.int32)

    prepared_initial, prepared_output, solution = _prepare_gdn_decode_state_path(
        states, initial, output
    )

    assert prepared_initial is initial
    assert prepared_output is output
    assert solution == expected_solution


def test_bf16_decode_without_output_indices_falls_back_to_triton():
    states = torch.empty(1, dtype=torch.bfloat16)
    initial = torch.tensor([3, -1], dtype=torch.int32)

    prepared_initial, prepared_output, solution = _prepare_gdn_decode_state_path(
        states, initial, None
    )

    assert prepared_initial is initial
    assert prepared_output is None
    assert solution == "triton"


def test_simple_mamba_pool_preserves_k_last_temporal_shape():
    pool = SimpleMambaPool(
        size=8,
        num_mamba_layers=1,
        conv_state_shape=(4,),
        temporal_state_shape=(2, 3, 5),
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        mamba_layer_ids=[0],
        device="cpu",
    )

    assert pool.ssm_state.shape == (1, 8, 2, 3, 5)


def test_simple_mamba_pool_current_input_map_uses_rank_local_req_pool_range():
    pool = SimpleMambaPool(
        size=48,
        num_mamba_layers=1,
        conv_state_shape=(4,),
        temporal_state_shape=(2, 2),
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        mamba_layer_ids=[0],
        device="cpu",
        page_size=64,
        speculative_num_draft_tokens=4,
        max_req_pool_size=21,
    )

    assert pool.current_input_size == 22
    assert pool.current_input_indices.shape[0] == 22
    # MTP draft slots are addressed by rank-local req_pool_idx, plus one
    # graph-padding sink row after the scheduler-owned 1-based range.
    assert pool.total_size == 48 + 22 * 3


def test_extend_publishes_one_page_below_the_chunk_end():
    """A chunk ending on a page boundary publishes one page BELOW it, not at it.

    The chunk's own end is the one depth a later match can never reach: an
    exact repeat must leave a token to compute, and a diverging prefix splits
    the node inside that last page. Publishing at the end measured
    `#cached-token: 0` on Qwen3.5-27B for every shape tried.

    seq 320 / prefix 192 / page 64 -> publish at 320-64 = 256, i.e. 64 tokens
    into this chunk, which is the chunked scan's first checkpoint.
    """
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([320], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([256], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([192], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_src.tolist() == [0]
    assert metadata.track_ssm_h_src_fla.tolist() == [1]
    assert metadata.track_ssm_h_dst.tolist() == [5]
    # The chunk's final state is never the one filed now: the publish depth is
    # always strictly inside the extend, so it comes from the scan's
    # intermediate checkpoints.
    assert metadata.track_ssm_final_src is None
    assert metadata.track_ssm_final_dst is None


def test_extend_publishes_below_an_unaligned_chunk_end():
    """Same rule when the chunk ends mid-page: floor to a page, then back one.

    seq 319 / prefix 192 / page 64 -> floor(319/64)*64 = 256, minus one page
    = 192, which is the prefix itself, so there is nothing new to publish.
    """
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([319], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([256], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([192], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_dst is None
    assert metadata.track_ssm_final_dst is None


def test_extend_publishes_one_page_below_a_multi_page_chunk():
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([350], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([256], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([192], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    # seq 350 / prefix 192 -> floor(350/64)*64 = 320, minus one page = 256,
    # i.e. 64 tokens into this chunk.
    assert metadata.track_ssm_h_src.tolist() == [0]
    assert metadata.track_ssm_h_src_fla.tolist() == [1]
    assert metadata.track_ssm_h_dst.tolist() == [5]
    assert metadata.track_ssm_final_dst is None


def test_extend_publishes_nothing_when_the_chunk_spans_under_two_pages():
    """seq 350 / prefix 256 -> publish depth 256 == the prefix: nothing new.

    A chunk this short has no reachable boundary of its own to offer. Harmless
    at the shipping `chunked_prefill_size` (8192), where a prompt is one chunk
    and the publish depth is far above the prefix; it only bites for chunks
    narrower than two pages.
    """
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([350], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([-1], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([256], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_src is None
    assert metadata.track_ssm_h_dst is None


def test_extend_skips_unaligned_inserted_page_boundary():
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([66], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([-1], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([41], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_dst is None
    assert metadata.track_ssm_final_dst is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton copy kernel"
)
def test_mamba_state_copy_single_layer_rank3_interprets_first_dim_as_slot():
    pool = (
        torch.arange(6 * 2 * 3, device="cuda", dtype=torch.float32)
        .reshape(6, 2, 3)
        .clone()
    )
    original = pool.clone()
    src = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    dst = torch.tensor([1, 4], device="cuda", dtype=torch.int32)

    fused_mamba_state_copy(pool, src, dst, single_layer=True)
    torch.cuda.synchronize()

    assert torch.equal(pool[1], original[0])
    assert torch.equal(pool[4], original[3])
    assert torch.equal(pool[0], original[0])
    assert torch.equal(pool[3], original[3])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton copy kernel"
)
def test_mamba_state_copy_full_pool_rank4_keeps_default_layer_slot_layout():
    pool = (
        torch.arange(2 * 6 * 2 * 3, device="cuda", dtype=torch.float32)
        .reshape(2, 6, 2, 3)
        .clone()
    )
    original = pool.clone()
    src = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    dst = torch.tensor([1, 4], device="cuda", dtype=torch.int32)

    fused_mamba_state_copy(pool, src, dst)
    torch.cuda.synchronize()

    assert torch.equal(pool[:, 1], original[:, 0])
    assert torch.equal(pool[:, 4], original[:, 3])
    assert torch.equal(pool[:, 0], original[:, 0])
    assert torch.equal(pool[:, 3], original[:, 3])
