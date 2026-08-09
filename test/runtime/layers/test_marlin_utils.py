"""Tests for the Marlin weight-layout helpers used by the wNa16 path.

These helpers were ported 1:1 from SGLang (which took them from vLLM). A
permutation that is internally consistent but differs from the reference layout
silently produces wrong numerics once the Marlin GEMM reads the weights, so the
port was accepted only after checking it bit-for-bit against vLLM's
``marlin_utils`` on the same inputs.

That cross-engine comparison needs vLLM importable, so it is not run here. What
this file pins is everything checkable without it: pack/unpack round-trips, the
two scale permutations spelled out independently of the implementation, the
permutation property (every element moved, none invented), the group-vs-channel
branch, and the shape-validation rules. See ``unieinfra/xref/70-PORTS.md`` for
the port and its acceptance criteria.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.quantization import marlin_utils
from tokenspeed.runtime.layers.quantization.utils import pack_cols, unpack_cols


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("size_k,size_n", [(16, 32), (32, 64), (128, 256)])
def test_pack_cols_round_trips_through_unpack(
    num_bits: int, size_k: int, size_n: int
) -> None:
    torch.manual_seed(0)
    q_w = torch.randint(0, 2**num_bits, (size_k, size_n), dtype=torch.int32)

    packed = pack_cols(q_w, num_bits, size_k, size_n)
    assert packed.shape == (size_k, size_n // (32 // num_bits))
    assert packed.dtype == torch.int32

    assert torch.equal(unpack_cols(packed, num_bits, size_k, size_n), q_w)


def test_pack_cols_rejects_wrong_shape() -> None:
    q_w = torch.zeros((8, 16), dtype=torch.int32)
    with pytest.raises(ValueError):
        pack_cols(q_w, 4, 16, 8)


def test_scale_perms_match_reference() -> None:
    """The two permutations Marlin's dequantize path assumes."""
    scale_perm, scale_perm_single = marlin_utils.get_scale_perms()

    assert scale_perm == [i + 8 * j for i in range(8) for j in range(8)]
    assert scale_perm_single == [
        2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)
    ]


@pytest.mark.parametrize(
    "size_k,size_n,group_size",
    [(1024, 128, 128), (512, 256, -1), (2048, 64, 64)],
)
def test_marlin_permute_scales_is_a_permutation(
    size_k: int, size_n: int, group_size: int
) -> None:
    """Permuting must move every element and invent none."""
    rows = 1 if group_size == -1 else size_k // group_size
    s = torch.arange(rows * size_n, dtype=torch.float32).reshape(rows, size_n)

    out = marlin_utils.marlin_permute_scales(s, size_k, size_n, group_size)

    assert out.shape == (rows, size_n)
    assert out.is_contiguous()
    assert torch.equal(torch.sort(out.flatten()).values, s.flatten())


def test_marlin_permute_scales_group_and_channel_paths_differ() -> None:
    """group_size < size_k picks the group permutation, otherwise the single one.

    Guards the branch: getting it backwards is silent and produces wrong output.
    """
    s = torch.arange(8 * 64, dtype=torch.float32).reshape(8, 64)

    grouped = marlin_utils.marlin_permute_scales(
        s, size_k=1024, size_n=64, group_size=128
    )
    channelwise = marlin_utils.marlin_permute_scales(
        s, size_k=64, size_n=64, group_size=-1
    )

    assert not torch.equal(grouped, channelwise)


@pytest.mark.parametrize("num_bits", [4, 8])
def test_marlin_zero_points_packs_to_int32(num_bits: int) -> None:
    torch.manual_seed(0)
    size_k, size_n = 8, 128
    zp = torch.randint(0, 2**num_bits, (size_k, size_n), dtype=torch.int32)

    out = marlin_utils.marlin_zero_points(zp, size_k, size_n, num_bits)

    assert out.dtype == torch.int32
    assert out.shape == (size_k, size_n // (32 // num_bits))


def test_marlin_zero_points_rejects_unsupported_width() -> None:
    zp = torch.zeros((8, 128), dtype=torch.int32)
    with pytest.raises(ValueError):
        marlin_utils.marlin_zero_points(zp, 8, 128, num_bits=2)


def test_marlin_sort_g_idx_returns_sorted_values_and_their_indices() -> None:
    torch.manual_seed(0)
    g_idx = torch.randint(0, 8, (1024,), dtype=torch.int32)

    sorted_g_idx, sort_indices = marlin_utils.marlin_sort_g_idx(g_idx)

    assert sort_indices.dtype == torch.int32
    assert torch.equal(sorted_g_idx, g_idx[sort_indices])
    assert bool((sorted_g_idx[1:] >= sorted_g_idx[:-1]).all())


@pytest.mark.parametrize(
    "act_order,is_row_parallel,expected",
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (False, False, True),
    ],
)
def test_marlin_is_k_full(
    act_order: bool, is_row_parallel: bool, expected: bool
) -> None:
    assert marlin_utils.marlin_is_k_full(act_order, is_row_parallel) is expected


@pytest.mark.parametrize(
    "act_order,group_size,is_row_parallel,expected",
    [
        (True, 128, True, True),
        (True, 128, False, True),
        (False, -1, True, True),  # channelwise on a row-parallel layer
        (False, -1, False, False),
        (False, 128, True, False),
    ],
)
def test_marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool, expected: bool
) -> None:
    assert (
        marlin_utils.marlin_repeat_scales_on_all_ranks(
            act_order, group_size, is_row_parallel
        )
        is expected
    )


def test_check_marlin_supports_shape_accepts_aligned_shapes() -> None:
    ok, err = marlin_utils.check_marlin_supports_shape(
        output_size_per_partition=512,
        input_size_per_partition=1024,
        input_size=1024,
        group_size=128,
    )
    assert ok and err is None


@pytest.mark.parametrize(
    "output_size_per_partition,input_size_per_partition,input_size,group_size",
    [
        (100, 1024, 1024, 128),  # N not divisible by min_thread_n
        (512, 100, 1024, 128),  # K not divisible by min_thread_k
        (512, 1024, 4096, 384),  # K per partition not divisible by group_size
    ],
)
def test_check_marlin_supports_shape_rejects_misaligned_shapes(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    ok, err = marlin_utils.check_marlin_supports_shape(
        output_size_per_partition, input_size_per_partition, input_size, group_size
    )
    assert not ok
    assert err


def test_wna16_scheme_creates_weights() -> None:
    """The regression this port fixes.

    ``compressed_tensors_wNa16`` was a copy of SGLang's file with the
    ``marlin_utils`` import block removed, so ``create_weights`` raised
    ``NameError`` on its first statement and the whole w4a16 path was dead.
    """
    from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
        CompressedTensorsWNA16,
    )

    scheme = CompressedTensorsWNA16(
        strategy="group", num_bits=4, group_size=128, symmetric=True
    )
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_size=512,
        input_size=1024,
        output_partition_sizes=[512],
        input_size_per_partition=1024,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )

    assert {name for name, _ in layer.named_parameters()} == {
        "weight_packed",
        "weight_scale",
        "weight_shape",
    }
