/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * TokenSpeed modifications:
 *   - Ported the PyTorch custom op boundary to TokenSpeed's TVM FFI CUDA
 *     extension style.
 *   - Exposed only the DeepSeek V4 CSA top-k contracts needed by the model
 *     runtime.
 */

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dsv4_persistent_topk.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {
namespace prefill_topk {

template <int step>
static inline __device__ uint32_t extract_bin_idx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
  }
  return 0;
}

template <int shift>
static inline __device__ bool is_partial_match(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads,
                                   const T* in, IdxT len, Func f) {
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const IdxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (IdxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <int step, int kNumThreadsPerBlock, int kNumBins,
          int kNumFinalItems, typename SmemFinalType, typename SmemOutputType>
__device__ bool process_histogram_step(
    const float* logits, int row_len, uint32_t& logit_pattern,
    int& threshold_bin_idx, SmemOutputType& smem_output,
    int* smem_threshold_bin_idx, int* smem_final_dst_idx,
    int* smem_final_bin_size, int* smem_found_topk_values,
    SmemFinalType& smem_final, int top_k, float* prefix_logits,
    int* prefix_indices) {
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smem_final.histo.data[idx] = 0;
  }
  __syncthreads();

  constexpr auto pattern_shift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logit_pattern =
        static_cast<uint32_t>(threshold_bin_idx & 0x7ff) << pattern_shift;
  } else if constexpr (step == 3) {
    logit_pattern |=
        static_cast<uint32_t>(threshold_bin_idx & 0x7ff) << pattern_shift;
  }

  auto distribute_to_bins = [&](float logit, int /*idx*/ = 0) {
    if (is_partial_match<pattern_shift>(logit, logit_pattern)) {
      uint32_t bin_idx = extract_bin_idx<step>(logit);
      atomicAdd(&smem_final.histo.data[bin_idx], 1);
    }
  };

  vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits, row_len,
                     distribute_to_bins);
  __syncthreads();

  int last_value = smem_found_topk_values[0];
  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int bin_count = smem_final.histo.data[idx];
    __syncthreads();

    int prefix_sum = 0;
    int total_sum = 0;
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smem_final.histo.scan).ExclusiveSum(bin_count, prefix_sum, total_sum);

    prefix_sum += last_value;
    total_sum += last_value;
    smem_final.histo.data[idx] = prefix_sum;
    __syncthreads();

    bool found_threshold = false;
    if (prefix_sum < top_k) {
      int next_prefix_sum =
          threadIdx.x == kNumThreadsPerBlock - 1
              ? total_sum
              : smem_final.histo.data[idx + 1];
      if (next_prefix_sum >= top_k) {
        smem_threshold_bin_idx[0] = idx;
        smem_final_bin_size[0] = next_prefix_sum - prefix_sum;
        found_threshold = true;
      }
    }

    if (__syncthreads_or(found_threshold)) {
      break;
    }
    last_value = total_sum;
  }
  __syncthreads();

  threshold_bin_idx = smem_threshold_bin_idx[0];

  auto process_bins = [&](float logit, int idx) {
    if (is_partial_match<pattern_shift>(logit, logit_pattern)) {
      uint32_t bin_idx = extract_bin_idx<step>(logit);
      bool should_write_directly =
          (step == 0 && smem_final_bin_size[0] <= kNumFinalItems) ||
          (step >= 1);
      if (bin_idx < threshold_bin_idx && should_write_directly) {
        int dst_idx = atomicAdd(&smem_found_topk_values[0], 1);
        // The prefix (bins strictly above the threshold) is guaranteed to be
        // in the top-k set. atomicAdd picks the storage slot in nondeterministic
        // thread-arrival order, so record (score, index) in a side pool and let
        // the finalization pass re-sort it with the same composite key used for
        // the threshold bin. Writing smem_output here as well keeps the >top_k
        // near-identical fallback (which never runs finalization) functional.
        smem_output[dst_idx] = idx;
        prefix_logits[dst_idx] = logit;
        prefix_indices[dst_idx] = idx;
      }
      if constexpr (step < 3) {
        if (bin_idx == threshold_bin_idx &&
            smem_final_bin_size[0] <= kNumFinalItems) {
          int dst_idx = atomicAdd(&smem_final_dst_idx[0], 1);
          smem_final.items.logits[dst_idx] = logit;
          smem_final.items.indices[dst_idx] = idx;
        }
      } else {
        if (bin_idx == threshold_bin_idx) {
          if (smem_final_bin_size[0] <= kNumFinalItems) {
            // Deterministic path: pool the whole threshold bin into
            // smem_final.items so the final (rank-desc, index-asc) sort SELECTS
            // which items make the cut. Writing here via histogram atomics would
            // pick an arbitrary subset of near-tie items in nondeterministic
            // thread order.
            int dst_idx = atomicAdd(&smem_final_dst_idx[0], 1);
            smem_final.items.logits[dst_idx] = logit;
            smem_final.items.indices[dst_idx] = idx;
          } else {
            // More than kNumFinalItems near-identical values in one bin cannot
            // be sorted; fall back to filling up to top_k (the ties are exact
            // by construction, so the order does not change the softmax set).
            int dst_idx = atomicAdd(&smem_final.histo.data[bin_idx], 1);
            if (dst_idx < top_k) {
              smem_output[dst_idx] = idx;
            }
          }
        }
      }
    }
  };

  vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits, row_len,
                     process_bins);
  __syncthreads();
  return smem_final_bin_size[0] > kNumFinalItems;
}

template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort>
static __device__ void topk_per_row_job(const float* logits, int row_start,
                                        int row_end, int* out_indices,
                                        int top_k) {
  static constexpr int kNumFinalItems = 2048;
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  using FinalSort = cub::BlockRadixSort<uint64_t, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage =
      std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  struct FinalItems {
    int indices[kNumFinalItems];
    float logits[kNumFinalItems];
  };
  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smem_final;

  extern __shared__ int32_t smem_output[];
  __shared__ int smem_threshold_bin_idx[1];
  __shared__ int smem_final_dst_idx[1];
  __shared__ int smem_final_bin_size[1];
  __shared__ int smem_found_topk_values[1];

  // Side pool for the below-threshold prefix (the guaranteed top-k set). The
  // slot picked by atomicAdd is nondeterministic, so we keep (score, index) and
  // re-rank the whole pool deterministically during finalization.
  float* prefix_logits = reinterpret_cast<float*>(smem_output + top_k);
  int* prefix_indices = smem_output + 2 * top_k;

  int row_len = row_end - row_start;
  if (row_len <= top_k) {
    for (int row_it = threadIdx.x; row_it < row_len;
         row_it += kNumThreadsPerBlock) {
      out_indices[row_it] = row_it;
    }
    for (int row_it = row_len + threadIdx.x; row_it < top_k;
         row_it += kNumThreadsPerBlock) {
      out_indices[row_it] = -1;
    }
    return;
  }

  if (threadIdx.x == 0) {
    smem_final_dst_idx[0] = 0;
    smem_found_topk_values[0] = 0;
  }
  __syncthreads();

  int threshold_bin_idx = -1;
  uint32_t logit_pattern = 0;
  const float* row_logits = logits + row_start;

  bool continue_to_next_step =
      process_histogram_step<0, kNumThreadsPerBlock, kNumBins,
                             kNumFinalItems>(
          row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
          smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
          smem_found_topk_values, smem_final, top_k, prefix_logits,
          prefix_indices);
  if (continue_to_next_step) {
    continue_to_next_step =
        process_histogram_step<1, kNumThreadsPerBlock, kNumBins,
                               kNumFinalItems>(
            row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
            smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
            smem_found_topk_values, smem_final, top_k, prefix_logits,
            prefix_indices);
  }
  if (continue_to_next_step) {
    continue_to_next_step =
        process_histogram_step<2, kNumThreadsPerBlock, kNumBins,
                               kNumFinalItems>(
            row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
            smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
            smem_found_topk_values, smem_final, top_k, prefix_logits,
            prefix_indices);
  }
  if (continue_to_next_step) {
    process_histogram_step<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
        row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
        smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
        smem_found_topk_values, smem_final, top_k, prefix_logits,
        prefix_indices);
  }

  if (!continue_to_next_step) {
    // Deterministically order the prefix pool (score desc, index asc). The
    // prefix is at most top_k long and each thread re-ranks every candidate
    // over the whole pool, so the result is invariant to the atomicAdd storage
    // order used while filling the pool.
    int prefix_count = smem_found_topk_values[0];
    for (int i = threadIdx.x; i < prefix_count; i += kNumThreadsPerBlock) {
      int out_index = 0;
      float logit_i = prefix_logits[i];
      int idx_i = prefix_indices[i];
      for (int j = 0; j < prefix_count; j++) {
        float logit_j = prefix_logits[j];
        int idx_j = prefix_indices[j];
        if (logit_j > logit_i || (logit_j == logit_i && idx_j < idx_i)) {
          out_index++;
        }
      }
      smem_output[out_index] = idx_i;
    }
    __syncthreads();
    if constexpr (useRadixSort) {
      uint64_t final_keys[kNumFinalItemsPerThread];
      int final_indices[kNumFinalItemsPerThread];
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int src_idx = ii * kNumThreadsPerBlock + threadIdx.x;
        if (src_idx < smem_final_dst_idx[0]) {
          float logit = smem_final.items.logits[src_idx];
          int idx = smem_final.items.indices[src_idx];
          // Monotone float->uint32 transform so larger key == larger score.
          uint32_t score_bits =
              (logit >= 0.0f) ? (__float_as_uint(logit) ^ 0x80000000u)
                              : ~__float_as_uint(logit);
          // Composite (score desc, index asc) key; the index term makes the key
          // unique so the sort is deterministic even on exact-score ties.
          final_keys[ii] =
              (static_cast<uint64_t>(score_bits) << 32) |
              (0xFFFFFFFFu - static_cast<uint32_t>(idx));
          final_indices[ii] = idx;
        } else {
          final_keys[ii] = 0;
          final_indices[ii] = -1;
        }
      }
      __syncthreads();
      FinalSort(smem_final.finalSort)
          .SortDescendingBlockedToStriped(final_keys, final_indices);
      int base_idx = smem_found_topk_values[0];
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int src_idx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dst_idx = base_idx + src_idx;
        if (dst_idx < top_k) {
          smem_output[dst_idx] = final_indices[ii];
        }
      }
    } else {
      auto base_idx = smem_found_topk_values[0];
      for (int i = threadIdx.x; i < smem_final_dst_idx[0];
           i += kNumThreadsPerBlock) {
        int out_index = 0;
        auto logit = smem_final.items.logits[i];
        auto idx_i = smem_final.items.indices[i];
        for (int j = 0; j < smem_final_dst_idx[0]; j++) {
          auto other_logit = smem_final.items.logits[j];
          auto idx_j = smem_final.items.indices[j];
          if (logit < other_logit || (logit == other_logit && idx_i < idx_j)) {
            out_index++;
          }
        }
        if (out_index + base_idx < top_k) {
          smem_output[out_index + base_idx] = smem_final.items.indices[i];
        }
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < top_k; i += kNumThreadsPerBlock) {
    out_indices[i] = smem_output[i];
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock)
void topk_per_row_prefill_kernel(const float* logits, const int* row_starts,
                                 const int* row_ends, int* out_indices,
                                 int stride0, const int top_k,
                                 const int offset_index) {
  static constexpr int kNumBins = 2048;
  int row_idx = blockIdx.x + offset_index;
  int row_start = row_starts[row_idx];
  int row_end = row_ends[row_idx];
  out_indices += static_cast<int64_t>(row_idx) * top_k;
  logits += static_cast<int64_t>(row_idx) * stride0;
  topk_per_row_job<kNumThreadsPerBlock, kNumBins, useRadixSort>(
      logits, row_start, row_end, out_indices, top_k);
}

void launch_indexer_topk_prefill(const TensorView& logits,
                                 const TensorView& row_starts,
                                 const TensorView& row_ends,
                                 TensorView& output, int64_t k,
                                 cudaStream_t stream) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  int num_rows = static_cast<int>(logits.size(0));
  int num_insertion_blocks = std::min(num_rows, kSortingAlgorithmThreshold);
  if (num_insertion_blocks > 0) {
    topk_per_row_prefill_kernel<kNumThreadsPerBlock, false>
        <<<num_insertion_blocks, kNumThreadsPerBlock,
           static_cast<size_t>(k) * 3 * sizeof(int32_t), stream>>>(
            static_cast<float*>(logits.data_ptr()),
            static_cast<int*>(row_starts.data_ptr()),
            static_cast<int*>(row_ends.data_ptr()),
            static_cast<int*>(output.data_ptr()), static_cast<int>(logits.stride(0)),
            static_cast<int>(k), 0);
  }
  if (num_rows > kSortingAlgorithmThreshold) {
    int num_radix_blocks = num_rows - kSortingAlgorithmThreshold;
    topk_per_row_prefill_kernel<kNumThreadsPerBlock, true>
        <<<num_radix_blocks, kNumThreadsPerBlock,
           static_cast<size_t>(k) * 3 * sizeof(int32_t), stream>>>(
            static_cast<float*>(logits.data_ptr()),
            static_cast<int*>(row_starts.data_ptr()),
            static_cast<int*>(row_ends.data_ptr()),
            static_cast<int*>(output.data_ptr()), static_cast<int>(logits.stride(0)),
            static_cast<int>(k), kSortingAlgorithmThreshold);
  }
}

}  // namespace prefill_topk
}  // namespace

void dsv4_indexer_topk_prefill(TensorView logits, TensorView row_starts,
                                      TensorView row_ends, TensorView output,
                                      int64_t k) {
  CHECK_CUDA(logits);
  CHECK_CUDA(row_starts);
  CHECK_CUDA(row_ends);
  CHECK_CUDA(output);
  CHECK_DIM(2, logits);
  CHECK_DIM(1, row_starts);
  CHECK_DIM(1, row_ends);
  CHECK_DIM(2, output);
  CHECK_CONTIGUOUS(logits);
  CHECK_CONTIGUOUS(row_starts);
  CHECK_CONTIGUOUS(row_ends);
  CHECK_CONTIGUOUS(output);
  TVM_FFI_ICHECK(logits.dtype() == dl_float32) << "logits must be float32";
  TVM_FFI_ICHECK(row_starts.dtype() == dl_int32)
      << "row_starts must be int32";
  TVM_FFI_ICHECK(row_ends.dtype() == dl_int32) << "row_ends must be int32";
  TVM_FFI_ICHECK(output.dtype() == dl_int32) << "output must be int32";
  TVM_FFI_ICHECK(logits.stride(1) == 1)
      << "logits last dimension must be contiguous";
  TVM_FFI_ICHECK(row_starts.size(0) == logits.size(0))
      << "row_starts size mismatch";
  TVM_FFI_ICHECK(row_ends.size(0) == logits.size(0))
      << "row_ends size mismatch";
  TVM_FFI_ICHECK(output.size(0) == logits.size(0) && output.size(1) == k)
      << "output size mismatch";
  TVM_FFI_ICHECK(k > 0) << "k must be positive";

  cudaSetDevice(logits.device().device_id);
  cudaStream_t stream = get_stream(logits.device());
  prefill_topk::launch_indexer_topk_prefill(logits, row_starts, row_ends, output,
                                            k, stream);
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "dsv4_indexer_topk_prefill failed: "
      << cudaGetErrorString(err);
}

namespace {

cudaError_t launch_persistent_cluster_topk(
    const vllm::persistent::PersistentTopKParams& params, uint32_t num_rows,
    uint32_t cluster_size, cudaStream_t stream,
    void (*kernel)(vllm::persistent::PersistentTopKParams), bool enable_pdl,
    bool allow_nonportable_cluster_size) {
  namespace P = vllm::persistent;
  if (allow_nonportable_cluster_size) {
    const cudaError_t attribute_error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (attribute_error != cudaSuccess) return attribute_error;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(num_rows, cluster_size, 1);
  config.blockDim = dim3(P::kThreadsPerBlock);
  config.dynamicSmemBytes = 0;
  config.stream = stream;

  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  attributes[1].id = cudaLaunchAttributeClusterDimension;
  attributes[1].val.clusterDim = {1, cluster_size, 1};
  config.attrs = attributes;
  config.numAttrs = 2;

  return cudaLaunchKernelEx(&config, kernel, params);
}

bool device_supports_cluster_launch(int device_id) {
  int supported = 0;
  const cudaError_t error = cudaDeviceGetAttribute(
      &supported, cudaDevAttrClusterLaunch, device_id);
  if (error != cudaSuccess) {
    // Treat an unavailable capability query conservatively. Clear the runtime
    // error before launching the non-cluster fallback below.
    (void)cudaGetLastError();
    return false;
  }
  return supported != 0;
}

bool is_unsupported_cluster_launch_error(cudaError_t error) {
  return error == cudaErrorInvalidValue ||
         error == cudaErrorInvalidConfiguration ||
         error == cudaErrorInvalidClusterSize ||
         error == cudaErrorNotSupported;
}

int device_sm_count(int device_id) {
  // Serving workers use one GPU or homogeneous full GPUs.
  static const int sm_count = [device_id] {
    int count = 0;
    const cudaError_t error = cudaDeviceGetAttribute(
        &count, cudaDevAttrMultiProcessorCount, device_id);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    return count;
  }();
  return sm_count;
}

template <int TopK, int Level, bool kUsePDL>
void launch_noncluster_topk_level(
    const vllm::persistent::PersistentTopKParams& params,
    bool prefer_single_cta, cudaStream_t stream) {
  namespace P = vllm::persistent;
  auto kernel = P::persistent_topk_kernel<TopK, Level, kUsePDL, 2>;
  if constexpr (kUsePDL) {
    // A one-block register budget spreads small grids of waiting CTAs
    // across SMs as scoring finishes. The compiler may trade occupancy
    // for fewer instructions in register and streaming selectors.
    if (prefer_single_cta) {
      kernel = P::persistent_topk_kernel<TopK, Level, true, 1>;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(params.num_rows);
    config.blockDim = dim3(P::kThreadsPerBlock);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    config.attrs = &attribute;
    config.numAttrs = 1;
    const cudaError_t error = cudaLaunchKernelEx(&config, kernel, params);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  } else {
    kernel<<<params.num_rows, P::kThreadsPerBlock, 0, stream>>>(params);
  }
}

template <int TopK, bool kUsePDL>
void launch_noncluster_persistent_topk(
    const vllm::persistent::PersistentTopKParams& params, int64_t num_rows,
    int64_t max_seq_len, int device_id, cudaStream_t stream) {
  namespace P = vllm::persistent;
  bool prefer_single_cta = false;
  if constexpr (kUsePDL) {
    prefer_single_cta = num_rows <= device_sm_count(device_id);
  }
  if (max_seq_len <= P::kRegister2MaxSeqLen) {
    launch_noncluster_topk_level<TopK, 0, kUsePDL>(params, prefer_single_cta,
                                                   stream);
  } else if (max_seq_len <= P::kRegister4MaxSeqLen) {
    launch_noncluster_topk_level<TopK, 1, kUsePDL>(params, prefer_single_cta,
                                                   stream);
  } else {
    launch_noncluster_topk_level<TopK, 2, kUsePDL>(params, prefer_single_cta,
                                                   stream);
  }
}

template <int TopK, bool kUsePDL>
void launch_persistent_topk(const TensorView& logits, const TensorView& lengths,
                            TensorView& output, int64_t max_seq_len,
                            int64_t q_len_per_req, cudaStream_t stream) {
  namespace P = vllm::persistent;

  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.size(1);

  P::PersistentTopKParams params;
  params.input = static_cast<float*>(logits.data_ptr());
  params.output = static_cast<int32_t*>(output.data_ptr());
  params.lengths = static_cast<int32_t*>(lengths.data_ptr());
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = static_cast<uint32_t>(stride);
  params.top_k = static_cast<uint32_t>(TopK);
  params.max_seq_len = static_cast<uint32_t>(max_seq_len);
  params.q_len_per_req = static_cast<uint32_t>(q_len_per_req);

  cudaError_t launch_error = cudaSuccess;
  const bool use_cluster =
      max_seq_len > P::kClusterFloor && num_rows <= 32 &&
      device_supports_cluster_launch(logits.device().device_id);
  if (use_cluster) {
    const auto rows = static_cast<uint32_t>(num_rows);
    if (max_seq_len > P::kClusterHist10Floor &&
        max_seq_len <= P::kClusterRegister4MaxSeqLen) {
      launch_error = launch_persistent_cluster_topk(
          params, rows, 8, stream,
          P::cluster8_register4_topk_kernel<TopK, 10, kUsePDL>, kUsePDL, false);
    } else if (max_seq_len > P::kClusterRegister4MaxSeqLen &&
               max_seq_len <= P::kCluster16MaxSeqLen) {
      launch_error = launch_persistent_cluster_topk(
          params, rows, 16, stream, P::cluster16_topk_kernel<TopK, 10, kUsePDL>,
          kUsePDL, true);
      if (is_unsupported_cluster_launch_error(launch_error)) {
        // A cluster of 16 CTAs is an architecture-specific optimization. Keep
        // the portable cluster-8 implementation as the correctness fallback.
        (void)cudaGetLastError();
        launch_error = launch_persistent_cluster_topk(
            params, rows, P::kClusterSize, stream,
            P::cluster_topk_kernel<TopK, 10, kUsePDL>, kUsePDL, false);
      }
    } else if (max_seq_len > P::kClusterHist10Floor) {
      launch_error = launch_persistent_cluster_topk(
          params, rows, P::kClusterSize, stream,
          P::cluster_topk_kernel<TopK, 10, kUsePDL>, kUsePDL, false);
    } else {
      launch_error = launch_persistent_cluster_topk(
          params, rows, P::kClusterSize, stream,
          P::cluster_topk_kernel<TopK, 9, kUsePDL>, kUsePDL, false);
    }
    if (is_unsupported_cluster_launch_error(launch_error)) {
      // Some Hopper SKUs reject cluster launches despite sharing sm_90 with
      // cluster-capable devices. Fall back to the original single-CTA kernel.
      (void)cudaGetLastError();
      launch_noncluster_persistent_topk<TopK, kUsePDL>(
          params, num_rows, max_seq_len, logits.device().device_id, stream);
      launch_error = cudaSuccess;
    }
  } else {
    launch_noncluster_persistent_topk<TopK, kUsePDL>(
        params, num_rows, max_seq_len, logits.device().device_id, stream);
  }

  cudaError_t err =
      launch_error == cudaSuccess ? cudaGetLastError() : launch_error;
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "dsv4_persistent_topk failed: " << cudaGetErrorString(err);
}

template <int TopK>
void dispatch_persistent_topk(const TensorView& logits,
                              const TensorView& lengths, TensorView& output,
                              int64_t max_seq_len, int64_t q_len_per_req,
                              bool enable_pdl, cudaStream_t stream) {
  if (enable_pdl) {
    launch_persistent_topk<TopK, true>(logits, lengths, output, max_seq_len,
                                       q_len_per_req, stream);
  } else {
    launch_persistent_topk<TopK, false>(logits, lengths, output, max_seq_len,
                                        q_len_per_req, stream);
  }
}

}  // namespace

void dsv4_persistent_topk(TensorView logits, TensorView lengths,
                          TensorView output, int64_t k, int64_t max_seq_len,
                          int64_t q_len_per_req, bool enable_pdl) {
  CHECK_CUDA(logits);
  CHECK_CUDA(lengths);
  CHECK_CUDA(output);
  CHECK_DIM(2, logits);
  TVM_FFI_ICHECK(lengths.ndim() == 1 || lengths.ndim() == 2)
      << "lengths must be 1D or 2D";
  CHECK_DIM(2, output);
  CHECK_CONTIGUOUS(logits);
  CHECK_CONTIGUOUS(lengths);
  CHECK_CONTIGUOUS(output);
  TVM_FFI_ICHECK(logits.dtype() == dl_float32) << "logits must be float32";
  TVM_FFI_ICHECK(lengths.dtype() == dl_int32) << "lengths must be int32";
  TVM_FFI_ICHECK(output.dtype() == dl_int32) << "output must be int32";
  TVM_FFI_ICHECK(q_len_per_req >= 1) << "q_len_per_req must be >= 1";
  TVM_FFI_ICHECK(logits.size(0) % q_len_per_req == 0)
      << "q_len_per_req must divide num_rows";
  TVM_FFI_ICHECK(lengths.numel() == logits.size(0) / q_len_per_req)
      << "lengths size mismatch (expected num_rows / q_len_per_req)";
  TVM_FFI_ICHECK(output.size(0) == logits.size(0) && output.size(1) == k)
      << "output size mismatch";
  TVM_FFI_ICHECK(k == 512 || k == 1024 || k == 2048)
      << "persistent topk supports k=512, k=1024, or k=2048, got " << k;
  TVM_FFI_ICHECK(max_seq_len > 0) << "max_seq_len must be positive";

  cudaError_t err = cudaSetDevice(logits.device().device_id);
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "cudaSetDevice failed: " << cudaGetErrorString(err);
  cudaStream_t stream = get_stream(logits.device());
  if (k == 512) {
    dispatch_persistent_topk<512>(logits, lengths, output, max_seq_len,
                                  q_len_per_req, enable_pdl, stream);
  } else if (k == 1024) {
    dispatch_persistent_topk<1024>(logits, lengths, output, max_seq_len,
                                   q_len_per_req, enable_pdl, stream);
  } else {
    dispatch_persistent_topk<2048>(logits, lengths, output, max_seq_len,
                                   q_len_per_req, enable_pdl, stream);
  }
}
