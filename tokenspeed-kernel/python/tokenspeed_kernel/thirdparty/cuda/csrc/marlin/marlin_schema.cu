// Marlin operator schemas. Implementations are provided by the vendored
// vLLM-derived CUDA source in this directory.
#include <torch/library.h>

TORCH_LIBRARY(_C, m) {
  m.def("marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, Tensor? b_bias_or_none, Tensor b_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor workspace, int b_type_id, SymInt size_m, SymInt size_n, SymInt size_k, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
}
