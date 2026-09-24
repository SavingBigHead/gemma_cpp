// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Transformer components shared between vit.cc and attention.cc.

#include <stddef.h>
#include <stdint.h>

#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "hwy/profiler.h"
#include "ops/matmul.h"
#include "util/mat.h"
#include "util/threading.h"
#include "util/zones.h"

// Include guard (still compiled once per target)
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
#endif

#include "hwy/highway.h"
// After highway.h
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// For use by Vit even if !GEMMA_FUSED_FFN.
template <typename T1, typename T2>
void Activation(ActivationType activation, T1* HWY_RESTRICT c1,
                const T2* HWY_RESTRICT c2, const size_t count,
                ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kGenActivation);
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  // ActivationType::Gelu
  if (c2 == nullptr) {  // No multiplier, just Gelu.
    Gelu(c1, count);
    return;
  };
  // Has multiplier, Gelu(c1) * c2.
  Decompress1AndCompressInplace(DF(), c1, count, c2, /*p1_ofs=*/0,
                                [](DF df, VF v1, VF v2) HWY_ATTR -> VF {
                                  return hn::Mul(v2, Gelu(df, v1));
                                });
}

// No C2 multiplier - used by Vit.
template <class Mat>
void ActivationBatched(
    ActivationType activation, Mat& c1, ThreadingContext& ctx,
    size_t cluster_idx = 0,
    ParallelismStrategy parallelism = ParallelismStrategy::kFlat) {
  using T = typename Mat::T;
  ParallelFor(parallelism, c1.Rows(), ctx, cluster_idx,
              Callers::kActivationBatched, [&](uint64_t task, size_t worker) {
                // Cast to correct type so type deduction works.
                Activation(activation, c1.Row(task),
                           static_cast<const T*>(nullptr), c1.Cols(), ctx,
                           worker);
              });
}

#if GEMMA_FUSED_FFN

// Called during `TwoMatMul`.
static inline void Activation(ActivationType activation, const RowPtrsBF C1,
                              const IndexRange range_r,
                              const IndexRange range_c, const StridedViewBF C2,
                              ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kGenActivationFused);

  const size_t cols = range_c.Num();
  HWY_DASSERT(C2.Cols() == cols);

  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  // ActivationType::Gelu
  // Gated: Gelu(c1) * c2.
  for (size_t ir = 0; ir < range_r.Num(); ++ir) {
    Decompress1AndCompressInplace(
        DF(), C1.Row(range_r.begin() + ir) + range_c.begin(), cols, C2.Row(ir),
        /*p1_ofs*/ 0, [](DF df, VF v1, VF v2) HWY_ATTR -> VF {
          return hn::Mul(v2, Gelu(df, v1));
        });
  }
}

#endif  // GEMMA_FUSED_FFN

// Only used if !GEMMA_FUSED_FFN, but define anyway so that we can check
// using if constexpr rather than #if, which interferes with code folding.
template <class Mat1, class Mat2>
HWY_NOINLINE void ActivationBatched(
    ActivationType activation, Mat1& c1, const Mat2* c2, ThreadingContext& ctx,
    size_t cluster_idx = 0,
    ParallelismStrategy parallelism = ParallelismStrategy::kFlat) {
  HWY_DASSERT(c1.SameShape(*c2));
  if (c2 && c2->HasPtr()) {
    ParallelFor(parallelism, c1.Rows(), ctx, cluster_idx,
                Callers::kActivationBatched, [&](uint64_t task, size_t worker) {
                  Activation(activation, c1.Row(task), c2->Row(task), c1.Cols(),
                             ctx, worker);
                });
  } else {  // No multiplier
    ParallelFor(parallelism, c1.Rows(), ctx, cluster_idx,
                Callers::kActivationBatched, [&](uint64_t task, size_t worker) {
                  Activation(activation, c1.Row(task),
                             static_cast<const typename Mat2::T*>(nullptr),
                             c1.Cols(), ctx, worker);
                });
  }
}

template <typename T2, class LayerWeights>
HWY_NOINLINE void ResidualConnection(const MatPtrT<T2>& other,
                                     MatPtrT<float>& HWY_RESTRICT x,
                                     const LayerWeights& layer,
                                     bool is_attention, ThreadingContext& ctx) {
  // 残差连接：x = x + f(x)。
  // other 是子层输出（BF16），x 是残差流（float），相加时自动提升精度。
  // 残差让信息可以"跳过"当前层直达深层，是训练深网络的关键。
  // ResidualType::Add
  AddFromBatched(other, x, ctx);
}

template <typename InOutT>
void PostNorm(PostNormType post_norm, const MatPtr& weights,
              MatPtrT<InOutT>& inout, ThreadingContext& ctx) {
  // PostNorm：在子层输出上加残差之前，可选地做一次 RMSNorm。
  // Gemma 3 引入，进一步稳定深层残差流的数值范围。
  HWY_DASSERT(weights.Rows() == 1);
  if (post_norm == PostNormType::Scale) {
    RMSNormInplaceBatched(weights, inout, ctx);
  }
}

static inline void FFWNoVit(const LayerWeightsPtrs& layer,
                            Activations& activations, MatMulEnv& env) {
  // 前馈网络（FFN），使用门控 GELU（GeGLU）结构：
  //   output = (GELU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
  // 其中 ⊙ 是逐元素乘。两个分支共享同一输入 pre_ffw_rms_out。
  //
  // 默认走 GEMMA_FUSED_FUSED 融合路径（CallTwoMatMul）：
  //   两个矩阵乘的 tile 算完后立刻做激活+逐元素乘，中间结果不落地，
  //   输入只加载一次，权重 tile 的缓存复用率更高。
  // 非融合路径先分别算两个矩阵乘再激活，便于阅读但多一趟内存往返。
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenFFW);
  const LayerConfig& layer_config = layer.layer_config;

  HWY_DASSERT(!layer_config.ff_biases);  // Only used in Vit.

#if GEMMA_FUSED_FFN
  const auto fused = [&](RowPtrsBF C1, IndexRange range_r, IndexRange range_c,
                         StridedViewBF C2, size_t worker) {
    Activation(layer_config.activation, C1, range_r, range_c, C2, env.ctx,
               worker);
  };
  MMOptions options;
  options.SetFunc(fused);
  CallTwoMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1,
                layer.gating_einsum_w2, env, activations.C1, options);
#else
  // Compute the hidden layer activations.
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1, nullptr, env,
             activations.C1);
  CallMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w2, nullptr, env,
             activations.C2);
  // Activation (Gelu) and maybe multiply by gate. Store activations in act.
  ActivationBatched(layer_config.activation, activations.C1, &activations.C2,
                    env.ctx);
#endif

  // Hidden layer -> output layer.
  CallMatMul(activations.C1, layer.linear_w, nullptr, env, activations.ffw_out);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_GEMMA_INL_H_
