// Copyright 2025 Google LLC
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

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "util/zones.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/configs.h"  // kMaxQKVDim
#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "util/threading.h"
#include "util/threading_context.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/attention.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/flash_attention.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Computes Q.K scores, which are "logits" (or scores) stored to att.
// `k` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
//
// 注意力得分计算的第一步：当前查询向量 Q 与每个历史位置的
// 键向量 K 做点积，得到"相似度分数"，暂存到 att 数组中。
// 点积结果越大，表示当前 token 越应该"关注"那个历史位置的 token。
// 这里没有做 softmax，softmax 在调用方 QDotK 之后再统一做。
// div_seq_len 用于 KV Cache 环形缓冲区的取模回绕（wraparound）。
static HWY_INLINE void QDotK(const size_t start_pos, const size_t last_pos,
                             const hwy::Divisor& div_seq_len,
                             const float* HWY_RESTRICT q,
                             const MatPtrT<KV_t>& k, float* HWY_RESTRICT att,
                             ThreadingContext& ctx, const size_t worker) {
  GCPP_ZONE(ctx, worker, Zones::kGenAttentionQDotK);
  if (HWY_LIKELY(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()))) {
    // Slightly faster: no wraparound.
    for (size_t pos = start_pos; pos <= last_pos; ++pos) {
      const float score = Dot(q, k.Row(pos), k.Cols());
      att[pos] = score;
    }
  } else {
    for (size_t pos = start_pos; pos <= last_pos; ++pos) {
      const size_t pos_modulo = div_seq_len.Remainder(pos);
      const float score = Dot(q, k.Row(pos_modulo), k.Cols());
      att[pos_modulo] = score;
    }
  }
}

void PositionalEncodingQK(float* qk, const size_t layer_idx,
                          const LayerWeightsPtrs& layer,
                          const AttentionActivations& activations,
                          ThreadingContext& ctx, const size_t worker,
                          const size_t pos, const float mul) {
  // 对 Q 或 K 向量施加 RoPE 旋转位置编码。
  // RoPE 把向量的每一对维度看作复平面上的一个点，按 token 位置旋转
  // 相应角度，使得 Q·K 的点积只依赖于两个 token 的"相对位置"。
  // mul 通常是 query_scale（即 1/sqrt(d)），在旋转的同时完成缩放，
  // 减少一趟独立的内存读写。
  const size_t qkv_dim = layer.layer_config.qkv_dim;
  const PostQKType& post_qk = layer.layer_config.post_qk;
  // qk is either q or k, so qkv_dim is the length we operate on.
  const float* inv_timescale = activations.inv_timescale.PackedScale1();
  const bool is_global_layer = activations.config.IsGlobalLayer(layer_idx);
  // TODO: add a config flag instead of hardcoding the model.
  if (is_global_layer && IsVLM(activations.config.model)) {
    inv_timescale = activations.inv_timescale_global.PackedScale1();
  }
  // PostQKType::Rope
  if (post_qk == PostQKType::HalfRope) {
    Rope(qk, qkv_dim / 2, inv_timescale, pos, ctx, worker);
    if (mul != 1.0f) MulByConst(mul, qk, qkv_dim);
  } else {
    RopeAndMulBy(mul, qk, qkv_dim, inv_timescale, pos, ctx, worker);
  }
}

// Accumulates the sum of v (from `kv_cache`) * probability (`att`) into
// `att_out`. Equivalent in gemma/modules.py:
// encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
// `v` is a strided view of the kv cache with dimensions [seq_len, qkv_dim].
//
// 注意力的最后一步：用 softmax 得到的概率分布 att，
// 对所有历史位置的值向量 V 做加权求和，得到当前 head 的输出。
// 数学上：output = Σ p_i * V_i。
// 第一个位置用 MulByConstTo 初始化输出，后续位置用 MulByConstAndAdd
// 累加。这是一个带宽受限的循环——V 数据量通常远大于 att。
static HWY_INLINE void WeightedSumV(
    const size_t start_pos, const size_t last_pos,
    const hwy::Divisor& div_seq_len, const float* HWY_RESTRICT att,
    const MatPtrT<KV_t>& v, float* HWY_RESTRICT att_out, ThreadingContext& ctx,
    const size_t worker) {
  if (HWY_LIKELY(last_pos < static_cast<size_t>(div_seq_len.GetDivisor()))) {
    // Slightly faster: no wraparound. Could be replaced with MatMul(att, v) if
    // we supported non-transposed B.
    // TODO: 2..4x unroll
    MulByConstTo(att[start_pos], v.Row(start_pos), att_out, v.Cols(), ctx,
                 worker);
    for (size_t pos = start_pos + 1; pos <= last_pos; ++pos) {
      MulByConstAndAdd(att[pos], v.Row(pos), att_out, v.Cols());
    }
  } else {
    {
      const size_t pos_mod = div_seq_len.Remainder(start_pos);
      MulByConstTo(att[pos_mod], v.Row(pos_mod), att_out, v.Cols(), ctx,
                   worker);
    }
    for (size_t pos = start_pos + 1; pos <= last_pos; ++pos) {
      const size_t pos_mod = div_seq_len.Remainder(pos);
      MulByConstAndAdd(att[pos_mod], v.Row(pos_mod), att_out, v.Cols());
    }
  }
}

// Calculates the attention outputs for a single q, which may be updated
// in place for RMSNorm.
//
// 朴素注意力的完整流程（单查询、单 head）：
//   1. 可选：对 Q 做 RMSNorm（query_norm）
//   2. 对 Q 施加 RoPE + query_scale
//   3. Q·K 计算注意力得分
//   4. SoftCap（可选的 tanh 限幅，防止分数过度集中）
//   5. Softmax 归一化为概率分布
//   6. 按概率对 V 加权求和得到输出
void SingleDotSoftmaxWeightedSum(
    const size_t pos, const size_t start_pos, const size_t last_pos,
    float* HWY_RESTRICT q, const MatPtrT<KV_t>& k, const MatPtrT<KV_t>& v,
    const size_t layer_idx, const LayerWeightsPtrs& layer,
    const AttentionActivations& activations, float* HWY_RESTRICT att,
    float* HWY_RESTRICT att_out, ThreadingContext& ctx, const size_t worker) {
  const float att_cap = activations.config.att_cap;
  const float query_scale = activations.query_scale;
  const size_t seq_len =
      static_cast<size_t>(activations.div_seq_len.GetDivisor());

  // Apply rope and scaling to Q.
  if (layer.query_norm_scale.HasPtr()) {
    CallUpcasted(&layer.query_norm_scale, [&](const auto* weights_t) {
      RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, q,
                     layer.layer_config.qkv_dim, ctx, worker);
    });
  }

  PositionalEncodingQK(q, layer_idx, layer, activations, ctx, worker, pos,
                       query_scale);

  QDotK(start_pos, last_pos, activations.div_seq_len, q, k, att, ctx, worker);

  // SoftMax with optional SoftCap yields "probabilities" in att.
  const size_t att_len = HWY_MIN(last_pos + 1, seq_len);
  const Logits logits(att, att_len);
  MaybeLogitsSoftCap(att_cap, logits, ctx, worker);
  Softmax(logits, ctx, worker, /*temperature=*/1.0f);

  WeightedSumV(start_pos, last_pos, activations.div_seq_len, att, v, att_out,
               ctx, worker);
}

// The attention window usually starts at 0 unless `pos` is larger than
// the attention window size, then it is `pos` - window_size + 1.
//
// 滑动窗口注意力的起点计算。
// Gemma 3 的层分为 local（只看最近 N 个 token）和 global（看全部）。
// 如果当前位置 pos 超过窗口大小，注意力只需从 pos - (window-1) 开始，
// 更早的 KV Cache 直接跳过，大幅降低长文本的计算量和带宽。
size_t StartPos(size_t pos, const ModelConfig& config, size_t layer_idx) {
  const size_t att_window_size = config.attention_window_sizes[layer_idx];
  return pos - HWY_MIN(att_window_size - 1, pos);
}

void DotSoftmaxWeightedSum(const size_t num_tokens, const size_t layer_idx,
                           const LayerWeightsPtrs& layer,
                           AttentionActivations& activations, QBatch& qbatch,
                           ThreadingContext& ctx) {
  // 朴素注意力路径的批处理调度入口。
  // 将 [token × query × head] 三维任务空间展开成一维，用线程池并行。
  // 每个任务处理一个 (token, head) 组合：
  //   1. 从 KV Cache 构造当前 head 对应的 K、V 视图（零拷贝）
  //   2. 调用 SingleDotSoftmaxWeightedSum 完成完整注意力
  //
  // GQA 映射：head / kHeadGroups 把多个 Q head 映射到同一个 KV head，
  // 多个 Q head 共享同一块 K/V 内存区域，从而缩小 KV Cache。
  GCPP_ZONE(ctx, 0, Zones::kGenAttentionDotSoftmaxWeightedSumInclusive);

  const hwy::Divisor div_qbatch(qbatch.Size());
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;

  // A "head group" in the context of GQA refers to a collection of query
  // heads that share the same key and value heads.
  const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;

  const size_t cache_layer_size = layer_config.CacheLayerSize();
  const size_t seq_len =
      static_cast<size_t>(activations.div_seq_len.GetDivisor());
  // All layers should have the same number of heads.
  HWY_DASSERT(activations.div_heads.GetDivisor() == layer_config.heads);

  // For each head/token/query, compute Q.K, softmax, and weighted V.
  const auto func = [&](const size_t task, size_t worker) HWY_ATTR {
    const size_t tq_idx = activations.div_heads.Divide(task);
    const size_t head = activations.div_heads.Remainder(task);
    GCPP_ZONE(ctx, worker, Zones::kGenAttentionDotSoftmaxWeightedSumPar);

    const size_t qi = div_qbatch.Remainder(tq_idx);
    const size_t batch_idx = div_qbatch.Divide(tq_idx);
    auto& kv_cache = qbatch.KV(qi).kv_cache;

    // Find the token position in the query and calculate
    // the range of cache positions to attend to.
    const size_t pos = qbatch.Pos(qi) + batch_idx;
    const size_t start_pos = StartPos(pos, activations.config, layer_idx);
    size_t last_pos = pos;
    const size_t prefix_end = qbatch.PrefixEnd(qi);
    if (prefix_end > 0 && prefix_end - 1 > last_pos) {
      // last_pos in QDotK and WeightedSumV is inclusive.
      last_pos = prefix_end - 1;
    }

    float* HWY_RESTRICT q = activations.q.Row(tq_idx) + head * qkv_dim;
    float* HWY_RESTRICT att = activations.att.Row(tq_idx) + head * seq_len;
    float* HWY_RESTRICT att_out =
        activations.att_out.Row(tq_idx) + head * qkv_dim;

    // Make strided read-only views into the kv cache for
    // this query and head.
    const size_t head_offset = (head / kHeadGroups) * qkv_dim * 2;
    const size_t kv_head_offset = layer_idx * cache_layer_size + head_offset;
    MatPtrT<KV_t> k("k_view", Extents2D(seq_len, qkv_dim));
    k.SetPtr(kv_cache.Row(0) + kv_head_offset, kv_cache.Stride());
    MatPtrT<KV_t> v("v_view", Extents2D(seq_len, qkv_dim));
    v.SetPtr(kv_cache.Row(0) + kv_head_offset + qkv_dim, kv_cache.Stride());

    SingleDotSoftmaxWeightedSum(pos, start_pos, last_pos, q, k, v, layer_idx,
                                layer, activations, att, att_out, ctx, worker);
  };

  {
    PROFILER_ZONE("Gen.Attention.DotSoftmax.ForkJoin");
    // Full parallelism is helpful, kAcrossClusters is insufficient.
    HierarchicalParallelFor(
        num_tokens * div_qbatch.GetDivisor() * layer_config.heads, ctx,
        Callers::kAttDotSoftmaxWeightedSum, func);
  }
}

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`. `num_tokens` is the
// number of tokens from one query: 1 for decode, otherwise prefill_tbatch_size.

// Fills activations.q and writes to KV cache.
//
// QKV 投影：把归一化后的输入 x 乘以两个权重矩阵，
// 生成 Q（只需当前 token）和 K/V（写入 KV Cache 供后续复用）。
// K/V 的矩阵乘输出行指针直接指向 Cache 对应位置（零拷贝写入）。
// 写入后对 K 做进一步处理（可选 RMSNorm + RoPE）并压缩存储。
static HWY_INLINE void ComputeQKV(size_t num_tokens, const size_t layer_idx,
                                  const LayerWeightsPtrs& layer,
                                  AttentionActivations& activations,
                                  const QBatch& qbatch, const int flags,
                                  MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(),
            Zones::kGenAttentionComputeQKV);

  const hwy::Divisor div_qbatch(qbatch.Size());
  const size_t num_interleaved = num_tokens * div_qbatch.GetDivisor();
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t kv_heads = layer_config.kv_heads;
  const size_t cache_layer_size = layer_config.CacheLayerSize();

  // The original qkv_einsum_w has shape [(heads + kv_heads * 2), qkv_dim,
  // model_dim], which we reshaped to (heads + kv_heads * 2) * qkv_dim rows.
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w1,
             /*add=*/nullptr, env, activations.q);

  // Set up MatMul row pointers for writing to KV, which consists of
  // `kv_heads` pairs of (k, v) vectors. This safely handles wraparound
  // because rows are computed modulo seq_len.
  MatPtrT<KV_t> kv_rows("kv", Extents2D(activations.pre_att_rms_out.Rows(),
                                        layer.qkv_einsum_w2.Rows()));
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    const size_t qi = div_qbatch.Remainder(interleaved_idx);
    const size_t batch_idx = div_qbatch.Divide(interleaved_idx);
    const size_t cache_pos =
        activations.div_seq_len.Remainder(qbatch.Pos(qi) + batch_idx);
    env.row_ptrs[0][interleaved_idx] = reinterpret_cast<uint8_t*>(
        qbatch.KV(qi).kv_cache.Row(cache_pos) + layer_idx * cache_layer_size);
  }
  kv_rows.AttachRowPtrs(env.row_ptrs[0].get());
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w2,
             /*add=*/nullptr, env, kv_rows);

  // Apply positional encodings for K.
  // Note that 2D parallelism is not worth the fork/join overhead because the
  // tasks are very lightweight.
  ParallelFor(
      ParallelismStrategy::kFlat, kv_heads * num_interleaved, env.ctx,
      /*cluster_idx=*/0, Callers::kAttComputeQKV,
      [&](size_t task, size_t worker) HWY_ATTR {
        const size_t head = task % kv_heads;
        const size_t interleaved_idx = task / kv_heads;
        const size_t qi = div_qbatch.Remainder(interleaved_idx);
        const size_t batch_idx = div_qbatch.Divide(interleaved_idx);
        const size_t pos = qbatch.Pos(qi) + batch_idx;
        const size_t cache_pos = activations.div_seq_len.Remainder(pos);
        auto& kv_cache = qbatch.KV(qi).kv_cache;
        KV_t* HWY_RESTRICT kv = kv_cache.Row(cache_pos) +
                                layer_idx * cache_layer_size +
                                head * qkv_dim * 2;

        HWY_ALIGN float kv_f32[2 * kMaxQKVDim];
        const hn::ScalableTag<float> df;
        DecompressAndZeroPad(df, MakeSpan(kv, 2 * qkv_dim), 0, kv_f32,
                             2 * qkv_dim);

        // Apply further processing to K.
        if (layer.key_norm_scale.HasPtr()) {
          CallUpcasted(&layer.key_norm_scale, [&](const auto* weights_t) {
            RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, kv_f32,
                           qkv_dim, env.ctx, worker);
          });
        }

        PositionalEncodingQK(kv_f32, layer_idx, layer, activations, env.ctx,
                             worker, pos, /*mul=*/1.0f);
        CompressPerThread tls;
        Compress(kv_f32, 2 * qkv_dim, tls, MakeSpan(kv, 2 * qkv_dim), 0);
      });
}

// Sums encoded (`att_out`) over num_heads (`layer_config.heads`) and
// head_dim (`qkv_dim`) into output (`layer_out`).
//
// 合并各头输出。每个 head 独立算出的 att_out（长度 qkv_dim）
// 拼接成一行，再乘以输出投影矩阵，求和还原为 model_dim 维向量。
// 这一步实现了 W_o 投影：output = concat(head_1..head_h) @ W_o。
static HWY_INLINE void SumHeads(const LayerWeightsPtrs& layer,
                                AttentionActivations& activations,
                                MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenAttentionSumHeads);
  const LayerConfig& layer_config = layer.layer_config;
  (void)layer_config;  // For HWY_DASSERT
  // att_weights and att_out are concatenated heads, each of length
  // layer_config.qkv_dim. Thus the [num_interleaved,
  // layer_config.model_dim] matmul output is the sum over heads. Compare
  // gemma/modules.py: attn_output = self.attn_vec_einsum('BTNH,NHD->BTD',
  // encoded)
  HWY_DASSERT(layer_config.model_dim != 0 && layer_config.heads != 0 &&
              layer_config.qkv_dim != 0);
  CallMatMul(activations.att_out, layer.att_weights, /*add=*/nullptr, env,
             activations.att_sums);
}

void GemmaAttention(size_t num_tokens, const size_t layer_idx,
                    const LayerWeightsPtrs& layer,
                    AttentionActivations& activations, QBatch& qbatch,
                    MatMulEnv& env, int flags) {
  // Gemma 注意力的总入口，分三步：
  //   1. ComputeQKV：投影生成 Q/K/V，K/V 写入缓存
  //   2. 注意力主体：朴素（kAttentionUseOld）或 Flash Attention
  //   3. SumHeads：合并多头输出并投影回 model_dim
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenAttention);

  const LayerConfig& layer_config = layer.layer_config;
  HWY_DASSERT(!layer_config.IsMHA());  // No longer supported.
  HWY_DASSERT_M((layer_config.heads % layer_config.kv_heads) == 0,
                "query heads must be a multiple of key-value heads");
  (void)layer_config;  // only used in HWY_DASSERT

  ComputeQKV(num_tokens, layer_idx, layer, activations, qbatch, flags, env);
  if (flags & kAttentionUseOld) {
    DotSoftmaxWeightedSum(num_tokens, layer_idx, layer, activations, qbatch,
                          env.ctx);
  } else {
    // * 2 does not help on Turin.
    FlashAttention(num_tokens,
                   /*target_parallelism=*/env.ctx.pools.MaxWorkers() * 1,
                   layer_idx, layer, activations, qbatch, env.ctx);
  }
  SumHeads(layer, activations, env);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
