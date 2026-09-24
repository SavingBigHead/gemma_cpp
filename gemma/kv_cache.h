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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_

#include <stddef.h>

#include "gemma/configs.h"  // ModelConfig
#include "gemma/gemma_args.h"  // InferenceArgs
#include "util/basics.h"       // BF16
#include "util/mat.h"

namespace gcpp {

using KV_t = float;

struct KVCache {
  // KV Cache：缓存每个 token 在每一层、每个 KV head 的键和值向量。
  //
  // 内存布局：[seq_len, layers × kv_heads × qkv_dim × 2]
  //   - 行（第一维）= 时间步（token 位置），追加写入局部性好
  //   - 列（第二维）= 按层 → head → [K|V] 交错排列
  //   - 末尾 ×2 是因为每个 head 存 K 和 V 各 qkv_dim 个元素
  //
  // 推理时每生成一个新 token，只需把它的 K/V 写入对应行；
  // 注意力阶段通过 MatPtrT 视图按 stride 灵活读取某一层某一头的 K 或 V。
  // 当写入位置超过 seq_len 时，用取模回绕实现环形缓冲（配合滑动窗口）。
  KVCache(const ModelConfig& config, const InferenceArgs& inference_args,
          const Allocator& allocator);

  // Returns a deep copy of the KVCache. Use explicit function instead of
  // copy ctor to make the cost explicit.
  KVCache Copy();

  size_t SeqLen() const { return kv_cache.Rows(); }

  MatStorageT<KV_t> kv_cache;  // [seq_len, layers * kv_heads * qkv_dim * 2]

 private:
  const Allocator& allocator_;

  // For use by other ctor and Copy()
  KVCache(const Extents2D& kv_extents, const Allocator& allocator);
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_KV_CACHE_H_
