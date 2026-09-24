# Transformer 推理核心详解（基于 gemma.cpp 源码）

> 本文基于本仓库 `gemma/`、`ops/` 目录的真实源码，逐层拆解 Gemma 模型在 CPU 上做推理的完整链路，讲清每一步"用了什么技术"以及"为什么要这么设计"。

---

## 目录

1. [推理全景图](#1-推理全景图)
2. [Embedding 查表与缩放](#2-embedding-查表与缩放)
3. [Transformer 层总览](#3-transformer-层总览)
4. [RMSNorm：归一化](#4-rmsnorm归一化)
5. [注意力机制（Attention）](#5-注意力机制attention)
6. [KV Cache：避免重复计算](#6-kv-cache避免重复计算)
7. [RoPE：旋转位置编码](#7-rope旋转位置编码)
8. [Flash Attention：分块在线 softmax](#8-flash-attention分块在线-softmax)
9. [前馈网络（FFN）](#9-前馈网络ffn)
10. [残差连接](#10-残差连接)
11. [输出层与采样](#11-输出层与采样)
12. [Prefill 与 Decode 两阶段](#12-prefill-与-decode-两阶段)
13. [技术原理速查表](#13-技术原理速查表)
14. [模拟面试题](#14-模拟面试题)

---

## 1. 推理全景图

一次完整的文本生成，从用户输入 prompt 到逐 token 输出，数据流如下：

```text
用户 prompt
    │
    ▼
┌────────────────────┐
│ SentencePiece 分词  │  "你好世界" → [token_id_1, token_id_2, ...]
└────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                     Prefill（预填充阶段）                      │
│  把 prompt 中所有 token 一次性（或分批）送入模型，              │
│  填充每一层的 KV Cache                                        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                     Decode（解码阶段）                         │
│  每次只处理 1 个 token（或 batch 中每个请求各 1 个 token），   │
│  复用 KV Cache，逐个生成新 token，直到 EOS 或达到上限          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────┐
│  分词器 Decode      │  token_id → 文本
└────────────────────┘
    │
    ▼
用户看到的流式输出
```

对应源码位置：

| 阶段 | 文件 | 关键函数 |
|---|---|---|
| 整体调度 | `gemma/gemma.cc` | `GenerateT`、`PrefillQBatch`、`PrefillTBatch` |
| 单步前向 | `gemma/gemma.cc` | `Transformer` → `TransformerLayer` |
| 注意力 | `gemma/attention.cc` | `GemmaAttention`、`ComputeQKV`、`DotSoftmaxWeightedSum` |
| Flash Attention | `gemma/flash_attention.cc` | `FlashAttention`、`SingleFlashAttentionStep` |
| FFN | `gemma/gemma-inl.h` | `FFWNoVit` |
| 算子 | `ops/ops-inl.h` | `RMSNorm`、`Rope`、`Softmax`、`Top1OfSoftmax` |

---

## 2. Embedding 查表与缩放

### 做了什么

模型的输入不是文字，而是 token id（一个整数）。Embedding 层是一张大表 `[vocab_size, model_dim]`，用 token id 去查一行，得到一个 `model_dim` 维向量，作为后续所有层的输入。

源码：`gemma/gemma.cc` → `EmbedMMToken`

```cpp
const size_t embedding_ofs = token * weights_t->Stride();
DecompressAndZeroPad(df, embedding_span, embedding_ofs, x.Row(x_row), model_dim);
MulByConst(emb_scaling * weights_t->Scale(), x.Row(x_row), model_dim);
```

### 背后的原理

**为什么不直接用 one-hot？** 词表通常有 25 万个 token。one-hot 向量维度 25 万、且只有一个 1，极度稀疏。Embedding 把它映射到低维稠密空间（如 2048 维），语义相近的词在这个空间中距离也近——这是从"符号"到"连续数学"的关键桥梁。

**为什么要乘 `sqrt(model_dim)`？** Transformer 的残差流（residual stream）在每层都会累加信息。如果初始嵌入的数值太小，多层累加后信号会被淹没；乘以 `sqrt(model_dim)` 把初始信号放大到与后续各层输出相当的量级。Gemma 特意按 bf16 精度取整这个缩放值，以精确匹配参考实现。

**代码里的 `DecompressAndZeroPad` 是什么？** 权重可能以压缩格式（SFP、NUQ 等）存储，这个函数在读取时实时解压成 float，不落地中间数组——这是 gemma.cpp 把"压缩"与"计算"融合的核心手法。

### 共享权重

注意一个巧妙的细节：输出层（logits 计算）用的 `embedder_input_embedding` 和输入查表用的是**同一张表**的转置。输入时是"第 token 行 → 向量"，输出时是"向量 × 整张表 → 每个 token 的分数"。这叫 tied embedding，节省了一个巨大的参数矩阵。

---

## 3. Transformer 层总览

一个 Gemma 模型由 N 层（如 18 层、26 层）结构完全相同的 Transformer 层堆叠而成。每层的计算流程（源码 `TransformerLayer` 函数）：

```text
输入 x
  │
  ├─→ RMSNorm ──→ Attention ──────────────┐ (分支 1)
  │                                        │
  │              ┌── PostNorm (可选) ──────┤
  │              │                         ▼
  └──────────────┴──────────────→ x + attention_out   (残差相加)
                                     │
                                     ▼
                          RMSNorm ──→ FFN (Gated GELU) ─┐
                                                     │
                          ┌── PostNorm (可选) ────────┤
                          │                           ▼
                          └────────────────→ x + ffw_out  (残差相加)
                                             │
                                             ▼
                                        传给下一层
```

对应源码（`gemma/gemma.cc`）：

```cpp
static void TransformerLayer(...) {
  RMSNormBatched(activations.x, layer.pre_attention_norm_scale, ...);
  Attention(layer_config.type, num_tokens, layer_idx, layer, ...);
  PostNorm(layer_config.post_norm, layer.post_attention_norm_scale, ...);
  ResidualConnection(activations.attention.att_sums, activations.x, layer, ...);
  RMSNormBatched(activations.x, layer.pre_ffw_norm_scale, ...);
  FFWNoVit(layer, activations, env);
  PostNorm(layer_config.post_norm, layer.post_ffw_norm_scale, ...);
  ResidualConnection(activations.ffw_out, activations.x, layer, ...);
}
```

### 为什么要"归一化 → 子层 → 加残差"这个模式？

- **归一化（Pre-Norm）**：每层入口先把数值拉回稳定范围。如果不做，几十层累加后数值会指数级膨胀（梯度爆炸）或消失（梯度消失）。Pre-Norm（先归一化再进子层）比 Post-Norm 训练更稳定，是现代 LLM 的标准做法。
- **残差连接**：`x = x + f(x)` 而不是 `x = f(x)`。梯度可以通过"加法"这条捷径直接流回浅层，让几十层深的网络也能训练。推理时它只是简单的逐元素加法。

---

## 4. RMSNorm：归一化

### 做了什么

把一个向量的"长度"归一化到 1，再乘一个可学习的缩放系数。

数学公式：

```text
RMSNorm(x) = x / sqrt(mean(x_i²) + ε) * (1 + w)
```

源码：`ops/ops-inl.h` → `RMSNorm`

```cpp
// 第一步：算 x·x 的均值，再开根号的倒数
const float l2 = DecompressAndCall(d, MakeSpan(x, size), DotKernelDefault());
const float mul = 1.0f / sqrtf(l2 / size + 1e-6f);

// 第二步：逐元素乘 mul，再乘 (1 + w)
Decompress2AndCompressTo(DF(), out, size, x, weight, w_ofs,
    [pmul](DF, VF vx, VF vw) {
      const VF m = hn::Mul(*pmul, vx);
      // (1 + w) * m = m + w * m，恰好一条 FMA 指令
      return hn::MulAdd(m, vw, m);
    });
```

### 与 LayerNorm 的区别

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | 是 | 否 |
| 除以 | 标准差 | 均方根 |
| 计算量 | 需两遍扫描 | 只需一遍（算点积） |
| 效果 | — | 实验证明在大模型上与 LayerNorm 几乎无差 |

**为什么可以不减均值？** 在高维空间中，随机向量的各个分量近似零均值，减去均值带来的修正非常小，但去掉它省掉一趟完整的内存扫描。在大模型里，每层两次归一化 × 几十层，这个节省非常可观。

**`(1 + w)` 技巧**：Gemma 用 `(1 + w)` 而不是 `w` 作为缩放，这样 `w` 初始化为 0 时，归一化层是恒等变换，训练更稳定。代码中 `m + w*m` 被编译成一条 FMA（融合乘加）指令。

---

## 5. 注意力机制（Attention）

这是 Transformer 的心脏。Gemma 使用 **GQA（Grouped Query Attention）** 变体。

### 5.1 直觉

注意力回答的问题是："当前这个 token，应该对前文哪些 token 多看几眼？"

三个角色：
- **Q（Query）**：我在找什么信息
- **K（Key）**：每个 token 有什么标签可以被检索
- **V（Value）**：每个 token 实际携带的内容

计算过程（单个 head）：

```text
score_i = dot(Q, K_i) / √d            ← 当前 Q 与每个历史 K 的相似度
p_i = softmax(score_i)                ← 归一化成概率分布
output = Σ p_i * V_i                  ← 按权重加权求和
```

### 5.2 源码实现

源码：`gemma/attention.cc` → `GemmaAttention`

```cpp
void GemmaAttention(...) {
  ComputeQKV(num_tokens, layer_idx, layer, activations, qbatch, flags, env);
  if (flags & kAttentionUseOld) {
    DotSoftmaxWeightedSum(...);  // 朴素实现：完整 softmax
  } else {
    FlashAttention(...);         // 默认：分块在线 softmax
  }
  SumHeads(layer, activations, env);
}
```

三个阶段：

**阶段一：`ComputeQKV` —— 投影生成 Q、K、V**

```cpp
CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w1, ..., activations.q);
// qkv_einsum_w2 的输出直接写入 KV Cache 对应位置（通过 row_ptrs）
CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w2, ..., kv_rows);
```

Q 用一个矩阵算出；K、V 用另一个矩阵算出后**直接写入 KV Cache**，之后的 token 不再重算。对 K 还要做归一化（如果配置了）和 RoPE（见第 7 节）。

**阶段二：注意力得分与加权求和（`DotSoftmaxWeightedSum`）**

```cpp
QDotK(start_pos, last_pos, div_seq_len, q, k, att, ctx, worker);   // score = Q·K
MaybeLogitsSoftCap(att_cap, logits, ctx, worker);                  // tanh 软限幅
Softmax(logits, ctx, worker, 1.0f);                                // 归一化
WeightedSumV(start_pos, last_pos, div_seq_len, att, v, att_out, ...); // 加权 V
```

**阶段三：`SumHeads` —— 合并各头输出**

```cpp
CallMatMul(activations.att_out, layer.att_weights, ..., activations.att_sums);
```

### 5.3 GQA（Grouped Query Attention）

Gemma 不用标准 MHA（Multi-Head Attention），而是 GQA。源码中：

```cpp
// LayerConfig
uint32_t heads;      // query 头数，如 8
uint32_t kv_heads;   // key/value 头数，如 4
bool IsMHA() const { return heads == kv_heads; }  // Gemma 不再支持 MHA
```

多个 Q 头共享同一组 K/V 头。源码中体现为 `head / kHeadGroups` 的整数除法：

```cpp
const size_t kHeadGroups = layer_config.heads / layer_config.kv_heads;
const size_t head_offset = (head / kHeadGroups) * qkv_dim * 2;
```

**为什么这么做？** KV Cache 的大小正比于 `kv_heads`。MHA 如果 32 个头都要独立缓存 K/V，内存和带宽开销巨大。GQA 让 32 个 Q 头共享 8 个 K/V 头，KV Cache 直接缩小 4 倍，而质量损失很小（介于 MHA 和 MQA 之间）。

```text
MHA:  32 个 Q 头，32 个 KV 头 → KV Cache = 32 × seq_len × dim
GQA:  32 个 Q 头， 8 个 KV 头 → KV Cache =  8 × seq_len × dim （缩4倍）
MQA:  32 个 Q 头， 1 个 KV 头 → KV Cache =  1 × seq_len × dim （质量下降）
```

### 5.4 Soft Cap（软限幅）

Gemma 2 引入了对注意力得分的 tanh 限幅（`att_cap`）：

```text
score' = cap * tanh(score / cap)
```

当 score 远小于 cap 时 `tanh(x) ≈ x`，几乎不变；当 score 极大时被限制在 `(-cap, cap)` 内。这防止某些异常 token 的注意力得分独大，把其他 token 的概率压成 0，是稳定训练的手段。推理时也要精确复现。

### 5.5 滑动窗口注意力

Gemma 3 引入分层注意力窗口：一些层只看最近 N 个 token（local），一些层看全部（global）。

源码：`attention.cc` → `StartPos`

```cpp
size_t StartPos(size_t pos, const ModelConfig& config, size_t layer_idx) {
  const size_t att_window_size = config.attention_window_sizes[layer_idx];
  return pos - HWY_MIN(att_window_size - 1, pos);
}
```

如果窗口大小是 512，那么第 10000 个 token 只需要看第 9489 到第 10000 个 token 的 K/V，KV Cache 中更早的部分可以不读。这大幅降低长文本下的注意力计算量和带宽。

---

## 6. KV Cache：避免重复计算

### 问题：朴素方法的浪费

自回归生成时每步多一个 token。如果每步都对整个序列重新算所有层的 K 和 V，第 t 步的计算量是 O(t)，总计算量 O(n²)，而其中绝大部分是重复的——前面的 token 的 K/V 根本没变。

### 解法：只算新 token 的 K/V，历史缓存起来

源码：`gemma/kv_cache.h`

```cpp
struct KVCache {
  // 布局: [seq_len, layers × kv_heads × qkv_dim × 2]
  // 每行是一个时间步，行内按 (层, kv头, [K|V]) 排列
  MatStorageT<KV_t> kv_cache;
};
```

内存布局示意（2 层、2 个 KV 头、head_dim=4 的极简例子）：

```text
时间步 0:  [ L0_K_h0 ][ L0_V_h0 ][ L0_K_h1 ][ L0_V_h1 ][ L1_K_h0 ]...
时间步 1:  [ L0_K_h0 ][ L0_V_h0 ][ L0_K_h1 ][ L0_V_h1 ][ L1_K_h0 ]...
时间步 2:  [...]
...
```

**为什么按"时间步"为行、层为列？** 这样同一层、同一头、连续时间步的 K 或 V 在内存中是按列stride 排列的，注意力扫描时可以构造一个 `MatPtrT` 视图按行读取，对缓存预取友好。

### 写入时机

在 `ComputeQKV` 中，矩阵乘法的输出行指针直接指向 KV Cache 的对应位置：

```cpp
env.row_ptrs[0][interleaved_idx] = reinterpret_cast<uint8_t*>(
    qbatch.KV(qi).kv_cache.Row(cache_pos) + layer_idx * cache_layer_size);
kv_rows.AttachRowPtrs(env.row_ptrs[0].get());
CallMatMul(..., layer.qkv_einsum_w2, ..., kv_rows);
```

算完 K/V 就地写入缓存，零拷贝。

### 循环覆盖（wraparound）

当写入位置超过 `seq_len` 时，用取模回绕：

```cpp
const size_t cache_pos = activations.div_seq_len.Remainder(pos);
```

这实现了环形缓冲区。配合滑动窗口注意力，理论上只需要一个窗口大小的缓存，但实现中仍预分配完整 `seq_len` 大小。

### 计算量的改善

| | 无 KV Cache | 有 KV Cache |
||---|---|---|
| 第 t 步 | 重算 t 个 token 的 K/V | 只算 1 个新 token 的 K/V |
| 总量 | O(n²) | O(n) |
| 注意力得分 | 仍需 O(t) 个点积 | 仍需 O(t) 个点积（这步无法省） |

KV Cache 节省的是"投影计算"，注意力本身的 O(t) 点积依然要做——这正是 Flash Attention 要优化的部分。

---

## 7. RoPE：旋转位置编码

### 问题

Attention 本身对位置不敏感（把 token 顺序打乱，输出只是相应置换）。必须注入位置信息。

### 原理：把向量当作复数，按位置旋转

把 Q、K 向量的每两个维度看作一个二维平面上的点（复数）。位置为 m 的 token，把第 i 对维度旋转角度 `m × θ_i`：

```text
θ_i = 10000^(-2i/d)

[q'_2i  ]   [cos(mθ_i)  -sin(mθ_i)] [q_2i  ]
[q'_2i+1] = [sin(mθ_i)   cos(mθ_i)] [q_2i+1]
```

**为什么这样就编码了相对位置？** 两个旋转后的向量做点积时：

```text
dot(R(m)q, R(n)k) = dot(q, R(n-m)k)
```

旋转矩阵的性质保证点积只依赖于**角度差** `n - m`，即相对位置。低维度旋转快（捕捉细粒度相邻关系），高维度旋转慢（捕捉长距离关系），类似傅里叶基。

### 源码实现

源码：`ops/ops-inl.h` → `Rope` / `RopeAndMulBy`

Gemma 支持两种配对方式（`PostQKType`）：
- `Rope`：旋转 `(v_i, v_{i + d/2})` 这对维度（Gemma 风格）
- `HalfRope`：旋转 `(v_{2i}, v_{2i+1})` 相邻对（常见风格）

预先计算好每种频率的倒数（`inv_timescale`），推理时按位置索引查表即可：

```cpp
const float* inv_timescale = activations.inv_timescale.PackedScale1();
RopeAndMulBy(mul, qk, qkv_dim, inv_timescale, pos, ctx, worker);
```

**只作用于 Q 和 K，不作用于 V。** 因为 Q·K 的点积用于衡量"查询与键的匹配程度"，位置信息只需要在匹配阶段体现；V 是被加权取出的内容本身，不需要位置。

**Q 上乘 query_scale。** 源码中 `RopeAndMulBy(mul, ...)` 同时完成旋转和缩放（乘 `1/√d`），减少一趟内存读写。根据模型不同，scale 可能是 `1/√(model_dim/heads)` 或 `1/√qkv_dim`。

---

## 8. Flash Attention：分块在线 softmax

### 问题：朴素注意力的内存瓶颈

朴素实现（`DotSoftmaxWeightedSum`）：

```text
1. 算出所有 score = Q·K，写入 att 数组        ← 写一遍内存
2. 扫描 att 找 max
3. 逐个 exp(score - max)，写回 att            ← 再读写一遍
4. 算 sum，逐个除以 sum                        ← 再读写一遍
5. 加权求和 V
```

当 seq_len 很长时，att 数组本身（`seq_len × heads` 个 float）反复读写内存，带宽成为瓶颈。

### 核心思想：在线 softmax

softmax 通常需要先知道全局 max 和全局 sum。Flash Attention 用一个巧妙的递推公式，**边扫描边维护精确的 max 和 sum，从不需要一次看到全部数据**：

处理完一批新 score 后：

```text
m_new = max(m_old, x_new)              ← 新旧最大值取更大
d_new = e^(x_new - m_new) + d_old × e^(m_old - m_new)
       └── 新贡献──┘    └───── 修正旧累积 ─────┘
```

对输出向量的修正：

```text
out_new = out_old × (d_old / d_new) × e^(m_old - m_new)
        + v_new × e^(x_new - m_new) / d_new
```

数学上可以证明，按任意顺序处理、按任意大小分块，最终结果与先算完整 softmax 再加权完全一致（浮点误差除外）。

### 源码实现

标量版本（`flash_attention.cc` → `SingleFlashAttentionStep`）：

```cpp
void SingleFlashAttentionStep(float x, float cap, float& old_max,
                              float& old_d, const float* v,
                              const size_t v_cols, float* att_out) {
  if (cap > 0.0f) {
    x = cap * std::tanh(x / cap);       // 注意力软限幅
  }
  float m = std::max(x, old_max);        // 新 max
  x = std::exp(x - m);                   // 当前贡献
  float scale = old_d * std::exp(old_max - m);  // 修正因子
  old_d = x + scale;                     // 新分母
  float one_over_d = 1.0f / old_d;
  scale *= one_over_d;
  x *= one_over_d;
  MulByConst(scale, att_out, v_cols);         // 旧的 out 缩放
  MulByConstAndAdd(x, v, att_out, v_cols);    // 加上新的 v 贡献
}
```

### SIMD 分块版本

`TileFlashAttention` 把这个递推推广到向量级：

- Q 先做转置（`TransposeQ`），让同一 head 的多个 token 的同一维度排在连续内存
  中（对 SIMD 加载友好，且让"相同 KV"的查询聚在一起，缓存命中率高）
  - 同时处理 **NF 个 Q**（一个 SIMD 向量宽度的查询数）× **8 个 K 时间步** 的 tile
  - 用 `hn::MulAdd`（FMA）批量算 8 个点积
  - 用 `ElementwiseMaxOf8` / `ElementwiseSumOf8` 做向量级 max/sum 归约
  - 在线更新每列的 running max / running sum

```cpp
// 一个 tile 内同时算 8 个 K 时间步的 Q·K
for (size_t i = 0; i < k.Cols(); ++i) {
  VF q_vec = hn::Load(df, q);       // 同时加载 NF 个查询的这个维度
  sum0 = hn::MulAdd(q_vec, hn::Set(df, k_row[0][i]), sum0);
  sum1 = hn::MulAdd(q_vec, hn::Set(df, k_row[1][i]), sum1);
  ...
}
```

### 与朴素版对比

| | 朴素注意力 | Flash Attention |
||---|---|---|
| 内存 | O(seq_len × heads) 中间数组 | O(1) 额外内存（只有 m、d、out） |
| 读写 | 多趟完整扫描 | 一趟顺序扫描 |
| 数值稳定性 | 需先找 max 再 exp（两趟） | 在线维护 max，单趟 |
| Prefill 加速 | — | 显著（seq 越长越明显） |
| Decode 加速 | — | 也有收益（batch 时 tile 复用 K/V） |

---

## 9. 前馈网络（FFN）

### 做了什么

FFN 是逐 token 的"加工车间"——对每个 token 的向量独立地升维、非线性变换、降维。Gemma 使用**门控 GELU**（也叫 GeGLU）结构。

数学公式：

```text
FFN(x) = (GELU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

其中 ⊙ 是逐元素乘（Hadamard 积）
```

源码：`gemma/gemma-inl.h` → `FFWNoVit`

```cpp
#if GEMMA_FUSED_FFN  // 默认开启：融合版本
  const auto fused = [&](RowPtrsBF C1, IndexRange range_r, IndexRange range_c,
                         StridedViewBF C2, size_t worker) {
    Activation(layer_config.activation, C1, range_r, range_c, C2, ...);
  };
  MMOptions options;
  options.SetFunc(fused);
  CallTwoMatMul(activations.pre_ffw_rms_out, layer.gating_einsum_w1,
                layer.gating_einsum_w2, env, activations.C1, options);
#else  // 非融合版本
  CallMatMul(..., layer.gating_einsum_w1, ..., activations.C1);  // gate 分支
  CallMatMul(..., layer.gating_einsum_w2, ..., activations.C2);  // up 分支
  ActivationBatched(layer_config.activation, activations.C1, &activations.C2, ...);
#endif
  CallMatMul(activations.C1, layer.linear_w, ..., activations.ffw_out);  // 降维
```

### 为什么要有 FFN？

Attention 负责"token 之间的信息交换"，但它本质是加权平均，表达能力有限。FFN 对每个 token 独立做一次大容量的非线性变换，通常升维到 3~4 倍（`ff_hidden_dim`），让模型记住大量"模式"知识。研究表明 FFN 层存储了模型大部分的事实性知识，类似一个巨大的键值记忆库。

### 为什么用门控（Gating）？

```text
普通 FFN:   y = GELU(x W1) W2
门控 FFN:   y = (GELU(x W_gate) ⊙ x W_up) W2
```

门控让 `x W_up` 分支的每一个维度可以被 `GELU(x W_gate)` 独立"开关"控制。这相当于给每个神经元加了一个输入相关的阀门，只在需要时放行信息，比单纯的非线性变换表达力更强。GLM、Llama、Gemma 都采用此设计。

### GELU 近似

GELU 的精确定义涉及高斯分布 CDF，计算较贵。工程上用 tanh 近似或多项式近似。gemma.cpp 在 `ops/ops-inl.h` 中用 SIMD 友好的多项式实现，避免调用 `erf`。

### 融合优化（GEMMA_FUSED_FFN）

默认路径 `CallTwoMatMul` 把两个矩阵乘（gate 分支和 up 分支）和激活函数融合在一起：

- 权重只在缓存中加载一次
- C1 的每个 tile 算完立刻乘以 C2 的对应 tile，不落地中间结果
- 省掉一整块 `ff_hidden_dim` 大小数组的内存往返

这体现了 gemma.cpp "算法与算子协同设计"的典型思路：不是先写好算子再拼装，而是为了整体性能重塑边界。

### FFW 的参数量占大头

以 `model_dim=2048, ff_hidden_dim=8192` 为例：

```text
FFN 参数:  3 × 2048 × 8192 ≈ 50M （三个矩阵）
QKV 参数:  约 2048 × 2048 ≈ 4M
```

所以推理时间也主要消耗在 FFN 的两个大 GEMM 上。gemma.cpp 对这些矩阵做了最多的优化（分块、打包、自动调优）。

---

## 10. 残差连接

源码：`gemma/gemma-inl.h` → `ResidualConnection`

```cpp
static void ResidualConnection(MatPtrT<BF16>& other, MatPtrT<float>& x,
                               const LayerWeightsPtrs& layer,
                               bool is_attention, ThreadingContext& ctx) {
  // ResidualType::Add
  AddFromBatched(other, x, ctx);
}
```

两个子层各有一次残差：

```text
x = x + Attention_out
x = x + FFN_out
```

### 原理

残差连接是 ResNet（2015）提出的思想：与其让 `f(x)` 直接作为输出，不如学"增量" `x + f(x)`。

对推理的影响：
- 每层只需学习对已有信息的"修改量"，不需要从头重建
    - 数值上保持了信息流的连续性——即使某一层输出接近零，信息也不会丢失
    - 训练时梯度有直达路径，这是能训练百层网络的关键

### 实现细节

`other` 是 BF16 类型（注意力/FFN 的输出已经降精度存储），`x` 是 float。相加时自动做精度提升。输出向量 `x`（残差流）保持 float 精度，避免多层累加的误差。

---

## 11. 输出层与采样

### 11.1 最终归一化 + logits

所有层跑完后，最后再做一次 RMSNorm，然后乘以 Embedding 表的转置（tied weights），得到词表中每个 token 的分数（logits）：

源码：`gemma/gemma.cc` → `SampleAndStream`

```cpp
RMSNormBatched(activations.x, weights.final_norm_scale, activations.x_bf, env.ctx);
CallMatMul(activations.x_bf, weights.embedder_input_embedding, ..., activations.logits);
```

对 logits 做最终的 soft cap（`final_cap`，与注意力里的 `att_cap` 类似）：

```cpp
MaybeLogitsSoftCapBatched(config.final_cap, activations.logits, non_eos, env.ctx);
```

### 11.2 采样策略

采样把 logits 变成一个具体的 token id。gemma.cpp 实现了：

| 策略 | 行为 | 源码函数 |
||---|---|---|
| Top-1（贪心） | 永远选概率最大的 token | `Top1OfSoftmax` |
| Top-K | 在概率最高的 K 个中按概率随机采样 | `FusedSoftmaxAndSampleTopK` |
| 温度 | 控制 softmax 的"陡峭程度" | `Softmax` 的 `temperature` 参数 |

**Top-1 的融合优化**（源码注释明确说明动机）：

```cpp
// Returns argmax of softmax and its probability. This overwrites `logits`,
// but not with normalized probabilities. Only equivalent to `Softmax` +
// `sample_func` if `kTopK` == 1. This is worthwhile because `logits.size()` is
// typically `kVocabSize == 256K`, and this avoids writing and then scanning
// again for the max.
static TokenAndProb Top1OfSoftmax(Logits logits) {
  const TokenAndProb argmax = ArgmaxAndMax(logits);  // 一趟找 max
  // 直接算 exp(logit_argmax - max) / sum_exp，不需要归一化整个数组
  ...
}
```

当 `top_k == 1` 时，不需要知道每个 token 的概率，只需要知道 argmax 和它的概率。所以跳过全量 softmax，省掉一次 25 万元素数组的读写。

**Top-K 采样流程**：

```cpp
std::vector<TokenAndProb> token_logits = TopK(logits, k, accept_token);
Softmax(Logits(topk_logits), ctx, worker, temperature);  // 只对 K 个做 softmax
std::discrete_distribution<int> distribution(topk_logits, ...);
int sampled = distribution(gen);
```

**温度的原理**：

```text
p_i = exp(logit_i / T) / Σ exp(logit_j / T)
```

- T → 0：分布退化为 argmax（完全确定性）
    - T = 1：模型本来的分布
    - T → 大：分布趋于均匀（完全随机）

### 11.3 EOS 判断与停止

生成到 EOS（end of sequence）token 时，对应的查询被标记结束：

```cpp
if (config.IsEOS(token)) non_eos.Clear(qi);
```

`non_eos` 是一个 `BitSet4096`，每个 batch 查询占一位。当所有位都清零，或达到 `max_generated_tokens`，解码循环停止。

---

## 12. Prefill 与 Decode 两阶段

### Prefill：批量填 Cache

源码：`gemma/gemma.cc` → `PrefillTBatch`

```text
prompt 有 1000 个 token
  │
  ├── 第 1 批（如 256 个 token 同时进模型）→ 算 QKV → 写 KV Cache
  ├── 第 2 批 → ...
  ├── ...直到倒数第 2 个 token
  │
  └── 最后 1 个 token 留给 Decode 作为起点
```

**为什么可以批量？** Prompt 的所有 token 是已知的，可以并行处理。批量化提高了 GEMM 的算术强度（每次加载权重矩阵后用于多个 token），对 memory-bound 的 CPU 推理收益巨大。

**为什么不算最后一个 token？** 最后一个 token 之后要生成第一个新 token，需要把最后一个 token 也送入模型。如果 Prefill 把它算了，Decode 再算一遍会重复写入 KV Cache 同一位置。所以 Prefill 到倒数第二个，留最后一个作为 Decode 的第一步输入。

### Decode：逐 token 生成

源码：`gemma/gemma.cc` → `GenerateT` 主循环

```cpp
for (size_t gen = 0; gen < max_gen_steps && non_eos.Any(); ++gen) {
  Transformer(config, runtime_config, weights, activations, qbatch, env);
  SampleAndStream(config, runtime_config, weights, sample_token, ...);
}
```

每步：
1. 取上一步采样的 token
2. 查 Embedding
3. 过所有 Transformer 层（只算当前 token 的 Q，但扫描全部历史 K/V）
4. 算 logits、采样、流式输出

**为什么 Decode 不能批量（对单个请求）？** 第 t+1 个 token 依赖第 t 个 token 的输出，存在串行依赖。但可以跨请求批量（`GenerateBatchT`）：让多个用户各出一个 token 组成一个 batch，分摊权重加载。

### 两阶段对比

| | Prefill | Decode |
|---|---|---|
| 每次 token 数 | 多个（如 128~512） | 每请求 1 个 |
| GEMM 形状 | [batch, dim] × [dim, dim] | [1, dim] × [dim, dim] |
| 瓶颈 | 计算（compute-bound） | 带宽（memory-bound，权重太大装不进缓存） |
| 注意力 | 可用 Flash Attention 分块 | 逐 token 扫 KV Cache |
| 加速手段 | 增大 batch 提高复用率 | 多请求合 batch、量化压缩权重 |

---

## 13. 技术原理速查表

| 技术 | 一句话原理 | 解决什么问题 |
||---|---|---|
| Embedding | 离散 token → 稠密向量 | 把符号变成可计算的连续数学对象 |
| 共享 Embedding | 输入查表和输出投影用同一矩阵 | 节省 vocab × dim 的巨大参数 |
| RMSNorm | 除以均方根 | 稳定数值，比 LayerNorm 快 |
| Attention | Q·K softmax 加权 V | token 间信息交换 |
| GQA | 多 Q 头共享 K/V 头 | 缩小 KV Cache |
| KV Cache | 缓存历史 K/V | 避免重复投影计算 |
| RoPE | 按位置旋转向量对 | 注入相对位置信息 |
| SoftCap | tanh 限幅 | 防止注意力过度集中 |
| 滑动窗口 | 只看最近 N 个 token | 长文本降计算 |
| Flash Attention | 在线 softmax 分块 | 减少内存读写 |
| 门控 FFN | 两分支逐元素相乘 | 增强非线性表达 |
| 残差连接 | x + f(x) | 保持信息流，训练深网络 |
| Top-K + 温度 | 截断+缩放后采样 | 控制生成多样性与质量 |
| Prefill/Decode 分离 | 批量已知，串行未知 | 匹配不同阶段的负载特征 |
| SIMD | 一条指令处理多个数据 | 提高 CPU 吞吐 |
| 量化 | 低精度存储权重 | 减少内存带宽和占用 |

---

## 14. 模拟面试题

### 基础题

**Q1：Transformer 推理时为什么要用 KV Cache？它节省了哪部分计算？**

> **答**：自回归生成时，历史 token 的 K 和 V 不随新 token 加入而改变。若不缓存，每生成一个 token 就要对全部历史重新算一遍所有层的 K/V 投影，总计算量 O(n²)。KV Cache 把每个 token 每层的 K/V 存起来，新 token 只需算自己的 K/V 一次，总投影计算量降到 O(n)。注意：它不节省注意力本身的 Q·K 点积，那部分仍是每步 O(t)，这也是 Flash Attention 存在的意义。

---

**Q2：RMSNorm 和 LayerNorm 的区别是什么？为什么 Gemma 选 RMSNorm？**

> **答**：LayerNorm 要减去均值再除以标准差；RMSNorm 不减均值，直接除以均方根 `sqrt(mean(x²))`。数学上 LayerNorm 需要"和"与"平方和"两个统计量（且减均值时引入耦合），RMSNorm 只需一个点积（自身平方和）。在高维情况下各分量近似零均值，减均值带来的修正可忽略。Gemma 是部署导向的模型，用更少的计算达到几乎相同的效果。

---

**Q3：为什么 Attention 要除以 √d？**

> **答**：如果 Q 和 K 的每个分量独立同分布、方差为 1，那么长度为 d 的点积的方差是 d。当 d 较大（如 256）时，点积的数值方差很大，softmax 会过度尖锐（某些位置概率接近 1，其余接近 0），导致梯度消失。除以 √d 把方差拉回 1，让 softmax 处于对训练友好的敏感区间。gemma.cpp 中这通过 `query_scale` 预先乘在 Q 上实现。

---

**Q4：RoPE 为什么只作用于 Q 和 K，不作用于 V？**

> **答**：位置信息的目的是让"查询与键的匹配分数"能反映相对距离。数学上，对 Q 和 K 分别做旋转 R(m)q 和 R(n)k，其点积等于 dot(q, R(n-m)k)，只依赖相对位置。V 是被注意力权重加权取出的"内容"本身，它不参与点积匹配，加位置没有意义。

---

### 进阶题

**Q5：解释 Flash Attention 的"在线 softmax"数学原理，为什么可以分块处理而不需要先看到全部数据？**

> **答**：朴素 softmax 需要全局 max 和全局 sum。Flash Attention 维护 running max m 和 running sum d。处理新的 score x 时：新 max `m' = max(m, x)`；新分母 `d' = e^(x-m') + d × e^(m-m')`。关键是旧贡献乘以 `e^(m-m')` 这个修正因子，把旧的部分从旧 max 的坐标系"搬"到新 max 的坐标系。对输出同理：`out' = out × (d/d') × e^(m-m') + v × e^(x-m')/d'`。这个递推保证任意分组、任意顺序处理，结果都和全局 softmax 数学等价。

---

**Q6：GQA（Grouped Query Attention）与 MHA、MQA 的区别？为什么 Gemma 用 GQA？**

> **答**：MHA 中每个 Q 头有独立的 K/V 头，KV Cache 大小正比于总头数。MQA 所有 Q 头共享 1 个 K/V 头，Cache 最小但质量下降。GQA 折中：若干 Q 头（如 4 个）共享 1 组 K/V 头。Cache 相比 MHA 缩小了"分组数"倍，质量接近 MHA。对 CPU 推理尤其重要，因为带宽是第一瓶颈，KV Cache 缩小直接提升解码速度。

---

**Q7：为什么 Prefill 和 Decode 要用不同的策略？各自的瓶颈是什么？**

> **答**：Prefill 时所有 prompt token 已知，可以批量送入。GEMM 形状是 [batch, dim] × [dim, dim]，每次加载权重矩阵可以服务多个 token，算术强度高，瓶颈在计算（compute-bound）。Decode 时每请求每步只有 1 个 token，GEMM 退化为向量×矩阵，算术强度极低，瓶颈在把权重从内存搬到 CPU 的带宽（memory-bound）。所以 Prefill 追求 batch 和 Flash Attention，Decode 追求量化（减少字节数）和多请求合 batch。

---

**Q8：gemma.cpp 中的 `Top1OfSoftmax` 为什么可以不做全量 softmax？**

> **答**：如果只需要 argmax（top_k=1），最终输出的 token 就是 logits 最大的那个，softmax 是单调变换不改变排序。唯一还想要的是该 token 的概率值，这只需要 `exp(l_max - max) / Σ exp(l_i - max) = 1 / Σ exp(l_i - l_max)`。源码在找 argmax 的同一趟中顺便累计 sum_exp，跳过了对 25 万词表做完整归一化的读写。这是典型的"算法层知道下游要什么，就跳过不需要的中间结果"的融合优化。

---

**Q9：滑动窗口注意力在 KV Cache 中是怎么实现的？**

> **答**：`StartPos(pos, config, layer_idx)` 根据该层的窗口大小 `att_window_size` 计算扫描起点 `pos - min(window-1, pos)`。如果 pos 小于窗口，从 0 开始；否则只扫描最近 window 个位置。KV Cache 本身仍是完整的 [seq_len, layers × kv_heads × dim × 2] 矩阵，但注意力循环只从 `start_pos` 到 `pos`，不去读更早的行。这样每层注意力计算量从 O(seq_len) 变为 O(window)。

---

### 开放设计题

**Q10：如果让你在 gemma.cpp 中新增一种采样策略（比如 Top-P），你会怎么设计？**

> **参考思路**：
> 1. 在 `RuntimeConfig` 中加 `float top_p` 参数（默认 1.0 表示关闭）
> 2. 在 `ops/ops-inl.h` 中实现 `TopP` 算子：先对 logits 排序（或用部分选择算法找到截断点），取累计概率刚超过 p 的前缀集合，再在这个集合中按概率采样
> 3. 修改 `ChooseSampleFunc`，当 `top_p < 1.0` 时返回新的采样 lambda
> 4. 性能考虑：词表 25 万，全排序 O(v log v) 太贵；可以先找 max，再做 bucket 或部分排序（`std::nth_element`）把候选缩到较小集合再精确排序
> 5. 与 Top-K 的关系：可以先做 K=1000 粗截断，再做 Top-P 精截断

---

**Q11：为什么 gemma.cpp 要把两个 FFN 矩阵乘和激活函数融合在一起（`GEMMA_FUSED_FFN`）？**

> **参考思路**：gate 和 up 两个分支的输入是同一个 `pre_ffw_rms_out`。融合后：① 输入只从内存加载一次；② `C1 = GELU(A×W1) ⊙ (A×W2)` 中的 ⊙ 在 tile 级别立刻完成，中间不落地一个 `ff_hidden_dim` 大小的数组；③ 权重 tile 留在 L1/L2 的时间更紧凑，缓存命中率更高。对 memory-bound 的 CPU 推理，"少一次内存往返"往往比"少几条计算指令"收益更大。

---

**Q12：如果推理速度很慢，你会从哪几个方向排查和优化？**

> **参考思路**（分层排查）：
> - **模型层**：是否可以用 SFP/NUQ 量化权重减少字节数？GQA 的 kv_heads 是否已启用？
> - **调度层**：Prefill batch 是否够大？多请求 Decode 是否合并 batch？
> - **算子层**：GEMM 分块参数是否针对当前 CPU 自动调优过？SIMD 目标是否选择正确（AVX2 vs AVX-512）？
> - **注意力层**：长文本是否用了 Flash Attention？滑动窗口是否生效？
> - **系统层**：线程是否绑定到正确的核？NUMA 内存是否就近分配？磁盘加载是否用了 mmap？
> - **测量**：先用内置 profiler 定位热点，不要凭感觉优化

---

**Q13：KV Cache 的内存布局为什么设计成 [seq_len, layers × kv_heads × dim × 2]（时间步为行）而不是 [layers, kv_heads, seq_len, dim]（层为最外维）？**

> **参考思路**：
> - 时间步为行意味着写入 KV Cache 时是顺序追加一行，局部性好
> - 同一个 token 的所有层的 K/V 写入在时间上连续发生，如果布局以层为最外维，写入会跳到相距很远的不同内存区域
> - 注意力读取时通过 `MatPtrT` 视图加 stride 可以灵活地把某一层某一头的 K 或 V 视作一个 [seq_len, qkv_dim] 的矩阵，顺序扫描
> - 这是写入友好与读取可灵活寻址之间的权衡

---

**Q14：为什么 Attention 中的 Softmax 对数值稳定性很重要？gemma.cpp 做了哪些保障？**

> **参考思路**：exp(x) 当 x > 88（float32）会溢出为 inf，x < -104 会下溢为 0。softmax 必须先减去 max 再 exp。gemma.cpp 在朴素路径（`Softmax`）中先扫描找 max，再减 max 做 exp，最后除以 sum；在 Flash Attention 路径中用 running max 在线更新。另外 soft cap（tanh）也在 exp 之前限制了输入范围。多条防线确保不产生 inf/nan。

---

## 附：关键文件索引

| 文件 | 内容 |
|---|---|
| `gemma/gemma.cc` | 整体调度、Embedding、Prefill/Decode、采样 |
| `gemma/gemma-inl.h` | FFN、残差、PostNorm 的 SIMD 实现 |
| `gemma/attention.cc` | QKV 投影、朴素注意力、GQA head 映射 |
| `gemma/flash_attention.cc` | Flash Attention 的标量和 SIMD tile 版本 |
| `gemma/kv_cache.h/.cc` | KV Cache 的布局和分配 |
| `gemma/activations.h` | 中间激活的分配和布局 |
| `gemma/configs.h` | 模型结构（heads、kv_heads、窗口大小等） |
| `ops/ops-inl.h` | RMSNorm、RoPE、Softmax、采样等基础算子 |
