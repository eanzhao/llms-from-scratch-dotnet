---
title: Chapter 03 · 编码注意力机制
order: 3
chapter: 3
summary: 实现 Transformer 的核心——注意力机制！掌握缩放点积注意力、因果掩码和多头注意力，理解为什么这是 LLM 的"灵魂"。
status: done
tags:
  - attention
  - transformer
  - matrix
---

## 本章目标

从"知道注意力是什么"进入"能把注意力写出来"。

## C# 实现参考

| C# 类 | 文件路径 | 对应 Python | 说明 |
|--------|---------|------------|------|
| `SelfAttention` | [`Chapter03.Attention/SelfAttention.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter03.Attention/LlmsFromScratch.DotNet.Chapter03.Attention/SelfAttention.cs) | `SelfAttention_v2` | 单头自注意力（教学用） |
| `CausalSelfAttention` | [`Chapter03.Attention/CausalSelfAttention.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter03.Attention/LlmsFromScratch.DotNet.Chapter03.Attention/CausalSelfAttention.cs) | `CausalAttention` | 带因果掩码的注意力 |
| `MultiHeadAttention` | [`Chapter03.Attention/MultiHeadAttention.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter03.Attention/LlmsFromScratch.DotNet.Chapter03.Attention/MultiHeadAttention.cs) | `MultiHeadAttention` | 多头注意力完整实现 |

## 核心算法

### 1. 缩放点积注意力 (Scaled Dot-Product Attention)

这是注意力机制的数学核心：

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

逐步分解：

```
输入 X: [batch, seq_len, emb_dim]

1. 线性投影:
   Q = X × W_q    [batch, seq_len, head_dim]
   K = X × W_k    [batch, seq_len, head_dim]
   V = X × W_v    [batch, seq_len, head_dim]

2. 注意力分数:
   scores = Q × K^T / √head_dim    [batch, seq_len, seq_len]

3. 因果掩码 (可选):
   scores[i][j] = -∞  where j > i   (不能看到未来的 token)

4. 归一化:
   weights = softmax(scores)         [batch, seq_len, seq_len]

5. 加权求和:
   output = weights × V              [batch, seq_len, head_dim]
```

### 2. 为什么要缩放

点积会随着维度 `d_k` 增大而变大。假设 Q 和 K 的元素都是均值 0、方差 1 的随机变量：

- `Q · K` 的方差 = `d_k`
- 不缩放时，大维度的点积会很大，softmax 输出趋近 one-hot（梯度消失）
- 除以 `√d_k` 使方差回到 1，softmax 分布更平滑

### 3. 因果掩码

GPT 是自回归模型，生成第 `i` 个 token 时只能看到 `0..i-1`：

```
掩码矩阵 (4×4 序列):
1  0  0  0      ← token 0 只看自己
1  1  0  0      ← token 1 看 0,1
1  1  1  0      ← token 2 看 0,1,2
1  1  1  1      ← token 3 看 0,1,2,3
```

实现方式：用上三角矩阵将未来位置设为 `-∞`，softmax 后这些位置权重为 0。

C# 实现使用 `Tensor.Triu` 生成上三角掩码 + `TensorOps.MaskedFill` 填充 `-∞`。

### 4. 多头注意力

将 `emb_dim` 拆分为 `num_heads × head_dim`，每个 head 独立计算注意力：

```
输入: [batch, seq_len, emb_dim]

1. Q/K/V 投影: [batch, seq_len, emb_dim]
2. reshape:    [batch, seq_len, num_heads, head_dim]
3. transpose:  [batch, num_heads, seq_len, head_dim]
4. 每个 head 独立做 scaled dot-product attention
5. transpose:  [batch, seq_len, num_heads, head_dim]
6. reshape:    [batch, seq_len, emb_dim]   (concat all heads)
7. 输出投影:   [batch, seq_len, emb_dim]
```

**为什么多头更好**：不同 head 可以学到不同的关系模式（语法关系、语义关系、位置关系等），比单个大 head 表达能力更强。

## 关键维度变换

以 `batch=2, seq=4, emb=8, heads=2, head_dim=4` 为例：

```
输入          [2, 4, 8]
Q/K/V 投影后  [2, 4, 8]
reshape      [2, 4, 2, 4]
transpose    [2, 2, 4, 4]   ← 每个 head 在独立的维度
attention    [2, 2, 4, 4]
transpose    [2, 4, 2, 4]
reshape      [2, 4, 8]      ← concat 回原始维度
out_proj     [2, 4, 8]
```

## 验证方式

- 用 2×2 极小矩阵手算 attention score，与代码输出对比
- 验证因果掩码：softmax 输出的上三角应为 0
- 检查多头注意力的输入输出形状一致：`[batch, seq, emb]`
- 确认 `num_heads × head_dim == emb_dim`

## 补充知识

### 注意力的直觉理解

注意力像是一个**软性查找表**：
- Query = "我在找什么"
- Key = "我有什么"
- Value = "我的实际内容"

Q 和 K 的点积衡量"匹配度"，softmax 归一化后作为权重对 V 加权求和。每个 token 根据与其他 token 的相关性来汇聚信息。

### Attention Is All You Need

2017 年 Vaswani 等人提出 Transformer 架构，用纯注意力机制取代了 RNN/LSTM 的循环结构。关键优势是：
1. **并行计算**：所有位置同时计算，不需要逐步传播
2. **长距离依赖**：任意两个位置之间只隔一层注意力
3. **可解释性**：注意力权重可视化可以帮助理解模型关注了什么
