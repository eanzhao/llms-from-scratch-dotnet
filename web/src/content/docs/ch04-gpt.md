---
title: Chapter 04 · 从零实现 GPT
order: 4
chapter: 4
summary: 组装完整的 GPT 模型！将 embedding、transformer block、残差连接、layer norm 和输出头组合起来，实现前向传播。
status: done
tags:
  - gpt
  - transformer-block
  - forward-pass
---

## 本章目标

把前几章的零件装配成一个完整的 GPT 模型，实现前向传播和贪心文本生成。

## C# 实现参考

| C# 类 | 文件路径 | 对应 Python | 说明 |
|--------|---------|------------|------|
| `GptConfig` | `Chapter04.Gpt/GptConfig.cs` | `GPT_CONFIG_124M` | 模型配置（C# record） |
| `LayerNorm` | `Chapter04.Gpt/LayerNorm.cs` | `LayerNorm` | 层归一化 |
| `Gelu` | `Chapter04.Gpt/Gelu.cs` | `GELU` | GELU 激活函数 |
| `FeedForward` | `Chapter04.Gpt/FeedForward.cs` | `FeedForward` | 前馈网络 |
| `TransformerBlock` | `Chapter04.Gpt/TransformerBlock.cs` | `TransformerBlock` | Transformer 块 |
| `GptModel` | `Chapter04.Gpt/GptModel.cs` | `GPTModel` | 完整 GPT 模型 |
| `TextGenerator` | `Chapter04.Gpt/TextGenerator.cs` | `generate_text_simple` | 贪心文本生成 |

### 预设配置

`GptConfig` 提供多种预设：

| 预设 | vocab | emb_dim | layers | heads | 用途 |
|------|-------|---------|--------|-------|------|
| `Gpt2Small` | 50257 | 768 | 12 | 12 | GPT-2 124M 参数 |
| `Gpt2Medium` | 50257 | 1024 | 24 | 16 | GPT-2 355M 参数 |
| `SmallTraining` | 100 | 64 | 2 | 2 | 小规模训练实验 |
| `Tiny` | 100 | 32 | 1 | 1 | 最小可运行模型 |

## 核心算法

### 1. GPT 模型架构

```
输入 Token IDs: [batch, seq_len]
       │
       ▼
 ┌─────────────────┐
 │  Token Embedding │  token_id → [emb_dim] 向量
 │  + Pos Embedding │  position → [emb_dim] 向量
 │  + Dropout       │
 └────────┬────────┘
          │ [batch, seq_len, emb_dim]
          ▼
 ┌─────────────────┐
 │ TransformerBlock │ ×N 层
 │  (详见下方)      │
 └────────┬────────┘
          │ [batch, seq_len, emb_dim]
          ▼
 ┌─────────────────┐
 │  Final LayerNorm │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Output Head     │  Linear(emb_dim → vocab_size)
 └────────┬────────┘
          │
          ▼
 Logits: [batch, seq_len, vocab_size]
```

### 2. TransformerBlock (Pre-Norm 架构)

GPT-2 使用 **Pre-Norm** 变体（LayerNorm 在注意力/FFN 之前）：

```
输入 x
  │
  ├──────────────────┐
  │                  │ (残差连接)
  ▼                  │
 LayerNorm           │
  ▼                  │
 MultiHeadAttention  │
  ▼                  │
 Dropout             │
  ▼                  │
 + ←─────────────────┘
  │
  ├──────────────────┐
  │                  │ (残差连接)
  ▼                  │
 LayerNorm           │
  ▼                  │
 FeedForward         │
  ▼                  │
 Dropout             │
  ▼                  │
 + ←─────────────────┘
  │
  ▼
 输出
```

**为什么 Pre-Norm**：梯度直接通过残差连接流动，训练更稳定。原始 Transformer 用 Post-Norm（先计算再归一化），GPT-2 及之后的模型多用 Pre-Norm。

### 3. 层归一化 (LayerNorm)

对每个样本的每个位置独立归一化：

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) × scale + shift
```

- `mean` 和 `var` 沿最后一个维度（emb_dim）计算
- `scale` 和 `shift` 是可学习参数
- `ε = 1e-5` 防止除零

### 4. GELU 激活函数

```
GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

GELU 是 GPT 使用的激活函数。相比 ReLU：
- 在零点附近更平滑（不是硬截断）
- 有门控效应：输入越大，"通过率"越高
- 负值不完全为零，保留了一些梯度信息

### 5. 前馈网络 (FeedForward)

```
FeedForward(x) = Linear_2(GELU(Linear_1(x)))

Linear_1: [emb_dim] → [4 × emb_dim]   (升维)
Linear_2: [4 × emb_dim] → [emb_dim]   (降维)
```

4倍升维是 GPT 的设计选择，给模型更大的中间表示空间进行非线性变换。

### 6. 贪心文本生成

```
while seq_len < max_new_tokens:
    1. 裁剪上下文到 context_length
    2. 前向传播得到 logits [batch, seq, vocab]
    3. 取最后一个位置的 logits [batch, vocab]
    4. argmax 选择概率最高的 token
    5. 拼接到序列末尾
```

## 验证方式

- 小模型前向传播输出形状：`[batch, seq_len, vocab_size]`
- 验证残差连接：如果注意力/FFN 全零，输出 = 输入
- 生成文本：随机权重下输出是乱码（正确），训练后逐渐有意义
- Playground 中已验证：Tiny 配置 `[2, 4, 100]` 输出形状正确

## 常见问题

### 张量维度不一致
确保 `num_heads × head_dim == emb_dim`。GptConfig 中用 `emb_dim / num_heads` 自动计算。

### 残差和 LayerNorm 顺序
GPT-2 是 Pre-Norm：`x + Attn(LayerNorm(x))`，不是 Post-Norm：`LayerNorm(x + Attn(x))`。

### 因果掩码
掩码在 MultiHeadAttention 内部处理，TransformerBlock 不需要额外传递。
