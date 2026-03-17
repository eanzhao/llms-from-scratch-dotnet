---
title: Chapter 03 · 编码注意力机制
order: 3
chapter: 3
summary: 实现 Transformer 的核心——注意力机制！掌握缩放点积注意力、因果掩码和多头注意力，理解为什么这是 LLM 的"灵魂"。
status: planned
tags:
  - attention
  - transformer
  - matrix
---

## 本章目标

从“知道注意力是什么”进入“能把注意力写出来”。

## 建议的 C# 模块

- `MatrixOps`
- `AttentionMask`
- `SelfAttention`
- `CausalSelfAttention`
- `MultiHeadAttention`

## 关键问题

### 为什么要缩放

点积会随着维度增大而变大，softmax 会更尖锐，所以需要缩放。

### 为什么要 mask

GPT 不能偷看未来 token，所以要用 causal mask。

### 为什么要 multi-head

因为不同 head 可以学到不同的关系模式，而不是把所有关系都压进一组权重里。

## 建议验证方式

- 用极小矩阵手算一遍 attention score
- 用固定输入验证 mask 后的 softmax 输出
- 打印每一步 tensor shape

## 本章完成标志

你已经能单独跑通 self-attention，并解释每一步数学含义。
