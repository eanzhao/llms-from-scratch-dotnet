---
title: Chapter 04 · 从零实现 GPT
order: 4
chapter: 4
summary: 组装完整的 GPT 模型！将 embedding、transformer block、残差连接、layer norm 和输出头组合起来，实现前向传播。
status: planned
tags:
  - gpt
  - transformer-block
  - forward-pass
---

## 本章目标

把前几章的零件装配成一个完整模型。

## 典型组件

- token embedding
- positional embedding
- transformer block
- feed-forward network
- layer normalization
- residual connection
- language modeling head

## 在 C# 中建议先做到的程度

- 先支持前向传播
- 先用小维度模型
- 先验证 logits 形状正确
- 先不要急着把所有参数初始化策略做得很复杂

## 推荐交付物

- `GptConfig`
- `TransformerBlock`
- `GptModel`
- 一个固定输入下的 forward demo

## 这一章最容易出错的地方

- 张量维度不一致
- causal mask 位置错误
- residual 和 layer norm 顺序写反
