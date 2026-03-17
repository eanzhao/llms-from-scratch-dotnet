---
title: Chapter 05 · 在无标注数据上预训练
order: 5
chapter: 5
summary: 让模型真正”学习”起来！实现损失函数、训练循环、文本采样和模型检查点，在小规模语料上进行预训练。
status: planned
tags:
  - training
  - loss
  - generation
---

## 本章目标

让 GPT 在小规模无标注语料上真正开始学习。

## 建议拆分

- `Trainer`
- `LossFunctions`
- `CheckpointStore`
- `TextGenerator`
- `TrainingMetrics`

## 最小闭环

1. 加载文本数据
2. 生成 batch
3. 前向传播
4. 计算 next-token loss
5. 反向更新参数
6. 周期性采样文本

## C# 版本先关注什么

- 训练循环清晰
- 日志可读
- checkpoint 可恢复
- 小语料上能观察到 loss 下降

## 暂时不用太早追求什么

- 分布式训练
- 大规模语料工程
- 复杂实验平台
