---
title: Chapter 07 · 做指令微调
order: 7
chapter: 7
summary: 将基础模型调教成AI助手！通过指令微调让模型学会遵循人类指令，理解监督式微调流程和评估方法。
status: planned
tags:
  - instruction-tuning
  - sft
  - prompts
---

## 本章目标

理解 assistant-style 模型是怎么从 base model 演化来的。

## 核心任务

- 设计 prompt template
- 整理 instruction-following 数据
- 训练 supervised fine-tuning 流程
- 做最小评估

## 建议的 C# 模块

- `InstructionSample`
- `PromptTemplate`
- `SftDataset`
- `InstructionTrainer`
- `InstructionEvaluator`

## 本章边界

这一阶段先把 SFT 做清楚，不急着扩展到 RLHF、DPO 或更复杂的偏好优化。

## 完成本章后的状态

你会拥有一个相对完整的“从文本数据到预训练再到两类微调”的最小闭环。
