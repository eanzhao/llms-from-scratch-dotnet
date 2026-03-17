---
title: Chapter 06 · 做文本分类微调
order: 6
chapter: 6
summary: 教你的 GPT 做分类任务！通过微调将语言模型适配为分类器，理解任务头设计、标注数据准备和评估指标。
status: planned
tags:
  - finetuning
  - classification
  - evaluation
---

## 本章目标

理解如何把一个语言模型适配成分类器。

## 你会接触到的部分

- 分类数据集
- 标签映射
- 分类头
- 训练与验证循环
- accuracy、precision、recall 等指标

## 建议实现

- `ClassificationDataset`
- `ClassificationHead`
- `ClassificationTrainer`
- `ClassificationMetrics`

## 这一章真正重要的点

不是“把准确率做多高”，而是理解：

- 为什么 base model 可以复用
- 为什么不同任务只需要少量结构改动
- 分类任务和生成任务在目标函数上有什么差别
