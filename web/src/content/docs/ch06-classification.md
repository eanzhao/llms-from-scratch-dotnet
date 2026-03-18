---
title: Chapter 06 · 做文本分类微调
order: 6
chapter: 6
summary: 教你的 GPT 做分类任务！通过微调将语言模型适配为分类器，理解任务头设计、参数冻结和评估指标。
status: done
tags:
  - finetuning
  - classification
  - evaluation
---

## 本章目标

理解如何把一个预训练语言模型适配成分类器——用最少的改动完成新任务。

## C# 实现参考

| C# 类 | 文件路径 | 对应 Python | 说明 |
|--------|---------|------------|------|
| `SpamDataset` | [`Chapter06.Classification/SpamDataset.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter06.Classification/LlmsFromScratch.DotNet.Chapter06.Classification/SpamDataset.cs) | `SpamDataset` | 垃圾邮件分类数据集 |
| `LayerFreezer` | [`Chapter06.Classification/LayerFreezer.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter06.Classification/LlmsFromScratch.DotNet.Chapter06.Classification/LayerFreezer.cs) | 手动冻结代码 | 参数冻结/解冻工具 |
| `ClassificationTrainer` | [`Chapter06.Classification/ClassificationTrainer.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter06.Classification/LlmsFromScratch.DotNet.Chapter06.Classification/ClassificationTrainer.cs) | `train_classifier_simple` | 分类训练循环 |
| `ClassificationMetrics` | [`Chapter06.Classification/ClassificationMetrics.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter06.Classification/LlmsFromScratch.DotNet.Chapter06.Classification/ClassificationMetrics.cs) | `calc_accuracy_loader` | 准确率计算 |

## 核心概念

### 1. 从生成模型到分类器

GPT 是生成模型，输出 `[batch, seq_len, vocab_size]`。改造为分类器：

```
原始 GPT:
  输入 → Transformer → [batch, seq, vocab_size]   (每个位置预测下一个词)

分类 GPT:
  输入 → Transformer → 取最后一个 token → [batch, emb_dim]
                     → 新的分类头 → [batch, num_classes]
```

**为什么取最后一个 token**：由于因果掩码，最后一个 token 的隐藏状态已经"看过"了整个输入序列，信息最完整。

### 2. 参数冻结策略

微调的关键：不需要从头训练，只需要更新少量参数。

```
冻结策略 (LayerFreezer.PrepareForClassification):

1. FreezeAll()              ← 冻结所有参数（不计算梯度）
2. UnfreezeByPrefix("trf_blocks.{last}")  ← 解冻最后一个 Transformer Block
3. UnfreezeByPrefix("final_norm")         ← 解冻最终 LayerNorm
4. ReplaceOutHead(emb_dim, num_classes)   ← 替换输出头
```

**为什么这样冻结**：
- 底层学到的是通用语言特征（语法、语义），不需要改
- 顶层更接近任务，需要适配
- 新的分类头从随机初始化开始，必须训练

### 3. 数据集处理

`SpamDataset` 处理流程：

```
原始文本 → 分词 → 截断到 max_length → padding 到等长
标签 → 整数 (0=正常, 1=垃圾邮件)
```

padding 确保同一个 batch 内所有样本长度一致。

### 4. 分类训练循环

与预训练的区别：

```
预训练: loss = CrossEntropy(logits[:, :, :], targets)   (所有位置)
分类:   loss = CrossEntropy(logits[:, -1, :], labels)   (只用最后位置)
```

### 5. 评估指标

```
Accuracy = 正确预测数 / 总预测数

计算方式:
  logits = model(input)[:, -1, :]    ← 最后一个 token 的输出
  predictions = argmax(logits)        ← 预测类别
  correct = sum(predictions == labels)
```

## 迁移学习的直觉

微调就像"站在巨人肩膀上"：

1. **预训练模型**已经学会了语言的通用知识
2. **冻结底层**保留了这些知识
3. **解冻顶层 + 新头**让模型适配到具体任务
4. **少量标注数据**就能达到不错的效果

这就是为什么 base model 可以复用——不同任务只需要少量结构改动。

## 验证方式

- 冻结前后可训练参数数量大幅减少
- 分类准确率随训练提升
- 验证集准确率不明显低于训练集（无严重过拟合）
