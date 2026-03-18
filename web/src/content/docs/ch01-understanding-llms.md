---
title: Chapter 01 · 理解大型语言模型
order: 1
chapter: 1
summary: 先不写代码！建立整体认知框架，理解 LLM 的各个组件如何协同工作，为后续实现打下坚实基础。
status: done
tags:
  - fundamentals
  - concepts
---

## 本章目标

**先别碰键盘！** 这一章不写实现代码，专心**建立认知地图**。

### 学完这一章，你应该能用自己的话解释：

- **Token**：模型眼中的"文字"长什么样？
- **Embedding**：如何把文字变成数字（向量）？
- **注意力机制**：为什么它是 Transformer 的灵魂？
- **GPT**：凭什么能预测下一个词？
- **预训练 vs 微调**：为什么同一个模型能学会不同技能？

## 核心概念总览

### LLM 的完整数据流

```
文本 → Tokenizer → Token IDs → Embedding → Transformer Blocks → Logits → 下一个词
```

每一步的作用：

| 阶段 | 输入 | 输出 | 作用 |
|------|------|------|------|
| Tokenizer | 原始文本 | 整数序列 | 将文字切分成模型能识别的单元 |
| Embedding | Token IDs | 向量矩阵 `[seq_len, emb_dim]` | 把离散 ID 映射为连续向量 |
| Transformer | 向量矩阵 | 上下文化的向量矩阵 | 通过注意力机制捕获上下文关系 |
| Output Head | 最后一层输出 | Logits `[seq_len, vocab_size]` | 预测下一个 token 的概率分布 |

### Transformer 架构要点

1. **自注意力（Self-Attention）**：让每个 token 能"看到"序列中的其他 token，计算相关性权重
2. **因果掩码（Causal Mask）**：GPT 是自回归模型，只能看到当前位置之前的 token
3. **多头注意力（Multi-Head）**：多组注意力并行工作，捕获不同类型的关系模式
4. **前馈网络（Feed-Forward）**：对每个位置独立做非线性变换，增强表达能力
5. **残差连接 + 层归一化**：帮助深层网络稳定训练

### 预训练 vs 微调

- **预训练（Pre-training）**：在大量无标注文本上训练，学习语言的通用模式（Ch05）
- **分类微调（Classification Fine-tuning）**：冻结大部分参数，换分类头，在标注数据上训练（Ch06）
- **指令微调（Instruction Fine-tuning / SFT）**：用指令-回复数据对全模型微调，让模型学会遵循指令（Ch07）

## 技术决策

本项目采用**纯 C# 手写**方式实现所有组件：

- **张量系统**：基于 `float[]` 的自定义 Tensor 类，支持任意维度
- **自动微分**：Define-by-run 计算图 + 拓扑排序反向传播
- **神经网络模块**：仿 PyTorch 的 `Module` 抽象基类
- **优化器**：AdamW（解耦权重衰减）
- **不依赖任何外部 ML 库**

详细的基础设施文档见 [Shared 基础设施](./shared-infrastructure)。

## 与 Python 版本的对照

| Python (PyTorch) | C# (本项目) | 说明 |
|---|---|---|
| `torch.Tensor` | `Tensor` ([Shared/Tensors/Tensor.cs](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Shared/LlmsFromScratch.DotNet.Shared/Tensors/Tensor.cs)) | 自实现，含 autograd |
| `torch.nn.Module` | `Module` ([Shared/Nn/Module.cs](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Shared/LlmsFromScratch.DotNet.Shared/Nn/Module.cs)) | 抽象基类 |
| `torch.nn.Linear` | `Linear` ([Shared/Nn/Linear.cs](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Shared/LlmsFromScratch.DotNet.Shared/Nn/Linear.cs)) | 全连接层 |
| `torch.nn.Embedding` | `Embedding` ([Shared/Nn/Embedding.cs](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Shared/LlmsFromScratch.DotNet.Shared/Nn/Embedding.cs)) | 查找表 |
| `torch.optim.AdamW` | `AdamW` ([Shared/Optim/AdamW.cs](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Shared/LlmsFromScratch.DotNet.Shared/Optim/AdamW.cs)) | 优化器 |
| `torch.nn.CrossEntropyLoss` | `CrossEntropyLoss` ([Shared/Nn/LossFunctions.cs](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Shared/LlmsFromScratch.DotNet.Shared/Nn/LossFunctions.cs)) | 损失函数 |

## 后续章节预览

| 章节 | 主题 | 核心产出 |
|------|------|----------|
| Ch02 | 处理文本数据 | Tokenizer、DataLoader、Embedding |
| Ch03 | 注意力机制 | SelfAttention → MultiHeadAttention |
| Ch04 | GPT 模型 | LayerNorm、GELU、TransformerBlock、GptModel |
| Ch05 | 预训练 | Trainer、LossCalculator、TextGenerator |
| Ch06 | 分类微调 | 冻结/解冻、分类头替换、SpamDataset |
| Ch07 | 指令微调 | PromptTemplate、InstructionCollator、SFT |
