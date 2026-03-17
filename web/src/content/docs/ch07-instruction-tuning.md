---
title: Chapter 07 · 做指令微调
order: 7
chapter: 7
summary: 将基础模型调教成AI助手！通过指令微调让模型学会遵循人类指令，理解监督式微调流程和评估方法。
status: done
tags:
  - instruction-tuning
  - sft
  - prompts
---

## 本章目标

理解 assistant-style 模型是怎么从 base model 演化来的——实现完整的 SFT（Supervised Fine-Tuning）流程。

## C# 实现参考

| C# 类 | 文件路径 | 对应 Python | 说明 |
|--------|---------|------------|------|
| `InstructionSample` | `Chapter07.InstructionTuning/InstructionSample.cs` | dict 条目 | 指令数据结构 (record) |
| `PromptTemplate` | `Chapter07.InstructionTuning/PromptTemplate.cs` | `format_input` | 提示词模板 |
| `InstructionDataset` | `Chapter07.InstructionTuning/InstructionDataset.cs` | `InstructionDataset` | 指令数据集（预编码） |
| `InstructionCollator` | `Chapter07.InstructionTuning/InstructionCollator.cs` | `custom_collate_fn` | 变长 padding + 掩码 |
| `InstructionTrainer` | `Chapter07.InstructionTuning/InstructionTrainer.cs` | 训练循环 | SFT 训练循环 |
| `InstructionEvaluator` | `Chapter07.InstructionTuning/InstructionEvaluator.cs` | 评估代码 | 生成式评估 |

## 核心概念

### 1. 指令数据格式

每个训练样本包含三个字段：

```json
{
  "instruction": "将以下句子翻译成英文",
  "input": "你好世界",
  "output": "Hello World"
}
```

### 2. 提示词模板 (Prompt Template)

`PromptTemplate` 将结构化数据格式化为模型输入：

```
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
将以下句子翻译成英文

### Input:
你好世界

### Response:
Hello World
```

- `FormatInput()`：不含 Response（推理时用）
- `FormatFull()`：含 Response（训练时用）
- `### Input:` 段仅在 input 非空时包含

### 3. 变长 Padding + ignoreIndex 掩码

指令数据的关键挑战：每个样本长度不同。

```
样本1 (长度5): [101, 202, 303, 404, 505]
样本2 (长度3): [101, 202, 303]

Padding 后 (pad to max_length):
样本1: [101, 202, 303, 404, 505]
样本2: [101, 202, 303, PAD, PAD]

Target 掩码 (ignoreIndex = -100):
样本1: [202, 303, 404, 505, EOS]
样本2: [202, 303, EOS, -100, -100]   ← padding 位置不计算损失
```

`InstructionCollator` 实现：
1. 找出 batch 内最长的样本
2. 其他样本 padding 到相同长度
3. Target 中 padding 位置设为 `-100`（ignoreIndex）
4. `CrossEntropyLoss` 跳过 `-100` 位置

### 4. SFT 训练

与预训练的区别：

| 方面 | 预训练 | SFT |
|------|--------|-----|
| 数据 | 无标注文本 | (指令, 回复) 对 |
| 损失 | 所有 token | 仅回复部分 token |
| 参数 | 全部训练 | 全部训练（也可以冻结部分） |
| 目标 | 学习语言模式 | 学习遵循指令 |

本项目的 SFT 是全参数微调（所有参数都更新），这对于小模型是可行的。大模型通常使用 LoRA 等参数高效方法。

### 5. 生成式评估

`InstructionEvaluator` 的评估流程：

```
for each test_sample:
    1. prompt = FormatInput(sample)    ← 不含 response
    2. generated = model.Generate(prompt, max_tokens)
    3. response = ExtractResponse(generated)  ← 提取 "### Response:" 后的内容
    4. 对比 response vs sample.Output
```

## 从预训练到 ChatGPT 的路径

```
                    预训练 (Ch05)
大量无标注文本 ──────────────────→ Base Model
                                        │
                    SFT (Ch07)          │
(指令, 回复) 数据 ──────────────────→ SFT Model
                                        │
                    RLHF/DPO            │
人类偏好数据 ────────────────────→ Chat Model (如 ChatGPT)
```

本项目实现了前两步。RLHF/DPO 是更高级的对齐技术，不在本项目范围内。

## 本章边界

- 先把 SFT 做清楚
- 不急着扩展到 RLHF、DPO 或更复杂的偏好优化
- 完成本章后，你拥有一个完整的"文本数据→预训练→两类微调"的最小闭环

## 验证方式

- 检查 padding 后所有样本长度一致
- 验证 target 中 padding 位置都是 -100
- 训练 loss 下降
- 生成的回复格式正确（以 `### Response:` 开头的内容可提取）
