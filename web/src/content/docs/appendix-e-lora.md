---
title: "Appendix E: LoRA 微调"
order: 91
section: "appendix"
summary: "LoRA (Low-Rank Adaptation) 参数高效微调，用极少参数适配新任务"
status: "done"
tags: ["finetuning", "lora", "appendix"]
---

## 概述

LoRA (Low-Rank Adaptation) 是一种参数高效微调（PEFT）方法:
- **冻结**原模型所有参数
- 在目标层旁边添加**低秩分解矩阵** A 和 B
- 只训练 A 和 B，参数量减少 90%+

---

## 核心思想

原始全连接层:
```
y = x @ W      # W: [d_in, d_out]
```

LoRA 改造后:
```
y = x @ W + (alpha/rank) * x @ A @ B
    ─────     ──────────────────────
    冻结的           LoRA 旁路
```

| 矩阵 | 形状 | 参数量 | 训练？ |
|-------|------|--------|--------|
| W | `[d_in, d_out]` | d_in * d_out | 冻结 |
| A | `[d_in, rank]` | d_in * rank | 训练 |
| B | `[rank, d_out]` | rank * d_out | 训练 |

当 `rank = 8`, `d_in = d_out = 768` 时:
- W: 589,824 参数
- A + B: 12,288 参数（仅 2%）

---

## 初始化

```
A ← Kaiming 均匀初始化（随机）
B ← 全零初始化（关键！）
```

B 初始化为零确保训练开始时 `A @ B = 0`，LoRA 旁路不改变原模型输出。训练过程中 B 逐渐学习到有用的适配。

---

## C# 实现

### LoraLayer

```csharp
// src/Shared/.../Nn/LoraLayer.cs
public class LoraLayer : Module
{
    Tensor _a;  // [inDim, rank] — Kaiming init
    Tensor _b;  // [rank, outDim] — Zero init

    Forward(x) => (alpha / rank) * MatMul(MatMul(x, A), B)
}
```

### LinearWithLora

```csharp
// src/Shared/.../Nn/LinearWithLora.cs
public class LinearWithLora : Module
{
    Linear _original;   // 冻结
    LoraLayer _lora;    // 可训练

    Forward(x) => original.Forward(x) + lora.Forward(x)
}
```

### 应用到模型

```csharp
// src/Chapter06.Classification/.../LayerFreezer.cs

// 方式 1: 一键应用（替换所有注意力层的 Q/V）
LayerFreezer.ApplyLora(model, rank: 8, alpha: 16);

// 查看参数量变化
var (total, trainable) = LayerFreezer.CountParameters(model);
Console.WriteLine($"Total: {total}, Trainable: {trainable} ({100.0*trainable/total:F1}%)");
```

---

## 为什么只替换 Q 和 V？

原论文实验表明:
- 对 **W_query** 和 **W_value** 加 LoRA 效果最好
- W_key 的影响较小
- out_proj 也有一定效果，但增加参数量

本实现遵循原论文推荐，只在 Q 和 V 上加 LoRA。

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `Shared/.../Nn/LoraLayer.cs` | LoRA 低秩分解层（A @ B 矩阵） |
| `Shared/.../Nn/LinearWithLora.cs` | 包装器: original + LoRA 旁路 |
| `Chapter06/.../LayerFreezer.cs` | ApplyLora() + CountParameters() |
