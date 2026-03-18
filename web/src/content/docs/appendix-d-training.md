---
title: "Appendix D: 训练增强"
order: 90
summary: "学习率调度（Cosine Annealing + Warmup）和梯度裁剪，提升 GPT 训练稳定性"
status: "done"
tags: ["training", "optimization", "appendix"]
---

## 概述

Appendix D 在基础训练循环之上添加两个关键优化技巧:

1. **学习率调度** — Cosine Annealing with Warmup
2. **梯度裁剪** — L2 Norm Gradient Clipping

这两项是现代大语言模型训练的标配，本 repo 在 `CosineAnnealingWithWarmup` 和 `AdamW.ClipGradNorm` 中实现。

---

## 学习率调度

### 为什么需要调度？

固定学习率容易出现两个问题:
- **太大**: 训练发散（loss 暴涨）
- **太小**: 收敛太慢

Cosine Annealing 的解决方案: 先小后大再小。

### 三个阶段

```
lr
 ^
 |    peak_lr
 |     /‾‾‾\
 |    /      \
 |   /        \
 |  / warmup   \ cosine decay
 | /             \
 |/ initial_lr    \_____ min_lr
 +─────────────────────────> step
```

| 阶段 | 步数范围 | 公式 |
|------|---------|------|
| Warmup | `[0, warmup_steps)` | `lr = initial_lr + step * (peak_lr - initial_lr) / warmup_steps` |
| Cosine | `[warmup_steps, total_steps)` | `lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(pi * progress))` |
| 结束 | `>= total_steps` | `lr = min_lr` |

### C# 实现

```csharp
// src/Shared/.../Optim/LrScheduler.cs
var scheduler = new CosineAnnealingWithWarmup(
    peakLr: 5e-4f,
    totalSteps: 1000,
    warmupSteps: 100,
    initialLr: 3e-5f,
    minLr: 1e-6f
);

float lr = scheduler.GetLr(step);
```

---

## 梯度裁剪

### 为什么需要裁剪？

梯度爆炸（gradient explosion）是深度网络的常见问题。当梯度范数过大时，参数更新会"跳"得太远，可能导致训练崩溃。

### L2 Norm Clipping

```
total_norm = sqrt(sum(||grad_i||^2))

if total_norm > max_norm:
    scale = max_norm / total_norm
    grad_i *= scale   # 等比例缩小所有梯度
```

关键: 保持梯度方向不变，只缩小大小。

### C# 实现

```csharp
// src/Shared/.../Optim/AdamW.cs
float totalNorm = AdamW.ClipGradNorm(model.Parameters(), maxNorm: 1.0f);
```

---

## 集成到训练循环

```csharp
// src/Chapter05.Pretraining/.../Trainer.cs
trainer.Train(
    numEpochs: 10,
    evalFreq: 5,
    evalIter: 1,
    startContext: "Every effort moves you",
    lrScheduler: scheduler,    // 传入调度器
    maxGradNorm: 1.0f          // 传入裁剪阈值
);
```

训练循环中的调用顺序:

```
1. optimizer.ZeroGrad()
2. lr = scheduler.GetLr(globalStep)  ← 更新学习率
3. loss = forward(batch)
4. loss.Backward()
5. ClipGradNorm(params, maxNorm)     ← 裁剪梯度
6. optimizer.Step()
```

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `Shared/.../Optim/LrScheduler.cs` | CosineAnnealingWithWarmup 调度器 |
| `Shared/.../Optim/AdamW.cs` | ClipGradNorm 静态方法 + Lr 属性 |
| `Chapter05/.../Trainer.cs` | 集成了调度 + 裁剪的训练循环 |
