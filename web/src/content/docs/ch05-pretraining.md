---
title: Chapter 05 · 在无标注数据上预训练
order: 5
chapter: 5
summary: 让模型真正"学习"起来！实现损失函数、训练循环、文本采样和模型检查点，在小规模语料上进行预训练。
status: done
tags:
  - training
  - loss
  - generation
---

## 本章目标

让 GPT 在小规模无标注语料上真正开始学习——实现完整的训练循环。

## C# 实现参考

| C# 类 | 文件路径 | 对应 Python | 说明 |
|--------|---------|------------|------|
| `LossCalculator` | `Chapter05.Pretraining/LossCalculator.cs` | `calc_loss_batch` | 批次/数据集损失计算 |
| `Trainer` | `Chapter05.Pretraining/Trainer.cs` | `train_model_simple` | 核心训练循环 |
| `TrainingMetrics` | `Chapter05.Pretraining/TrainingMetrics.cs` | — | 损失跟踪 |
| `TextSampler` | `Chapter05.Pretraining/TextSampler.cs` | `text_to_token_ids` 等 | 训练中采样 |
| `CheckpointStore` | `Chapter05.Pretraining/CheckpointStore.cs` | `torch.save` | 模型保存/加载 |
| `AdvancedTextGenerator` | `Chapter05.Pretraining/AdvancedTextGenerator.cs` | `generate` | temperature + top-k 采样 |

## 核心算法

### 1. 训练循环

```
for each epoch:
    model.SetTraining(true)
    for each (input_batch, target_batch) in DataLoader:
        1. optimizer.ZeroGrad()       ← 清零所有参数梯度
        2. loss = CrossEntropy(        ← 前向传播 + 损失计算
               model(input_batch),
               target_batch)
        3. loss.Backward()             ← 反向传播计算梯度
        4. optimizer.Step()            ← 更新参数

    定期: 评估 train/val loss, 生成样本文本
```

### 2. 损失函数 (Cross-Entropy Loss)

GPT 的训练目标是最小化 next-token prediction 的交叉熵：

```
logits: [batch, seq_len, vocab_size]   (模型输出)
targets: [batch, seq_len]              (正确的下一个 token)

1. 展平: logits → [batch×seq_len, vocab_size]
         targets → [batch×seq_len]

2. Log-Softmax (数值稳定):
   log_softmax(x_i) = x_i - max(x) - log(Σ exp(x_j - max(x)))

3. NLL Loss:
   loss = -mean(log_softmax(logits)[target])
```

减去 `max(x)` 是为了数值稳定（避免 exp 溢出）。

### 3. AdamW 优化器

AdamW 是 GPT 训练的标准优化器：

```
对每个参数 θ:
    m = β₁ × m + (1-β₁) × grad            ← 一阶动量（梯度均值）
    v = β₂ × v + (1-β₂) × grad²           ← 二阶动量（梯度方差）
    m̂ = m / (1 - β₁ᵗ)                     ← 偏差修正
    v̂ = v / (1 - β₂ᵗ)                     ← 偏差修正
    θ = θ - lr × (m̂ / (√v̂ + ε) + λ × θ)  ← 更新 + 解耦权重衰减
```

**解耦权重衰减**：AdamW 的关键改进是将权重衰减从梯度更新中分离出来，直接对参数做衰减，效果比 L2 正则化更好。

### 4. Temperature + Top-k 采样

贪心解码（argmax）生成的文本单调重复。高级采样方法：

```
1. Temperature 缩放:
   logits = logits / temperature
   - temperature < 1: 分布更尖锐（更确定）
   - temperature > 1: 分布更平坦（更多样）

2. Top-k 过滤:
   - 只保留概率最高的 k 个 token
   - 其他 token 的 logits 设为 -∞

3. Softmax → 概率分布

4. 多项式采样:
   按概率随机选择下一个 token（不是取最大值）

5. EOS 停止:
   遇到结束 token 时停止生成
```

### 5. 模型检查点

```
保存: 参数名 + 形状 + float 数据 → 二进制文件
加载: 二进制文件 → 按名称匹配 → 恢复参数值
```

使用自定义二进制格式：`[参数数量][名称长度+名称+维度数+各维度+float数据] × N`

## 训练观察要点

1. **Loss 下降**：训练正常时，train_loss 和 val_loss 都应该下降
2. **过拟合信号**：train_loss 下降但 val_loss 上升
3. **生成质量**：随着训练进行，采样文本从乱码逐渐变得通顺
4. **学习率**：太大导致 loss 震荡，太小导致收敛慢

## 验证方式

- 小数据集上训练，观察 loss 持续下降
- 训练中生成文本，观察从乱码到通顺的过程
- 保存/加载检查点，验证恢复后 loss 不变
- 比较 temperature=0.1 和 temperature=1.5 的生成差异
