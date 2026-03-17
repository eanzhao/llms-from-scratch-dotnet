---
title: Shared 基础设施
order: -1
summary: 本项目的核心——纯 C# 手写的 Tensor 系统、自动微分引擎、神经网络模块和优化器，替代 PyTorch 的全部底层功能。
status: done
tags:
  - tensor
  - autograd
  - module
  - infrastructure
---

## 概述

本项目不使用任何外部 ML 库，所有底层功能均用纯 C# 实现。这些代码位于 `src/Shared/LlmsFromScratch.DotNet.Shared/` 目录下，是所有章节的共同基础。

## 模块一览

| 目录 | 文件 | 说明 |
|------|------|------|
| **Tensors/** | `Tensor.cs` | 核心张量类 |
| | `TensorOps.cs` | 张量运算（含 autograd） |
| | `BroadcastHelper.cs` | NumPy 风格广播 |
| | `Autograd.cs` | 反向传播引擎 |
| | `TensorRandom.cs` | 随机数生成 |
| **Nn/** | `Module.cs` | 神经网络模块基类 |
| | `Linear.cs` | 全连接层 |
| | `Embedding.cs` | 嵌入层 |
| | `DropoutLayer.cs` | Dropout |
| | `Sequential.cs` | 顺序容器 |
| | `LossFunctions.cs` | 损失函数 |
| **Optim/** | `AdamW.cs` | AdamW 优化器 |
| **Tokenization/** | `SimpleTokenizer.cs` | 简单分词器 |
| **IO/** | `ModelSerializer.cs` | 模型序列化 |

## Tensor 系统

### 存储模型

```csharp
public class Tensor
{
    public float[] Data;      // 扁平化存储
    public int[] Shape;       // 形状，如 [2, 3, 4]
    public int[] Strides;     // 步幅，如 [12, 4, 1]
    public Tensor? Grad;      // 梯度（同形状的 Tensor）
    public Action<Tensor>? BackwardFn;  // 反向传播函数
    public List<Tensor> Parents;        // 计算图父节点
}
```

**设计选择**：使用扁平 `float[]` 而非多维数组，通过 `Shape` 和 `Strides` 实现任意维度。这与 PyTorch、NumPy 的底层实现一致。

### 索引计算

```
元素 [i, j, k] 在 Data 中的位置:
offset = i × Strides[0] + j × Strides[1] + k × Strides[2]
```

### 工厂方法

| 方法 | 说明 | 示例 |
|------|------|------|
| `Tensor.Zeros(shape)` | 全零张量 | `Tensor.Zeros(2, 3)` |
| `Tensor.Ones(shape)` | 全一张量 | `Tensor.Ones(4)` |
| `Tensor.Randn(shape)` | 正态分布随机 | `Tensor.Randn(3, 3)` |
| `Tensor.FromArray(data, shape)` | 从数组创建 | `Tensor.FromArray(new[]{1f,2f}, 2)` |
| `Tensor.Arange(start, end)` | 等差序列 | `Tensor.Arange(0, 5)` → [0,1,2,3,4] |
| `Tensor.Triu(n, diagonal)` | 上三角矩阵 | 用于因果掩码 |

### 形状操作

| 方法 | 说明 |
|------|------|
| `Reshape(newShape)` | 改变形状（支持 -1 自动推断） |
| `Transpose(dim0, dim1)` | 交换两个维度 |
| `Unsqueeze(dim)` | 插入大小为 1 的维度 |
| `Squeeze(dim)` | 移除大小为 1 的维度 |

## 自动微分 (Autograd)

### 计算图

每个 `TensorOps` 操作在创建结果张量时注册：
- `Parents`：参与运算的输入张量
- `BackwardFn`：一个闭包，接收输出梯度，计算并累加输入梯度

```
示例: c = a + b

Forward:
  c.Data = a.Data + b.Data
  c.Parents = [a, b]
  c.BackwardFn = (grad_c) => {
      a.Grad += grad_c    // ∂L/∂a = ∂L/∂c × ∂c/∂a = grad_c × 1
      b.Grad += grad_c    // ∂L/∂b = ∂L/∂c × ∂c/∂b = grad_c × 1
  }
```

### 反向传播算法

```
Backward(root):
    1. root.Grad = Ones(root.Shape)           // dL/dL = 1
    2. sorted = TopologicalSort(root)         // DFS 拓扑排序
    3. for node in reverse(sorted):
         if node.BackwardFn != null:
             node.BackwardFn(node.Grad)       // 传播梯度给 Parents
```

拓扑排序保证：在处理任何节点时，它的所有消费者的梯度已经计算完毕。

### 广播 (Broadcasting)

遵循 NumPy 广播规则：

```
规则：从尾部维度对齐，每个维度要么相等，要么其中一个为 1

[3, 4] + [4]     → [3, 4]    ✓  (第二个扩展为 [1, 4])
[2, 1, 3] + [3]  → [2, 1, 3] ✓
[2, 3] + [3, 2]  → 错误       ✗
```

梯度反传时，广播维度需要 `Sum` 回原始形状（`AccumulateGrad`）。

## 神经网络模块 (Nn)

### Module 基类

```csharp
public abstract class Module
{
    Dictionary<string, Module> SubModules;
    Dictionary<string, Tensor> Params;

    abstract Tensor Forward(Tensor input);

    // 递归收集所有可训练参数
    IEnumerable<Tensor> Parameters();

    // 设置训练/推理模式（影响 Dropout 行为）
    void SetTraining(bool training);

    // 清零所有参数梯度
    void ZeroGrad();
}
```

### Linear 层

```
Forward(x):
    // x: [..., in_features]
    // W: [out_features, in_features]
    // b: [out_features]
    return x @ W^T + b
```

权重使用 **Kaiming 均匀初始化**：`U(-bound, bound)`，`bound = √(1/in_features)`

### Embedding 层

```
Forward(token_ids):
    // token_ids: [batch, seq_len]（整数）
    // Weight: [vocab_size, emb_dim]
    return Weight[token_ids]    // 按行索引
```

### CrossEntropyLoss

```
Forward(logits, targets):
    // logits: [N, C]
    // targets: [N]（整数）

    // 数值稳定的 log-softmax:
    max_val = max(logits, dim=-1)
    log_sum_exp = log(Σ exp(logits - max_val)) + max_val
    log_probs = logits - log_sum_exp

    // NLL:
    loss = -mean(log_probs[i, targets[i]])

Backward:
    // 梯度 = softmax(logits) - one_hot(targets)
    // 这是 cross-entropy 的经典梯度公式
```

支持 `ignoreIndex`：值为 -100 的 target 位置不参与损失计算（用于 padding 掩码）。

## AdamW 优化器

```
超参数:
    lr = 0.0004        学习率
    β₁ = 0.9           一阶动量衰减
    β₂ = 0.999         二阶动量衰减
    ε = 1e-8           数值稳定
    weight_decay = 0.1 权重衰减

每步更新:
    m = β₁ × m + (1-β₁) × g              // 梯度的指数移动平均
    v = β₂ × v + (1-β₂) × g²             // 梯度平方的指数移动平均
    m̂ = m / (1 - β₁ᵗ)                    // 偏差修正（t 从 1 开始）
    v̂ = v / (1 - β₂ᵗ)
    θ -= lr × (m̂ / (√v̂ + ε) + λ × θ)    // 参数更新 + 解耦衰减
```

## 与 PyTorch 的对应关系

| PyTorch | 本项目 C# | 关键区别 |
|---------|----------|---------|
| `torch.tensor([1,2,3])` | `Tensor.FromArray(...)` | — |
| `x.backward()` | `x.Backward()` | 内部调用 `Autograd.Backward` |
| `optimizer.zero_grad()` | `optimizer.ZeroGrad()` | — |
| `model.train()` | `model.SetTraining(true)` | — |
| `model.eval()` | `model.SetTraining(false)` | — |
| `model.parameters()` | `model.Parameters()` | 递归收集 |
| `nn.Module` | `Module` | 抽象基类 |
| `torch.no_grad()` | `tensor.Detach()` | 断开计算图 |
| `x.item()` | `x.ToScalar()` | 取标量值 |

## 局限性

本实现是教学用途，与生产级框架相比：

1. **仅 CPU**：不支持 GPU 加速
2. **不支持原地操作**：所有操作都创建新张量
3. **无内存池**：频繁分配 GC 压力大
4. **无 BLAS**：矩阵乘法是朴素三重循环
5. **float32 only**：不支持 half/bfloat16

这些限制在学习场景下是可接受的——目标是理解算法，而非追求性能。
