---
title: Chapter 02 · 处理文本数据
order: 2
chapter: 2
summary: 让文本"数字化"！搭建数据管道，实现 tokenizer、词表、batch 采样和 embedding 层，为模型准备好可消化的数据。
status: done
tags:
  - tokenization
  - dataset
  - embeddings
---

## 本章目标

**让文字变成数字**！搭建数据管道，把原始文本转换成模型能"消化"的格式。

模型只认识数字，我们要做翻译官，把人类语言翻译成模型语言。

## C# 实现参考

| C# 类 | 文件路径 | 对应 Python | 说明 |
|--------|---------|------------|------|
| `Vocabulary` | [`Chapter02.TextData/Vocabulary.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter02.TextData/LlmsFromScratch.DotNet.Chapter02.TextData/Vocabulary.cs) | `SimpleTokenizerV1` | Token↔ID 双向映射 |
| `GptDatasetV1` | [`Chapter02.TextData/GptDatasetV1.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter02.TextData/LlmsFromScratch.DotNet.Chapter02.TextData/GptDatasetV1.cs) | `GPTDatasetV1` | 滑动窗口数据集 |
| `DataLoader` | [`Chapter02.TextData/DataLoader.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter02.TextData/LlmsFromScratch.DotNet.Chapter02.TextData/DataLoader.cs) | `create_dataloader_v1` | 批次迭代器 |
| `EmbeddingDemo` | [`Chapter02.TextData/EmbeddingDemo.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Chapter02.TextData/LlmsFromScratch.DotNet.Chapter02.TextData/EmbeddingDemo.cs) | notebook 演示 | token_emb + pos_emb 演示 |
| `SimpleTokenizer` | [`Shared/Tokenization/SimpleTokenizer.cs`](https://github.com/eanzhao/llms-from-scratch-dotnet/blob/main/src/Shared/LlmsFromScratch.DotNet.Shared/Tokenization/SimpleTokenizer.cs) | `SimpleTokenizerV2` | 词级分词器 |

## 核心算法

### 1. 分词（Tokenization）

分词是 NLP 的第一步——将连续文本切分为离散单元。

```
"Hello, world!" → ["Hello", ",", " ", "world", "!"]
                → [15, 3, 7, 42, 5]   (通过词表映射)
```

本项目使用简单的词级分词器（教学用）。生产级 LLM 通常使用 BPE（Byte Pair Encoding）：
- BPE 从字符级开始，反复合并最常共现的 token 对
- 平衡了词表大小和 token 粒度
- GPT-2 使用约 50,257 个 token 的 BPE 词表

### 2. 滑动窗口数据集

GPT 的训练目标是 **next-token prediction**。给定前 N 个 token，预测第 N+1 个：

```
文本: [t0, t1, t2, t3, t4, t5, t6, ...]
      ─────────────────────────────────
窗口1: 输入 [t0, t1, t2, t3]  →  目标 [t1, t2, t3, t4]
窗口2: 输入 [t1, t2, t3, t4]  →  目标 [t2, t3, t4, t5]
窗口3: 输入 [t2, t3, t4, t5]  →  目标 [t3, t4, t5, t6]
```

`GptDatasetV1` 实现了这个滑动窗口：stride 控制窗口移动步长，max_length 控制窗口大小。

### 3. DataLoader 批次生成

`DataLoader` 将数据集的样本组合成 batch：

```
单个样本: input [seq_len], target [seq_len]
一个 batch: inputs [batch_size, seq_len], targets [batch_size, seq_len]
```

支持 shuffle（Fisher-Yates 洗牌）和 drop_last（丢弃不完整的最后一批）。

### 4. Embedding 层

将离散的 token ID 映射为连续的向量表示：

```
Token Embedding:      token_id → 向量 [emb_dim]      (查表操作)
Positional Embedding: position → 向量 [emb_dim]      (位置信息)
最终输入:             token_emb + pos_emb            (逐元素相加)
```

输出形状：`[batch_size, seq_len, emb_dim]`

## 实现顺序

1. **Vocabulary**：建立 token↔ID 映射
2. **SimpleTokenizer**：文本 → token 序列
3. **GptDatasetV1**：滑动窗口生成 (input, target) 对
4. **DataLoader**：批次迭代器
5. **EmbeddingDemo**：token_emb + pos_emb 演示

## 验证方式

- 输入固定文本，检查 token ID 序列是否正确
- 验证滑动窗口：input 右移一位 = target
- 检查 batch 形状：`[batch_size, seq_len]`
- Embedding 输出形状：`[batch_size, seq_len, emb_dim]`

## 补充知识

### BPE 分词的直觉

BPE 像是一个"压缩算法"：
1. 从字符级开始：`"low"` → `['l', 'o', 'w']`
2. 统计相邻 pair 出现频率
3. 合并最频繁的 pair：`'l' + 'o'` → `'lo'`
4. 重复直到达到目标词表大小

好处是既能处理常见词（整词一个 token），也能处理罕见词（拆成子词）。

### 为什么需要位置编码

注意力机制本身是**置换不变**的——打乱输入顺序，输出不变。但语言是有顺序的！位置编码告诉模型每个 token 在序列中的位置。GPT 使用可学习的位置编码（learnable positional embedding）。
