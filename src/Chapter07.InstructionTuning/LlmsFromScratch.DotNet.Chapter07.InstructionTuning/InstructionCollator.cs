using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter07.InstructionTuning;

/// <summary>
/// 指令数据整理器（对应 Python 版 custom_collate_fn）
///
/// 将变长的 token 序列整理为固定长度的批次:
/// 1. 找到批次中最长序列
/// 2. 添加 endoftext token
/// 3. 填充到统一长度
/// 4. 创建 input（去掉最后一个 token）和 target（去掉第一个 token）
/// 5. 将填充位置的 target 标记为 ignoreIndex (-100)
///
/// ignoreIndex 确保损失函数不计算填充位置的梯度。
/// </summary>
public static class InstructionCollator
{
    /// <summary>
    /// 整理一个批次
    /// </summary>
    /// <param name="batch">本批次的 token ID 序列列表</param>
    /// <param name="padTokenId">填充 token ID</param>
    /// <param name="ignoreIndex">损失函数忽略的索引值</param>
    /// <param name="maxLength">最大允许长度 (null=不限制)</param>
    public static (Tensor inputs, Tensor targets) Collate(
        List<int[]> batch, int padTokenId = 0, int ignoreIndex = -100, int? maxLength = null)
    {
        // 找最大长度（+1 for endoftext）
        int maxLen = batch.Max(b => b.Length) + 1;
        if (maxLength != null)
            maxLen = Math.Min(maxLen, maxLength.Value + 1);

        int batchSize = batch.Count;
        int inputLen = maxLen - 1; // input 和 target 都比原序列短1

        var inputData = new float[batchSize * inputLen];
        var targetData = new float[batchSize * inputLen];

        for (int b = 0; b < batchSize; b++)
        {
            // 创建序列: tokens + endoftext
            int seqLen = Math.Min(batch[b].Length, maxLen - 1);
            var fullSeq = new int[maxLen];
            for (int i = 0; i < seqLen; i++)
                fullSeq[i] = batch[b][i];
            fullSeq[seqLen] = padTokenId; // endoftext token

            // 填充
            for (int i = seqLen + 1; i < maxLen; i++)
                fullSeq[i] = padTokenId;

            // 分割为 input 和 target
            for (int i = 0; i < inputLen; i++)
            {
                inputData[b * inputLen + i] = fullSeq[i];
                targetData[b * inputLen + i] = fullSeq[i + 1];
            }

            // 将填充位置的 target 标记为 ignoreIndex
            // 保留第一个 padding token（作为自然结束标记），其余设为 ignoreIndex
            bool foundFirstPad = false;
            for (int i = 0; i < inputLen; i++)
            {
                if (targetData[b * inputLen + i] == padTokenId)
                {
                    if (foundFirstPad)
                        targetData[b * inputLen + i] = ignoreIndex;
                    else
                        foundFirstPad = true;
                }
            }
        }

        return (
            new Tensor(inputData, [batchSize, inputLen]),
            new Tensor(targetData, [batchSize, inputLen])
        );
    }
}
