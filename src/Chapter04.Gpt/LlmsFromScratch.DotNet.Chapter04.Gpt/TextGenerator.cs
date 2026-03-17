using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// 文本生成器 - 贪心解码（对应 Python 版 generate_text_simple）
///
/// 算法:
/// 1. 取当前 token 序列作为输入
/// 2. 前向传播获取 logits
/// 3. 取最后一个时间步的 logits
/// 4. argmax 选择概率最高的 token
/// 5. 追加到序列中
/// 6. 重复直到生成指定数量的 token
///
/// 注意: 这是最简单的贪心策略，总是选择概率最高的 token。
/// 更高级的策略（如 top-k、temperature）在 Chapter 05 的 AdvancedTextGenerator 中实现。
/// </summary>
public static class TextGenerator
{
    /// <summary>
    /// 贪心文本生成
    /// </summary>
    /// <param name="model">GPT 模型</param>
    /// <param name="idx">初始 token ID 序列 [batch, seqLen]</param>
    /// <param name="maxNewTokens">要生成的新 token 数量</param>
    /// <param name="contextSize">模型支持的最大上下文长度</param>
    /// <returns>扩展后的 token ID 序列 [batch, seqLen + maxNewTokens]</returns>
    public static Tensor GenerateSimple(GptModel model, Tensor idx, int maxNewTokens, int contextSize)
    {
        // 确保在推理模式下（关闭 dropout）
        model.SetTraining(false);

        var currentIdx = idx;

        for (int i = 0; i < maxNewTokens; i++)
        {
            // 裁剪上下文到模型支持的最大长度
            Tensor idxCond;
            int currentSeqLen = currentIdx.Shape[1];
            if (currentSeqLen > contextSize)
            {
                // 只保留最后 contextSize 个 token
                idxCond = TensorOps.Slice(currentIdx,
                    [(0, currentIdx.Shape[0]), (currentSeqLen - contextSize, currentSeqLen)]);
            }
            else
            {
                idxCond = currentIdx;
            }

            // 前向传播: [batch, seq] -> [batch, seq, vocabSize]
            var logits = model.Forward(idxCond);

            // 取最后一个时间步: [batch, vocabSize]
            int lastPos = logits.Shape[1] - 1;
            var lastLogits = TensorOps.Slice(logits,
                [(0, logits.Shape[0]), (lastPos, lastPos + 1), (0, logits.Shape[2])]);
            lastLogits = lastLogits.Reshape(logits.Shape[0], logits.Shape[2]);

            // Argmax 选择概率最高的 token: [batch]
            var idxNext = TensorOps.Argmax(lastLogits, dim: -1);
            // 变为 [batch, 1]
            idxNext = idxNext.Reshape(idxNext.Shape[0], 1);

            // 拼接到序列: [batch, seq+1]
            currentIdx = TensorOps.Concat([currentIdx, idxNext], dim: 1);
        }

        return currentIdx;
    }
}
