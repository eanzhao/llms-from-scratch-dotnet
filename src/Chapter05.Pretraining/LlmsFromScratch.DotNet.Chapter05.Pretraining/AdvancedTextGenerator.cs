using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// 高级文本生成器（对应 Python 版 generate，带 temperature 和 top-k）
///
/// 相比贪心解码（TextGenerator），增加了:
/// - Temperature 缩放: 控制随机性（&gt;1 更随机，&lt;1 更确定）
/// - Top-k 采样: 只从概率最高的 k 个 token 中采样
/// - 提前终止: 遇到 eosId 时停止生成
/// </summary>
public static class AdvancedTextGenerator
{
    /// <summary>
    /// 高级文本生成
    /// </summary>
    /// <param name="model">GPT 模型</param>
    /// <param name="idx">初始 token 序列 [batch, seqLen]</param>
    /// <param name="maxNewTokens">最大生成 token 数</param>
    /// <param name="contextSize">上下文窗口大小</param>
    /// <param name="temperature">温度参数 (0=贪心, &gt;0 采样)</param>
    /// <param name="topK">Top-K 采样参数 (null=不限制)</param>
    /// <param name="eosId">终止 token ID (null=不提前终止)</param>
    public static Tensor Generate(GptModel model, Tensor idx, int maxNewTokens,
        int contextSize, float temperature = 0.0f, int? topK = null, int? eosId = null)
    {
        model.SetTraining(false);
        var rng = new Random();
        var currentIdx = idx;

        for (int i = 0; i < maxNewTokens; i++)
        {
            // 裁剪上下文
            int currentSeqLen = currentIdx.Shape[1];
            Tensor idxCond = currentSeqLen > contextSize
                ? TensorOps.Slice(currentIdx, [(0, currentIdx.Shape[0]), (currentSeqLen - contextSize, currentSeqLen)])
                : currentIdx;

            var logits = model.Forward(idxCond);

            // 取最后时间步
            int lastPos = logits.Shape[1] - 1;
            var lastLogits = TensorOps.Slice(logits,
                [(0, logits.Shape[0]), (lastPos, lastPos + 1), (0, logits.Shape[2])]);
            lastLogits = lastLogits.Reshape(logits.Shape[0], logits.Shape[2]);

            // Top-K 过滤
            if (topK != null)
            {
                lastLogits = ApplyTopK(lastLogits, topK.Value);
            }

            Tensor idxNext;

            if (temperature > 0)
            {
                // 温度缩放 + 采样
                var scaled = TensorOps.MulScalar(lastLogits, 1.0f / temperature);
                var probs = TensorOps.Softmax(scaled, dim: -1);
                idxNext = MultinomialSample(probs, rng);
            }
            else
            {
                // 贪心
                idxNext = TensorOps.Argmax(lastLogits, dim: -1);
            }

            idxNext = idxNext.Reshape(idxNext.Shape[0], 1);
            currentIdx = TensorOps.Concat([currentIdx, idxNext], dim: 1);

            // 检查 EOS
            if (eosId != null && (int)idxNext.Data[0] == eosId.Value)
                break;
        }

        return currentIdx;
    }

    /// <summary>Top-K 过滤：只保留前 K 个最大值，其余设为 -∞</summary>
    private static Tensor ApplyTopK(Tensor logits, int k)
    {
        int batchSize = logits.Shape[0];
        int vocabSize = logits.Shape[1];
        var resultData = (float[])logits.Data.Clone();

        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * vocabSize;

            // 找第 K 大的值作为阈值
            var values = new float[vocabSize];
            Array.Copy(resultData, offset, values, 0, vocabSize);
            Array.Sort(values);
            Array.Reverse(values);
            float threshold = values[Math.Min(k - 1, vocabSize - 1)];

            // 低于阈值的设为 -∞
            for (int j = 0; j < vocabSize; j++)
            {
                if (resultData[offset + j] < threshold)
                    resultData[offset + j] = float.NegativeInfinity;
            }
        }

        return new Tensor(resultData, (int[])logits.Shape.Clone());
    }

    /// <summary>从概率分布中采样（多项式采样）</summary>
    private static Tensor MultinomialSample(Tensor probs, Random rng)
    {
        int batchSize = probs.Shape[0];
        int vocabSize = probs.Shape[1];
        var resultData = new float[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            float r = (float)rng.NextDouble();
            float cumSum = 0;
            int offset = b * vocabSize;

            for (int j = 0; j < vocabSize; j++)
            {
                cumSum += probs.Data[offset + j];
                if (r <= cumSum)
                {
                    resultData[b] = j;
                    break;
                }
            }
        }

        return new Tensor(resultData, [batchSize]);
    }
}
