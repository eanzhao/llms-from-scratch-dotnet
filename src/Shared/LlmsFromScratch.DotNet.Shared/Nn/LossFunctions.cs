using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// 损失函数集合
/// </summary>
public static class LossFunctions
{
    /// <summary>
    /// 交叉熵损失（对应 PyTorch 的 F.cross_entropy）
    /// 内部使用 log-softmax 以保证数值稳定性
    ///
    /// logits: [N, C] 其中 N = batch*seq, C = 类别数/词汇量
    /// targets: [N] 每个元素是 [0, C) 的整数类别索引
    /// ignoreIndex: 忽略此索引的损失（用于 padding 掩码）
    /// </summary>
    public static Tensor CrossEntropyLoss(Tensor logits, Tensor targets, int ignoreIndex = -100)
    {
        if (logits.Ndim != 2)
            throw new ArgumentException($"logits 必须是 2 维 [N, C]，当前形状 {Tensor.ShapeToString(logits.Shape)}");
        if (targets.Ndim != 1)
            throw new ArgumentException($"targets 必须是 1 维 [N]，当前形状 {Tensor.ShapeToString(targets.Shape)}");

        int N = logits.Shape[0];
        int C = logits.Shape[1];

        if (targets.Shape[0] != N)
            throw new ArgumentException($"logits 行数 {N} 与 targets 长度 {targets.Shape[0]} 不匹配");

        // 计算 log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
        var logSoftmax = new float[N * C];
        var softmaxValues = new float[N * C]; // 保存 softmax 值用于反向传播

        float totalLoss = 0;
        int validCount = 0;

        for (int i = 0; i < N; i++)
        {
            int targetIdx = (int)targets.Data[i];

            if (targetIdx == ignoreIndex)
                continue;

            // 找 max（数值稳定性）
            float maxVal = float.NegativeInfinity;
            for (int j = 0; j < C; j++)
            {
                float val = logits.Data[i * C + j];
                if (val > maxVal) maxVal = val;
            }

            // 计算 exp(x - max) 和 sum
            float sumExp = 0;
            for (int j = 0; j < C; j++)
            {
                softmaxValues[i * C + j] = MathF.Exp(logits.Data[i * C + j] - maxVal);
                sumExp += softmaxValues[i * C + j];
            }

            // 归一化得到 softmax 值
            for (int j = 0; j < C; j++)
                softmaxValues[i * C + j] /= sumExp;

            // log-softmax = x - max - log(sumExp)
            float logSumExp = MathF.Log(sumExp);
            for (int j = 0; j < C; j++)
                logSoftmax[i * C + j] = logits.Data[i * C + j] - maxVal - logSumExp;

            // NLL: -log_softmax[target]
            totalLoss -= logSoftmax[i * C + targetIdx];
            validCount++;
        }

        float meanLoss = validCount > 0 ? totalLoss / validCount : 0;
        var result = Tensor.FromScalar(meanLoss);
        result.RequiresGrad = logits.RequiresGrad;

        if (logits.RequiresGrad)
        {
            result.Parents = [logits];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(logits);
                float scale = result.Grad![0] / (validCount > 0 ? validCount : 1);

                // 交叉熵 + log-softmax 的合并梯度: softmax(x) - one_hot(target)
                for (int i = 0; i < N; i++)
                {
                    int targetIdx2 = (int)targets.Data[i];
                    if (targetIdx2 == ignoreIndex) continue;

                    for (int j = 0; j < C; j++)
                    {
                        float grad = softmaxValues[i * C + j];
                        if (j == targetIdx2)
                            grad -= 1.0f;
                        logits.Grad![i * C + j] += grad * scale;
                    }
                }
            };
        }

        return result;
    }
}
