using LlmsFromScratch.DotNet.Chapter02.TextData;
using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// 损失计算器（对应 Python 版 calc_loss_batch / calc_loss_loader）
///
/// 将模型输出的 logits 与目标 token 对比，计算交叉熵损失。
/// 支持单批次和整个数据加载器的损失计算。
/// </summary>
public static class LossCalculator
{
    /// <summary>
    /// 计算单个批次的损失
    /// inputBatch: [batch, seqLen], targetBatch: [batch, seqLen]
    /// </summary>
    public static Tensor CalcLossBatch(Tensor inputBatch, Tensor targetBatch, GptModel model)
    {
        // 前向传播: [batch, seq, vocabSize]
        var logits = model.Forward(inputBatch);

        int batch = logits.Shape[0];
        int seq = logits.Shape[1];
        int vocabSize = logits.Shape[2];

        // 展平: [batch*seq, vocabSize] 和 [batch*seq]
        var flatLogits = logits.Reshape(batch * seq, vocabSize);
        var flatTargets = targetBatch.Reshape(batch * seq);

        return LossFunctions.CrossEntropyLoss(flatLogits, flatTargets);
    }

    /// <summary>
    /// 计算整个数据加载器的平均损失
    /// </summary>
    public static float CalcLossLoader(DataLoader loader, GptModel model, int? numBatches = null)
    {
        float totalLoss = 0;
        int count = 0;
        int maxBatches = numBatches ?? int.MaxValue;

        model.SetTraining(false);

        foreach (var (inputBatch, targetBatch) in loader.GetBatches())
        {
            if (count >= maxBatches) break;
            var loss = CalcLossBatch(inputBatch, targetBatch, model);
            totalLoss += loss.ToScalar();
            count++;
        }

        model.SetTraining(true);
        return count > 0 ? totalLoss / count : float.NaN;
    }

    /// <summary>
    /// 评估模型：返回训练集和验证集的损失
    /// </summary>
    public static (float trainLoss, float valLoss) EvaluateModel(
        GptModel model, DataLoader trainLoader, DataLoader valLoader, int evalIter)
    {
        model.SetTraining(false);
        float trainLoss = CalcLossLoader(trainLoader, model, evalIter);
        float valLoss = CalcLossLoader(valLoader, model, evalIter);
        model.SetTraining(true);
        return (trainLoss, valLoss);
    }
}
