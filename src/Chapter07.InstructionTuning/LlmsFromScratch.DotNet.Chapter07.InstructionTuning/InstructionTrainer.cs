using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Optim;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter07.InstructionTuning;

/// <summary>
/// 指令微调训练器（对应 Python 版 gpt_instruction_finetuning.py 的训练部分）
///
/// 与分类微调的区别:
/// - 全参数微调（不冻结任何层）
/// - 使用自定义 collator 处理变长序列
/// - 使用 ignoreIndex 跳过填充 token 的损失
/// - 训练目标是生成完整的 response 文本
/// </summary>
public class InstructionTrainer
{
    public List<float> TrainLosses { get; } = new();
    public List<float> ValLosses { get; } = new();

    /// <summary>
    /// 执行指令微调
    /// </summary>
    public void Train(GptModel model, AdamW optimizer,
        InstructionDataset trainDataset, InstructionDataset valDataset,
        int numEpochs, int batchSize, int evalFreq = 5,
        int padTokenId = 0, int? maxLength = null)
    {
        int globalStep = 0;

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            model.SetTraining(true);

            // 遍历训练数据
            for (int i = 0; i < trainDataset.Count; i += batchSize)
            {
                int actual = Math.Min(batchSize, trainDataset.Count - i);
                var batch = new List<int[]>();
                for (int j = 0; j < actual; j++)
                    batch.Add(trainDataset.GetItem(i + j));

                var (inputs, targets) = InstructionCollator.Collate(
                    batch, padTokenId, maxLength: maxLength);

                optimizer.ZeroGrad();

                var logits = model.Forward(inputs);
                int vocabSize = logits.Shape[2];
                var flatLogits = logits.Reshape(-1, vocabSize);
                var flatTargets = targets.Reshape(-1);

                var loss = LossFunctions.CrossEntropyLoss(flatLogits, flatTargets, ignoreIndex: -100);
                loss.Backward();
                optimizer.Step();

                globalStep++;

                if (globalStep % evalFreq == 0)
                {
                    float trainLoss = EvalLoss(model, trainDataset, batchSize, padTokenId, maxLength);
                    float valLoss = EvalLoss(model, valDataset, batchSize, padTokenId, maxLength);
                    TrainLosses.Add(trainLoss);
                    ValLosses.Add(valLoss);

                    Console.WriteLine($"Ep {epoch + 1} Step {globalStep}: " +
                        $"Train loss {trainLoss:F3}, Val loss {valLoss:F3}");
                }
            }
        }
    }

    private static float EvalLoss(GptModel model, InstructionDataset dataset,
        int batchSize, int padTokenId, int? maxLength)
    {
        model.SetTraining(false);
        float total = 0;
        int count = 0;

        for (int i = 0; i < dataset.Count; i += batchSize)
        {
            int actual = Math.Min(batchSize, dataset.Count - i);
            var batch = new List<int[]>();
            for (int j = 0; j < actual; j++)
                batch.Add(dataset.GetItem(i + j));

            var (inputs, targets) = InstructionCollator.Collate(batch, padTokenId, maxLength: maxLength);
            var logits = model.Forward(inputs);
            int vocabSize = logits.Shape[2];
            var flatLogits = logits.Reshape(-1, vocabSize);
            var flatTargets = targets.Reshape(-1);

            total += LossFunctions.CrossEntropyLoss(flatLogits, flatTargets, ignoreIndex: -100).ToScalar();
            count++;
        }

        model.SetTraining(true);
        return count > 0 ? total / count : 0;
    }
}
