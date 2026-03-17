using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Optim;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter06.Classification;

/// <summary>
/// 分类训练器（对应 Python 版 train_classifier_simple）
///
/// 与预训练 Trainer 的区别:
/// - 使用最后一个 token 的 logits 做分类（而非所有 token）
/// - 跟踪准确率而非 perplexity
/// - 交叉熵损失只针对 numClasses 个类别
/// </summary>
public class ClassificationTrainer
{
    public List<float> TrainLosses { get; } = new();
    public List<float> ValLosses { get; } = new();
    public List<float> TrainAccs { get; } = new();
    public List<float> ValAccs { get; } = new();

    /// <summary>
    /// 计算分类损失（取最后一个 token 的 logits）
    /// </summary>
    public static Tensor CalcClassLoss(Tensor inputBatch, Tensor targetBatch, GptModel model)
    {
        var logits = model.Forward(inputBatch); // [batch, seq, numClasses]

        int batch = logits.Shape[0];
        int seq = logits.Shape[1];
        int numClasses = logits.Shape[2];

        // 取最后一个 token 的 logits: [batch, numClasses]
        var lastLogits = TensorOps.Slice(logits,
            [(0, batch), (seq - 1, seq), (0, numClasses)]);
        lastLogits = lastLogits.Reshape(batch, numClasses);

        // 目标标签 [batch]
        var targets = targetBatch.Reshape(batch);

        return LossFunctions.CrossEntropyLoss(lastLogits, targets);
    }

    /// <summary>
    /// 训练分类模型
    /// </summary>
    public void Train(GptModel model, AdamW optimizer,
        List<(Tensor input, Tensor label)> trainData,
        List<(Tensor input, Tensor label)> valData,
        int numEpochs, int batchSize, int evalFreq = 50)
    {
        int globalStep = 0;

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            model.SetTraining(true);

            for (int i = 0; i < trainData.Count; i += batchSize)
            {
                int actualBatch = Math.Min(batchSize, trainData.Count - i);
                var (inputBatch, targetBatch) = CreateBatch(trainData, i, actualBatch);

                optimizer.ZeroGrad();
                var loss = CalcClassLoss(inputBatch, targetBatch, model);
                loss.Backward();
                optimizer.Step();
                globalStep++;

                if (globalStep % evalFreq == 0)
                {
                    float trainLoss = EvalLoss(model, trainData, batchSize);
                    float valLoss = EvalLoss(model, valData, batchSize);
                    TrainLosses.Add(trainLoss);
                    ValLosses.Add(valLoss);

                    Console.WriteLine($"Ep {epoch + 1} Step {globalStep}: " +
                        $"Train loss {trainLoss:F3}, Val loss {valLoss:F3}");
                }
            }

            // 每 epoch 计算准确率
            float trainAcc = ClassificationMetrics.CalcAccuracy(model, trainData, batchSize);
            float valAcc = ClassificationMetrics.CalcAccuracy(model, valData, batchSize);
            TrainAccs.Add(trainAcc);
            ValAccs.Add(valAcc);

            Console.WriteLine($"Epoch {epoch + 1}: Train acc {trainAcc:P1}, Val acc {valAcc:P1}");
        }
    }

    private static float EvalLoss(GptModel model, List<(Tensor input, Tensor label)> data, int batchSize)
    {
        model.SetTraining(false);
        float total = 0;
        int count = 0;
        for (int i = 0; i < data.Count; i += batchSize)
        {
            int actual = Math.Min(batchSize, data.Count - i);
            var (inp, tgt) = CreateBatch(data, i, actual);
            total += CalcClassLoss(inp, tgt, model).ToScalar();
            count++;
        }
        model.SetTraining(true);
        return count > 0 ? total / count : 0;
    }

    private static (Tensor inputs, Tensor targets) CreateBatch(
        List<(Tensor input, Tensor label)> data, int start, int count)
    {
        int seqLen = data[0].input.Shape[0];
        var inputData = new float[count * seqLen];
        var targetData = new float[count];

        for (int i = 0; i < count; i++)
        {
            Array.Copy(data[start + i].input.Data, 0, inputData, i * seqLen, seqLen);
            targetData[i] = data[start + i].label.Data[0];
        }

        return (new Tensor(inputData, [count, seqLen]), new Tensor(targetData, [count]));
    }
}
