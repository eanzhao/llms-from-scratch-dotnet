using LlmsFromScratch.DotNet.Chapter02.TextData;
using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Optim;
using LlmsFromScratch.DotNet.Shared.Tokenization;

namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// 训练器（对应 Python 版 train_model_simple）
///
/// 核心训练循环:
/// for each epoch:
///     for each batch:
///         1. optimizer.ZeroGrad()     — 清零梯度
///         2. loss = forward(batch)    — 前向传播计算损失
///         3. loss.Backward()          — 反向传播计算梯度
///         4. optimizer.Step()         — 更新参数
///     evaluate and log
/// </summary>
public class Trainer
{
    private readonly GptModel _model;
    private readonly AdamW _optimizer;
    private readonly DataLoader _trainLoader;
    private readonly DataLoader _valLoader;
    private readonly SimpleTokenizer _tokenizer;

    public TrainingMetrics Metrics { get; } = new();

    public Trainer(GptModel model, AdamW optimizer, DataLoader trainLoader,
        DataLoader valLoader, SimpleTokenizer tokenizer)
    {
        _model = model;
        _optimizer = optimizer;
        _trainLoader = trainLoader;
        _valLoader = valLoader;
        _tokenizer = tokenizer;
    }

    /// <summary>
    /// 执行训练
    /// </summary>
    public void Train(int numEpochs, int evalFreq = 5, int evalIter = 1, string startContext = "Every effort moves you")
    {
        int tokensSeen = 0;
        int globalStep = -1;

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            _model.SetTraining(true);

            foreach (var (inputBatch, targetBatch) in _trainLoader.GetBatches())
            {
                _optimizer.ZeroGrad();

                var loss = LossCalculator.CalcLossBatch(inputBatch, targetBatch, _model);
                loss.Backward();
                _optimizer.Step();

                tokensSeen += inputBatch.Size;
                globalStep++;

                // 定期评估
                if (globalStep % evalFreq == 0)
                {
                    var (trainLoss, valLoss) = LossCalculator.EvaluateModel(
                        _model, _trainLoader, _valLoader, evalIter);

                    Metrics.TrainLosses.Add(trainLoss);
                    Metrics.ValLosses.Add(valLoss);
                    Metrics.TokensSeen.Add(tokensSeen);

                    Console.WriteLine($"Ep {epoch + 1} (Step {globalStep:D6}): " +
                        $"Train loss {trainLoss:F3}, Val loss {valLoss:F3}");
                }
            }

            // 每个 epoch 结束后生成示例文本
            TextSampler.GenerateAndPrint(_model, _tokenizer, startContext);
        }
    }
}
