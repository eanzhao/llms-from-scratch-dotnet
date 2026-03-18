using LlmsFromScratch.DotNet.Chapter02.TextData;
using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Optim;
using LlmsFromScratch.DotNet.Shared.Tokenization;

namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// 训练器（对应 Python 版 train_model_simple + Appendix D 增强）
///
/// 核心训练循环:
/// for each epoch:
///     for each batch:
///         1. optimizer.ZeroGrad()     — 清零梯度
///         2. lr = scheduler.GetLr()   — 调整学习率（Appendix D）
///         3. loss = forward(batch)    — 前向传播计算损失
///         4. loss.Backward()          — 反向传播计算梯度
///         5. ClipGradNorm()           — 梯度裁剪（Appendix D）
///         6. optimizer.Step()         — 更新参数
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
    /// 执行训练（基础版，无 LR scheduler）
    /// </summary>
    public void Train(int numEpochs, int evalFreq = 5, int evalIter = 1,
        string startContext = "Every effort moves you")
    {
        Train(numEpochs, evalFreq, evalIter, startContext, lrScheduler: null, maxGradNorm: null);
    }

    /// <summary>
    /// 执行训练（增强版，支持 LR scheduler + 梯度裁剪）
    /// </summary>
    public void Train(int numEpochs, int evalFreq, int evalIter, string startContext,
        CosineAnnealingWithWarmup? lrScheduler, float? maxGradNorm)
    {
        int tokensSeen = 0;
        int globalStep = -1;

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            _model.SetTraining(true);

            foreach (var (inputBatch, targetBatch) in _trainLoader.GetBatches())
            {
                _optimizer.ZeroGrad();
                globalStep++;

                // 学习率调度（Appendix D）
                if (lrScheduler != null)
                    _optimizer.Lr = lrScheduler.GetLr(globalStep);

                var loss = LossCalculator.CalcLossBatch(inputBatch, targetBatch, _model);
                loss.Backward();

                // 梯度裁剪（Appendix D）
                if (maxGradNorm.HasValue)
                    AdamW.ClipGradNorm(_model.Parameters(), maxGradNorm.Value);

                _optimizer.Step();

                tokensSeen += inputBatch.Size;

                // 定期评估
                if (globalStep % evalFreq == 0)
                {
                    var (trainLoss, valLoss) = LossCalculator.EvaluateModel(
                        _model, _trainLoader, _valLoader, evalIter);

                    Metrics.TrainLosses.Add(trainLoss);
                    Metrics.ValLosses.Add(valLoss);
                    Metrics.TokensSeen.Add(tokensSeen);

                    var lrStr = lrScheduler != null ? $", LR {_optimizer.Lr:E2}" : "";
                    Console.WriteLine($"Ep {epoch + 1} (Step {globalStep:D6}): " +
                        $"Train loss {trainLoss:F3}, Val loss {valLoss:F3}{lrStr}");
                }
            }

            // 每个 epoch 结束后生成示例文本
            TextSampler.GenerateAndPrint(_model, _tokenizer, startContext);
        }
    }
}
