namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// 训练指标跟踪器
/// 记录训练过程中的损失和 token 数量
/// </summary>
public class TrainingMetrics
{
    public List<float> TrainLosses { get; } = new();
    public List<float> ValLosses { get; } = new();
    public List<int> TokensSeen { get; } = new();

    public void PrintSummary()
    {
        Console.WriteLine("\n═══ 训练摘要 ═══");
        if (TrainLosses.Count > 0)
        {
            Console.WriteLine($"  初始训练损失: {TrainLosses[0]:F4}");
            Console.WriteLine($"  最终训练损失: {TrainLosses[^1]:F4}");
            Console.WriteLine($"  最终验证损失: {ValLosses[^1]:F4}");
            Console.WriteLine($"  已处理 token: {TokensSeen[^1]:N0}");
            Console.WriteLine($"  评估次数: {TrainLosses.Count}");
        }
    }
}
