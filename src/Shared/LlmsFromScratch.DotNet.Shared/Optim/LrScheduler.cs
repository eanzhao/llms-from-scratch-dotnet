using System;

namespace LlmsFromScratch.DotNet.Shared.Optim;

/// <summary>
/// 学习率调度器：线性 Warmup + 余弦退火
/// 对应 Appendix D 的 cosine annealing with warmup
///
/// 学习率曲线:
///   Warmup 阶段 (step &lt; warmupSteps):
///     lr = initialLr + step * (peakLr - initialLr) / warmupSteps
///
///   Cosine 退火阶段:
///     progress = (step - warmupSteps) / (totalSteps - warmupSteps)
///     lr = minLr + (peakLr - minLr) * 0.5 * (1 + cos(π * progress))
/// </summary>
public class CosineAnnealingWithWarmup
{
    private readonly float _peakLr;
    private readonly float _minLr;
    private readonly float _initialLr;
    private readonly int _warmupSteps;
    private readonly int _totalSteps;

    public CosineAnnealingWithWarmup(
        float peakLr,
        int totalSteps,
        int warmupSteps,
        float initialLr = 3e-5f,
        float minLr = 1e-6f)
    {
        _peakLr = peakLr;
        _totalSteps = totalSteps;
        _warmupSteps = warmupSteps;
        _initialLr = initialLr;
        _minLr = minLr;
    }

    /// <summary>根据当前 step 计算学习率</summary>
    public float GetLr(int step)
    {
        if (step < _warmupSteps)
        {
            // 线性 warmup
            return _initialLr + step * (_peakLr - _initialLr) / _warmupSteps;
        }

        if (step >= _totalSteps)
            return _minLr;

        // 余弦退火
        float progress = (float)(step - _warmupSteps) / (_totalSteps - _warmupSteps);
        return _minLr + (_peakLr - _minLr) * 0.5f * (1.0f + MathF.Cos(MathF.PI * progress));
    }
}
