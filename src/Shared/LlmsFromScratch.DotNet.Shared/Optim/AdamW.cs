using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Optim;

/// <summary>
/// AdamW 优化器（解耦权重衰减）
/// 对应 PyTorch 的 torch.optim.AdamW
///
/// 算法:
/// 1. m = β1 * m + (1-β1) * grad          (一阶动量)
/// 2. v = β2 * v + (1-β2) * grad²          (二阶动量)
/// 3. m̂ = m / (1-β1^t)                     (偏差修正)
/// 4. v̂ = v / (1-β2^t)
/// 5. param -= lr * weightDecay * param     (解耦权重衰减)
/// 6. param -= lr * m̂ / (√v̂ + ε)           (参数更新)
/// </summary>
public class AdamW
{
    private readonly List<Tensor> _params;
    private float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;

    /// <summary>当前学习率（可由 LR scheduler 动态设置）</summary>
    public float Lr
    {
        get => _lr;
        set => _lr = value;
    }

    // 每个参数的一阶和二阶动量
    private readonly List<float[]> _m;
    private readonly List<float[]> _v;
    private int _step;

    public AdamW(
        IEnumerable<Tensor> parameters,
        float lr = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.01f)
    {
        _params = parameters.ToList();
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;
        _step = 0;

        _m = new List<float[]>();
        _v = new List<float[]>();

        foreach (var p in _params)
        {
            _m.Add(new float[p.Size]);
            _v.Add(new float[p.Size]);
        }
    }

    /// <summary>执行一步参数更新</summary>
    public void Step()
    {
        _step++;
        float beta1CorrectionFactor = 1.0f - MathF.Pow(_beta1, _step);
        float beta2CorrectionFactor = 1.0f - MathF.Pow(_beta2, _step);

        for (int p = 0; p < _params.Count; p++)
        {
            var param = _params[p];
            if (param.Grad == null) continue;

            var m = _m[p];
            var v = _v[p];

            for (int i = 0; i < param.Size; i++)
            {
                float grad = param.Grad[i];

                // 更新动量
                m[i] = _beta1 * m[i] + (1 - _beta1) * grad;
                v[i] = _beta2 * v[i] + (1 - _beta2) * grad * grad;

                // 偏差修正
                float mHat = m[i] / beta1CorrectionFactor;
                float vHat = v[i] / beta2CorrectionFactor;

                // 解耦权重衰减
                param.Data[i] -= _lr * _weightDecay * param.Data[i];

                // 参数更新
                param.Data[i] -= _lr * mHat / (MathF.Sqrt(vHat) + _eps);
            }
        }
    }

    /// <summary>清零所有参数梯度</summary>
    public void ZeroGrad()
    {
        foreach (var param in _params)
            param.ZeroGrad();
    }

    /// <summary>
    /// 梯度裁剪（L2 范数）
    /// 对应 PyTorch 的 torch.nn.utils.clip_grad_norm_
    ///
    /// 算法:
    /// 1. 计算所有参数梯度的全局 L2 范数: ||G|| = sqrt(Σ ||g_i||²)
    /// 2. 如果 ||G|| > maxNorm: 缩放所有梯度 g_i *= maxNorm / ||G||
    /// </summary>
    /// <returns>裁剪前的梯度总范数</returns>
    public static float ClipGradNorm(IEnumerable<Tensor> parameters, float maxNorm)
    {
        // 计算全局 L2 范数
        float totalNormSq = 0f;
        var paramList = parameters as IList<Tensor> ?? parameters.ToList();

        foreach (var param in paramList)
        {
            if (param.Grad == null) continue;
            for (int i = 0; i < param.Grad.Length; i++)
                totalNormSq += param.Grad[i] * param.Grad[i];
        }

        float totalNorm = MathF.Sqrt(totalNormSq);

        // 如果超过 maxNorm，按比例缩放
        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / (totalNorm + 1e-6f);
            foreach (var param in paramList)
            {
                if (param.Grad == null) continue;
                for (int i = 0; i < param.Grad.Length; i++)
                    param.Grad[i] *= scale;
            }
        }

        return totalNorm;
    }
}
