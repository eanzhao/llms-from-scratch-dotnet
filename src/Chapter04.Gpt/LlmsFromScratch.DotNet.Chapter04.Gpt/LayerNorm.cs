using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// 层归一化（对应 PyTorch 版 LayerNorm）
///
/// 对每个样本的特征维度（最后一维）做归一化:
///   norm_x = (x - mean) / sqrt(var + eps)
///   output = scale * norm_x + shift
///
/// 其中 scale 和 shift 是可学习参数，初始值分别为 1 和 0。
/// 与 BatchNorm 不同，LayerNorm 在特征维度上归一化，不依赖 batch 统计量。
/// </summary>
public class LayerNorm : Module
{
    private readonly float _eps;
    private readonly Tensor _scale;
    private readonly Tensor _shift;
    private readonly int _embDim;

    public LayerNorm(int embDim, float eps = 1e-5f)
    {
        _embDim = embDim;
        _eps = eps;

        var scaleData = new float[embDim];
        Array.Fill(scaleData, 1.0f);
        _scale = new Tensor(scaleData, [embDim], requiresGrad: true);

        _shift = new Tensor(new float[embDim], [embDim], requiresGrad: true);

        RegisterParameter("scale", _scale);
        RegisterParameter("shift", _shift);
    }

    /// <summary>
    /// input 形状: [..., embDim]
    /// output 形状: [..., embDim]
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        // mean 和 variance 沿最后一维计算
        var mean = TensorOps.Mean(input, dim: -1, keepdim: true);
        var variance = TensorOps.Variance(input, dim: -1, keepdim: true);

        // 归一化: (x - mean) / sqrt(var + eps)
        var centered = TensorOps.Sub(input, mean);
        var epsT = Tensor.FromScalar(_eps);
        var varPlusEps = TensorOps.Add(variance, epsT);
        var stdInv = TensorOps.Sqrt(varPlusEps);
        var normX = TensorOps.Div(centered, stdInv);

        // 缩放和偏移: scale * norm_x + shift
        var scaled = TensorOps.Mul(normX, _scale);
        return TensorOps.Add(scaled, _shift);
    }
}
