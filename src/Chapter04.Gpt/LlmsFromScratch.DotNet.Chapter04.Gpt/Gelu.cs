using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// GELU 激活函数（对应 PyTorch 版 GELU）
///
/// 公式（tanh 近似）:
///   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// GELU 是 GPT 系列模型使用的激活函数，
/// 它比 ReLU 更平滑，在零点附近有渐进式的门控效应。
/// </summary>
public class Gelu : Module
{
    private static readonly float Sqrt2OverPi = MathF.Sqrt(2.0f / MathF.PI);

    public override Tensor Forward(Tensor input)
    {
        // x^3
        var xCubed = TensorOps.Pow(input, 3.0f);

        // 0.044715 * x^3
        var scaled = TensorOps.MulScalar(xCubed, 0.044715f);

        // x + 0.044715 * x^3
        var inner = TensorOps.Add(input, scaled);

        // sqrt(2/π) * (x + 0.044715 * x^3)
        inner = TensorOps.MulScalar(inner, Sqrt2OverPi);

        // tanh(...)
        var tanhResult = TensorOps.Tanh(inner);

        // 1 + tanh(...)
        var ones = Tensor.Ones(tanhResult.Shape);
        var onePlusTanh = TensorOps.Add(ones, tanhResult);

        // 0.5 * x * (1 + tanh(...))
        var half = TensorOps.MulScalar(input, 0.5f);
        return TensorOps.Mul(half, onePlusTanh);
    }
}
