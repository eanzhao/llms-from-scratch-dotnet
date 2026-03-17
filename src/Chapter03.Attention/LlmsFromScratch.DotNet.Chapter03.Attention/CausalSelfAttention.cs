using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter03.Attention;

/// <summary>
/// 因果自注意力 - 加入因果掩码的自注意力
/// 对应书中 Chapter 3 的因果注意力部分
///
/// 因果掩码确保每个位置只能关注它自身及之前的位置，
/// 防止模型"偷看"未来的 token。
/// 实现方式：将注意力分数的上三角部分填充为 -∞，
/// softmax 后这些位置的权重趋近于 0。
/// </summary>
public class CausalSelfAttention : Module
{
    private readonly Linear _wQuery;
    private readonly Linear _wKey;
    private readonly Linear _wValue;
    private readonly int _dOut;
    private readonly DropoutLayer _dropout;
    private readonly Tensor _mask; // 因果掩码（上三角矩阵）

    public CausalSelfAttention(int dIn, int dOut, int contextLength, float dropoutRate = 0.0f, Random? rng = null)
    {
        _dOut = dOut;
        _wQuery = new Linear(dIn, dOut, bias: false, rng: rng);
        _wKey = new Linear(dIn, dOut, bias: false, rng: rng);
        _wValue = new Linear(dIn, dOut, bias: false, rng: rng);
        _dropout = new DropoutLayer(dropoutRate, rng);

        // 上三角掩码: mask[i,j] = 1 if j > i (需要被遮蔽的位置)
        _mask = Tensor.Triu(contextLength, contextLength, diagonal: 1);

        RegisterModule("W_query", _wQuery);
        RegisterModule("W_key", _wKey);
        RegisterModule("W_value", _wValue);
        RegisterModule("dropout", _dropout);
    }

    public override Tensor Forward(Tensor input)
    {
        int numTokens = input.Shape[^2];

        var queries = _wQuery.Forward(input);
        var keys = _wKey.Forward(input);
        var values = _wValue.Forward(input);

        // 注意力分数
        var keysT = TensorOps.Transpose(keys, -2, -1);
        var scores = TensorOps.MatMul(queries, keysT);
        float scale = MathF.Sqrt(_dOut);
        scores = TensorOps.MulScalar(scores, 1.0f / scale);

        // 截取掩码到当前序列长度并应用
        var maskSlice = TensorOps.Slice(_mask, [(0, numTokens), (0, numTokens)]);
        scores = TensorOps.MaskedFill(scores, maskSlice, float.NegativeInfinity);

        var weights = TensorOps.Softmax(scores, dim: -1);
        weights = _dropout.Forward(weights);

        return TensorOps.MatMul(weights, values);
    }
}
