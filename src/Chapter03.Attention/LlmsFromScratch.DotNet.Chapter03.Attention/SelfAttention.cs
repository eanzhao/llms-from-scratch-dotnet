using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter03.Attention;

/// <summary>
/// 单头自注意力 - 教学用的最简版本
/// 对应书中 Chapter 3 的 "简单自注意力" 部分
///
/// 步骤:
/// 1. 通过线性变换得到 Q、K、V
/// 2. 计算注意力分数: scores = Q @ K^T / sqrt(d_k)
/// 3. 归一化: weights = softmax(scores)
/// 4. 加权求和: output = weights @ V
/// </summary>
public class SelfAttention : Module
{
    private readonly Linear _wQuery;
    private readonly Linear _wKey;
    private readonly Linear _wValue;
    private readonly int _dOut;

    public SelfAttention(int dIn, int dOut, Random? rng = null)
    {
        _dOut = dOut;
        _wQuery = new Linear(dIn, dOut, bias: false, rng: rng);
        _wKey = new Linear(dIn, dOut, bias: false, rng: rng);
        _wValue = new Linear(dIn, dOut, bias: false, rng: rng);

        RegisterModule("W_query", _wQuery);
        RegisterModule("W_key", _wKey);
        RegisterModule("W_value", _wValue);
    }

    /// <summary>
    /// input 形状: [batch, seqLen, dIn]
    /// output 形状: [batch, seqLen, dOut]
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        var queries = _wQuery.Forward(input);   // [batch, seq, dOut]
        var keys = _wKey.Forward(input);         // [batch, seq, dOut]
        var values = _wValue.Forward(input);     // [batch, seq, dOut]

        // 注意力分数: Q @ K^T
        var keysT = TensorOps.Transpose(keys, -2, -1); // [batch, dOut, seq]
        var scores = TensorOps.MatMul(queries, keysT);  // [batch, seq, seq]

        // 缩放: / sqrt(d_k)
        float scale = MathF.Sqrt(_dOut);
        scores = TensorOps.MulScalar(scores, 1.0f / scale);

        // Softmax 归一化
        var weights = TensorOps.Softmax(scores, dim: -1); // [batch, seq, seq]

        // 加权求和
        var output = TensorOps.MatMul(weights, values); // [batch, seq, dOut]

        return output;
    }
}
