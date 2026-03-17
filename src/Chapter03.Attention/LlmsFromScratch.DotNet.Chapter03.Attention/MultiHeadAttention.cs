using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter03.Attention;

/// <summary>
/// 多头注意力（对应 PyTorch 版 MultiHeadAttention）
///
/// 核心思想: 将注意力空间拆分为多个"头"，每个头独立计算注意力，
/// 最后拼接起来通过线性变换输出。这让模型能同时关注不同子空间的信息。
///
/// 形状变化流程:
/// input: [batch, seq, dIn]
///   → Q/K/V: [batch, seq, dOut]
///   → reshape: [batch, seq, numHeads, headDim]
///   → transpose: [batch, numHeads, seq, headDim]
///   → attention: [batch, numHeads, seq, headDim]
///   → transpose+reshape: [batch, seq, dOut]
///   → output projection: [batch, seq, dOut]
/// </summary>
public class MultiHeadAttention : Module
{
    private readonly int _dOut;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly Linear _wQuery;
    private readonly Linear _wKey;
    private readonly Linear _wValue;
    private readonly Linear _outProj;
    private readonly DropoutLayer _dropout;
    private readonly Tensor _mask;

    public MultiHeadAttention(int dIn, int dOut, int contextLength, int numHeads,
        float dropoutRate = 0.0f, bool qkvBias = false, Random? rng = null)
    {
        if (dOut % numHeads != 0)
            throw new ArgumentException($"dOut ({dOut}) 必须能被 numHeads ({numHeads}) 整除");

        _dOut = dOut;
        _numHeads = numHeads;
        _headDim = dOut / numHeads;

        _wQuery = new Linear(dIn, dOut, bias: qkvBias, rng: rng);
        _wKey = new Linear(dIn, dOut, bias: qkvBias, rng: rng);
        _wValue = new Linear(dIn, dOut, bias: qkvBias, rng: rng);
        _outProj = new Linear(dOut, dOut, bias: true, rng: rng);
        _dropout = new DropoutLayer(dropoutRate, rng);

        // 上三角因果掩码
        _mask = Tensor.Triu(contextLength, contextLength, diagonal: 1);

        RegisterModule("W_query", _wQuery);
        RegisterModule("W_key", _wKey);
        RegisterModule("W_value", _wValue);
        RegisterModule("out_proj", _outProj);
        RegisterModule("dropout", _dropout);
    }

    public override Tensor Forward(Tensor input)
    {
        int batch = input.Shape[0];
        int numTokens = input.Shape[1];

        // 线性变换得到 Q, K, V: [batch, seq, dOut]
        var queries = _wQuery.Forward(input);
        var keys = _wKey.Forward(input);
        var values = _wValue.Forward(input);

        // 拆分为多个头: [batch, seq, dOut] -> [batch, seq, numHeads, headDim]
        queries = queries.Reshape(batch, numTokens, _numHeads, _headDim);
        keys = keys.Reshape(batch, numTokens, _numHeads, _headDim);
        values = values.Reshape(batch, numTokens, _numHeads, _headDim);

        // 转置: [batch, seq, numHeads, headDim] -> [batch, numHeads, seq, headDim]
        queries = TensorOps.Transpose(queries, 1, 2);
        keys = TensorOps.Transpose(keys, 1, 2);
        values = TensorOps.Transpose(values, 1, 2);

        // 计算注意力分数: Q @ K^T -> [batch, numHeads, seq, seq]
        var keysT = TensorOps.Transpose(keys, 2, 3);
        var attnScores = TensorOps.MatMul(queries, keysT);

        // 应用因果掩码
        var maskSlice = TensorOps.Slice(_mask, [(0, numTokens), (0, numTokens)]);
        attnScores = TensorOps.MaskedFill(attnScores, maskSlice, float.NegativeInfinity);

        // 缩放 + Softmax
        float scale = MathF.Sqrt(_headDim);
        attnScores = TensorOps.MulScalar(attnScores, 1.0f / scale);
        var attnWeights = TensorOps.Softmax(attnScores, dim: -1);
        attnWeights = _dropout.Forward(attnWeights);

        // 加权求和: [batch, numHeads, seq, headDim]
        var contextVec = TensorOps.MatMul(attnWeights, values);

        // 合并多头: [batch, numHeads, seq, headDim] -> [batch, seq, numHeads, headDim] -> [batch, seq, dOut]
        contextVec = TensorOps.Transpose(contextVec, 1, 2);
        contextVec = contextVec.Reshape(batch, numTokens, _dOut);

        // 输出投影
        return _outProj.Forward(contextVec);
    }
}
