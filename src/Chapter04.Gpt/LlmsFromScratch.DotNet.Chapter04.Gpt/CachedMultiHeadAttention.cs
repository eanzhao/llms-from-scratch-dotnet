using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// 带 KV Cache 的多头注意力（对应 ch04/03_kv-cache）
///
/// KV Cache 优化原理:
/// - 标准注意力: 每生成一个新 token，要对完整序列重新计算 K 和 V
/// - KV Cache: 缓存之前的 K/V，新 token 只需计算 Q，与缓存的 K/V 做注意力
/// - 复杂度从 O(N²) 降低到 O(N)
///
/// 生成流程:
/// 1. Reset() 清空缓存
/// 2. Forward(prompt, useCache=true) — 初始化缓存
/// 3. Forward(newToken, useCache=true) — 每步只传 1 个 token
/// </summary>
public class CachedMultiHeadAttention : Module
{
    private readonly int _dOut;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _contextLength;
    private readonly Linear _wQuery;
    private readonly Linear _wKey;
    private readonly Linear _wValue;
    private readonly Linear _outProj;
    private readonly DropoutLayer _dropout;
    private readonly Tensor _mask;

    // KV Cache 状态
    private Tensor? _cacheK;
    private Tensor? _cacheV;
    private int _currentPos;

    public CachedMultiHeadAttention(int dIn, int dOut, int contextLength, int numHeads,
        float dropoutRate = 0.0f, bool qkvBias = false, Random? rng = null)
    {
        if (dOut % numHeads != 0)
            throw new ArgumentException($"dOut ({dOut}) 必须能被 numHeads ({numHeads}) 整除");

        _dOut = dOut;
        _numHeads = numHeads;
        _headDim = dOut / numHeads;
        _contextLength = contextLength;

        _wQuery = new Linear(dIn, dOut, bias: qkvBias, rng: rng);
        _wKey = new Linear(dIn, dOut, bias: qkvBias, rng: rng);
        _wValue = new Linear(dIn, dOut, bias: qkvBias, rng: rng);
        _outProj = new Linear(dOut, dOut, bias: true, rng: rng);
        _dropout = new DropoutLayer(dropoutRate, rng);

        _mask = Tensor.Triu(contextLength, contextLength, diagonal: 1);

        RegisterModule("W_query", _wQuery);
        RegisterModule("W_key", _wKey);
        RegisterModule("W_value", _wValue);
        RegisterModule("out_proj", _outProj);
        RegisterModule("dropout", _dropout);
    }

    /// <summary>无 cache 的标准前向传播</summary>
    public override Tensor Forward(Tensor input) => Forward(input, useCache: false);

    /// <summary>
    /// 带 KV Cache 选项的前向传播
    /// </summary>
    /// <param name="input">[batch, numNewTokens, dIn]</param>
    /// <param name="useCache">是否使用 KV Cache</param>
    public Tensor Forward(Tensor input, bool useCache)
    {
        int batch = input.Shape[0];
        int numNewTokens = input.Shape[1];

        // Q/K/V 投影: [batch, numNewTokens, dOut]
        var queries = _wQuery.Forward(input);
        var keysNew = _wKey.Forward(input);
        var valuesNew = _wValue.Forward(input);

        // reshape: [batch, numNewTokens, numHeads, headDim]
        keysNew = keysNew.Reshape(batch, numNewTokens, _numHeads, _headDim);
        valuesNew = valuesNew.Reshape(batch, numNewTokens, _numHeads, _headDim);
        queries = queries.Reshape(batch, numNewTokens, _numHeads, _headDim);

        // transpose: [batch, numHeads, numNewTokens, headDim]
        keysNew = TensorOps.Transpose(keysNew, 1, 2);
        valuesNew = TensorOps.Transpose(valuesNew, 1, 2);
        queries = TensorOps.Transpose(queries, 1, 2);

        Tensor keys, values;

        if (useCache)
        {
            if (_cacheK == null)
            {
                // 首次调用: 初始化 cache
                _cacheK = keysNew;
                _cacheV = valuesNew;
            }
            else
            {
                // 追加到 cache: dim=2 是 seq 维度
                _cacheK = TensorOps.Concat([_cacheK, keysNew], dim: 2);
                _cacheV = TensorOps.Concat([_cacheV!, valuesNew], dim: 2);
            }
            keys = _cacheK;
            values = _cacheV!;
        }
        else
        {
            keys = keysNew;
            values = valuesNew;
        }

        int numKeysTotal = keys.Shape[2]; // 总的 key 数量

        // 注意力分数: Q @ K^T → [batch, numHeads, numNewTokens, numKeysTotal]
        var keysT = TensorOps.Transpose(keys, 2, 3);
        var attnScores = TensorOps.MatMul(queries, keysT);

        // 缩放
        float scale = MathF.Sqrt(_headDim);
        attnScores = TensorOps.MulScalar(attnScores, 1.0f / scale);

        // 因果掩码（适配 cache 模式）
        Tensor maskSlice;
        if (useCache && numNewTokens < numKeysTotal)
        {
            // Cache 模式: 从 _currentPos 开始的行
            maskSlice = TensorOps.Slice(_mask, [
                (_currentPos, _currentPos + numNewTokens),
                (0, numKeysTotal)
            ]);
        }
        else
        {
            // 标准模式
            maskSlice = TensorOps.Slice(_mask, [
                (0, numNewTokens),
                (0, numKeysTotal)
            ]);
        }

        attnScores = TensorOps.MaskedFill(attnScores, maskSlice, float.NegativeInfinity);

        // Softmax + Dropout
        var attnWeights = TensorOps.Softmax(attnScores, dim: -1);
        attnWeights = _dropout.Forward(attnWeights);

        // 加权求和: [batch, numHeads, numNewTokens, headDim]
        var contextVec = TensorOps.MatMul(attnWeights, values);

        // 合并多头: → [batch, numNewTokens, dOut]
        contextVec = TensorOps.Transpose(contextVec, 1, 2);
        contextVec = contextVec.Reshape(batch, numNewTokens, _dOut);

        // 更新位置指针
        if (useCache)
            _currentPos += numNewTokens;

        return _outProj.Forward(contextVec);
    }

    /// <summary>重置 KV Cache</summary>
    public void ResetCache()
    {
        _cacheK = null;
        _cacheV = null;
        _currentPos = 0;
    }
}
