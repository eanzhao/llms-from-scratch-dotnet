using LlmsFromScratch.DotNet.Chapter03.Attention;
using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// Transformer 块（对应 PyTorch 版 TransformerBlock）
///
/// 采用 Pre-Norm 架构（GPT-2 风格）:
///   x → LayerNorm → MultiHeadAttention → Dropout → + residual
///   x → LayerNorm → FeedForward → Dropout → + residual
///
/// 残差连接（shortcut）让梯度可以直接跳过复杂运算回传，
/// 是深度网络训练的关键技巧。
///
/// 支持 KV Cache 模式（useKvCache=true），此时注意力层为
/// CachedMultiHeadAttention，可缓存 K/V 加速生成。
/// </summary>
public class TransformerBlock : Module
{
    private readonly Module _att; // MultiHeadAttention 或 CachedMultiHeadAttention
    private readonly bool _hasCachedAtt;
    private readonly FeedForward _ff;
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly DropoutLayer _dropShortcut;

    public TransformerBlock(GptConfig cfg, Random? rng = null, bool useKvCache = false)
    {
        if (useKvCache)
        {
            _att = new CachedMultiHeadAttention(
                dIn: cfg.EmbDim,
                dOut: cfg.EmbDim,
                contextLength: cfg.ContextLength,
                numHeads: cfg.NHeads,
                dropoutRate: cfg.DropRate,
                qkvBias: cfg.QkvBias,
                rng: rng);
            _hasCachedAtt = true;
        }
        else
        {
            _att = new MultiHeadAttention(
                dIn: cfg.EmbDim,
                dOut: cfg.EmbDim,
                contextLength: cfg.ContextLength,
                numHeads: cfg.NHeads,
                dropoutRate: cfg.DropRate,
                qkvBias: cfg.QkvBias,
                rng: rng);
            _hasCachedAtt = false;
        }

        _ff = new FeedForward(cfg, rng);
        _norm1 = new LayerNorm(cfg.EmbDim);
        _norm2 = new LayerNorm(cfg.EmbDim);
        _dropShortcut = new DropoutLayer(cfg.DropRate, rng);

        RegisterModule("att", _att);
        RegisterModule("ff", _ff);
        RegisterModule("norm1", _norm1);
        RegisterModule("norm2", _norm2);
        RegisterModule("drop_shortcut", _dropShortcut);
    }

    /// <summary>标准前向传播（无 cache）</summary>
    public override Tensor Forward(Tensor input)
    {
        return Forward(input, useCache: false);
    }

    /// <summary>
    /// 带 KV Cache 选项的前向传播
    /// </summary>
    public Tensor Forward(Tensor input, bool useCache)
    {
        // 注意力子层 + 残差连接
        var shortcut = input;
        var x = _norm1.Forward(input);

        if (useCache && _hasCachedAtt)
            x = ((CachedMultiHeadAttention)_att).Forward(x, useCache: true);
        else
            x = _att.Forward(x);

        x = _dropShortcut.Forward(x);
        x = TensorOps.Add(x, shortcut);

        // 前馈子层 + 残差连接
        shortcut = x;
        x = _norm2.Forward(x);
        x = _ff.Forward(x);
        x = _dropShortcut.Forward(x);
        x = TensorOps.Add(x, shortcut);

        return x;
    }

    /// <summary>重置 KV Cache（如果使用了 CachedMultiHeadAttention）</summary>
    public void ResetCache()
    {
        if (_hasCachedAtt && _att is CachedMultiHeadAttention cachedAtt)
            cachedAtt.ResetCache();
    }
}
