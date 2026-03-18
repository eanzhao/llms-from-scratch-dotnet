using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// GPT 模型（对应 PyTorch 版 GPTModel）
///
/// 完整架构:
///   Token IDs → Token Embedding + Position Embedding → Dropout
///   → N × TransformerBlock
///   → LayerNorm → Linear (output head) → Logits
///
/// 输入: [batch, seqLen] 的整数 token ID
/// 输出: [batch, seqLen, vocabSize] 的 logits（未归一化的概率）
///
/// 支持 KV Cache 模式（useKvCache=true），推理时缓存 K/V，
/// 每步只需传入新 token，大幅减少重复计算。
/// </summary>
public class GptModel : Module
{
    private readonly Embedding _tokEmb;
    private readonly Embedding _posEmb;
    private readonly DropoutLayer _dropEmb;
    private readonly TransformerBlock[] _trfBlocksArray;
    private readonly Sequential _trfBlocks;
    private readonly LayerNorm _finalNorm;
    private Linear _outHead;
    private readonly bool _useKvCache;

    // KV Cache 模式下的位置指针
    private int _currentPos;

    public GptConfig Config { get; }

    public GptModel(GptConfig cfg, Random? rng = null, bool useKvCache = false)
    {
        Config = cfg;
        _useKvCache = useKvCache;

        _tokEmb = new Embedding(cfg.VocabSize, cfg.EmbDim, rng);
        _posEmb = new Embedding(cfg.ContextLength, cfg.EmbDim, rng);
        _dropEmb = new DropoutLayer(cfg.DropRate, rng);

        _trfBlocksArray = new TransformerBlock[cfg.NLayers];
        var blocks = new Module[cfg.NLayers];
        for (int i = 0; i < cfg.NLayers; i++)
        {
            _trfBlocksArray[i] = new TransformerBlock(cfg, rng, useKvCache);
            blocks[i] = _trfBlocksArray[i];
        }
        _trfBlocks = new Sequential(blocks);

        _finalNorm = new LayerNorm(cfg.EmbDim);
        _outHead = new Linear(cfg.EmbDim, cfg.VocabSize, bias: false, rng: rng);

        RegisterModule("tok_emb", _tokEmb);
        RegisterModule("pos_emb", _posEmb);
        RegisterModule("drop_emb", _dropEmb);
        RegisterModule("trf_blocks", _trfBlocks);
        RegisterModule("final_norm", _finalNorm);
        RegisterModule("out_head", _outHead);
    }

    /// <summary>
    /// 标准前向传播（无 cache）
    /// inIdx: [batch, seqLen] token ID 张量
    /// 返回: [batch, seqLen, vocabSize] logits
    /// </summary>
    public override Tensor Forward(Tensor inIdx)
    {
        return Forward(inIdx, useCache: false);
    }

    /// <summary>
    /// 带 KV Cache 选项的前向传播
    /// </summary>
    /// <param name="inIdx">[batch, seqLen] token ID 张量</param>
    /// <param name="useCache">是否使用 KV Cache</param>
    public Tensor Forward(Tensor inIdx, bool useCache)
    {
        int batchSize = inIdx.Shape[0];
        int seqLen = inIdx.Shape[1];

        // Token 嵌入: [batch, seq] -> [batch, seq, embDim]
        var tokEmbeds = _tokEmb.Forward(inIdx);

        // 位置嵌入（Cache 模式需要偏移位置）
        Tensor positions;
        if (useCache && _useKvCache)
        {
            // 从 _currentPos 开始的位置
            positions = Tensor.Arange(_currentPos, _currentPos + seqLen);
        }
        else
        {
            positions = Tensor.Arange(seqLen);
        }
        var posEmbeds = _posEmb.Forward(positions);

        // 相加并 dropout
        var x = TensorOps.Add(tokEmbeds, posEmbeds);
        x = _dropEmb.Forward(x);

        // 通过 Transformer 块
        if (useCache && _useKvCache)
        {
            for (int i = 0; i < _trfBlocksArray.Length; i++)
                x = _trfBlocksArray[i].Forward(x, useCache: true);
        }
        else
        {
            x = _trfBlocks.Forward(x);
        }

        // 最终层归一化
        x = _finalNorm.Forward(x);

        // 输出投影到词汇表
        var logits = _outHead.Forward(x);

        // 更新位置指针
        if (useCache && _useKvCache)
            _currentPos += seqLen;

        return logits;
    }

    /// <summary>重置 KV Cache（生成新序列前调用）</summary>
    public void ResetKvCache()
    {
        _currentPos = 0;
        foreach (var block in _trfBlocksArray)
            block.ResetCache();
    }

    /// <summary>替换输出头（用于微调，如分类任务）</summary>
    public void ReplaceOutHead(Linear newHead)
    {
        _outHead = newHead;
        SubModules["out_head"] = newHead;
    }

    /// <summary>获取位置嵌入权重（用于确定上下文大小）</summary>
    public Tensor PosEmbWeight => _posEmb.Weight;
}
