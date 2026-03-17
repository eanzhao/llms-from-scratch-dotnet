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
/// </summary>
public class GptModel : Module
{
    private readonly Embedding _tokEmb;
    private readonly Embedding _posEmb;
    private readonly DropoutLayer _dropEmb;
    private readonly Sequential _trfBlocks;
    private readonly LayerNorm _finalNorm;
    private Linear _outHead;

    public GptConfig Config { get; }

    public GptModel(GptConfig cfg, Random? rng = null)
    {
        Config = cfg;

        _tokEmb = new Embedding(cfg.VocabSize, cfg.EmbDim, rng);
        _posEmb = new Embedding(cfg.ContextLength, cfg.EmbDim, rng);
        _dropEmb = new DropoutLayer(cfg.DropRate, rng);

        var blocks = new Module[cfg.NLayers];
        for (int i = 0; i < cfg.NLayers; i++)
            blocks[i] = new TransformerBlock(cfg, rng);
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
    /// 前向传播
    /// inIdx: [batch, seqLen] token ID 张量
    /// 返回: [batch, seqLen, vocabSize] logits
    /// </summary>
    public override Tensor Forward(Tensor inIdx)
    {
        int batchSize = inIdx.Shape[0];
        int seqLen = inIdx.Shape[1];

        // Token 嵌入: [batch, seq] -> [batch, seq, embDim]
        var tokEmbeds = _tokEmb.Forward(inIdx);

        // 位置嵌入: [seq] -> [seq, embDim]，广播到 batch 维度
        var positions = Tensor.Arange(seqLen);
        var posEmbeds = _posEmb.Forward(positions);

        // 相加并 dropout
        var x = TensorOps.Add(tokEmbeds, posEmbeds);
        x = _dropEmb.Forward(x);

        // 通过 Transformer 块
        x = _trfBlocks.Forward(x);

        // 最终层归一化
        x = _finalNorm.Forward(x);

        // 输出投影到词汇表
        var logits = _outHead.Forward(x);

        return logits;
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
