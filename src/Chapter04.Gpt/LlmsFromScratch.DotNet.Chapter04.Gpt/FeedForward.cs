using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// 前馈网络（对应 PyTorch 版 FeedForward）
///
/// 结构: Linear(embDim → 4*embDim) → GELU → Linear(4*embDim → embDim)
///
/// 两层 MLP，中间维度扩展 4 倍。GELU 激活函数提供非线性变换。
/// 这是 Transformer 中注意力层之后的"思考"步骤。
/// </summary>
public class FeedForward : Module
{
    private readonly Sequential _layers;

    public FeedForward(GptConfig cfg, Random? rng = null)
    {
        _layers = new Sequential(
            new Linear(cfg.EmbDim, 4 * cfg.EmbDim, rng: rng),
            new Gelu(),
            new Linear(4 * cfg.EmbDim, cfg.EmbDim, rng: rng)
        );

        RegisterModule("layers", _layers);
    }

    public override Tensor Forward(Tensor input)
    {
        return _layers.Forward(input);
    }
}
