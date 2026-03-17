namespace LlmsFromScratch.DotNet.Chapter04.Gpt;

/// <summary>
/// GPT 模型配置（对应 Python 版 GPT_CONFIG_124M 字典）
///
/// 预定义配置:
/// - Gpt2Small (124M): emb=768, layers=12, heads=12
/// - Gpt2Medium (355M): emb=1024, layers=24, heads=16
/// - Gpt2Large (774M): emb=1280, layers=36, heads=20
/// - Gpt2XL (1558M): emb=1600, layers=48, heads=25
/// </summary>
public record GptConfig(
    int VocabSize = 50257,
    int ContextLength = 1024,
    int EmbDim = 768,
    int NHeads = 12,
    int NLayers = 12,
    float DropRate = 0.1f,
    bool QkvBias = false)
{
    /// <summary>GPT-2 Small (124M 参数)</summary>
    public static readonly GptConfig Gpt2Small = new(
        VocabSize: 50257, ContextLength: 1024,
        EmbDim: 768, NHeads: 12, NLayers: 12,
        DropRate: 0.0f, QkvBias: true);

    /// <summary>GPT-2 Medium (355M 参数)</summary>
    public static readonly GptConfig Gpt2Medium = new(
        VocabSize: 50257, ContextLength: 1024,
        EmbDim: 1024, NHeads: 16, NLayers: 24,
        DropRate: 0.0f, QkvBias: true);

    /// <summary>训练用小型配置（缩短 context 以加速）</summary>
    public static readonly GptConfig SmallTraining = new(
        VocabSize: 50257, ContextLength: 256,
        EmbDim: 768, NHeads: 12, NLayers: 12,
        DropRate: 0.1f, QkvBias: false);

    /// <summary>微型测试配置（快速验证流程）</summary>
    public static readonly GptConfig Tiny = new(
        VocabSize: 100, ContextLength: 16,
        EmbDim: 32, NHeads: 2, NLayers: 2,
        DropRate: 0.0f, QkvBias: false);
}
