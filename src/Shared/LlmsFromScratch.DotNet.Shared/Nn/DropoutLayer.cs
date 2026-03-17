using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// Dropout 层（对应 PyTorch 的 nn.Dropout）
/// 训练时随机将元素置零，推理时直通
/// 使用 inverted dropout：训练时按 1/(1-p) 缩放，推理时不需要缩放
/// </summary>
public class DropoutLayer : Module
{
    public float Rate { get; }
    private readonly Random _rng;

    public DropoutLayer(float rate, Random? rng = null)
    {
        Rate = rate;
        _rng = rng ?? new Random();
    }

    public override Tensor Forward(Tensor input)
    {
        return TensorOps.Dropout(input, Rate, IsTraining, _rng);
    }
}
