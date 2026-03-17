using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// 顺序容器（对应 PyTorch 的 nn.Sequential）
/// 按注册顺序依次执行各层的 Forward
/// </summary>
public class Sequential : Module
{
    private readonly Module[] _layers;

    public Sequential(params Module[] layers)
    {
        _layers = layers;
        for (int i = 0; i < layers.Length; i++)
            RegisterModule($"layer_{i}", layers[i]);
    }

    public override Tensor Forward(Tensor input)
    {
        var x = input;
        foreach (var layer in _layers)
            x = layer.Forward(x);
        return x;
    }

    public int Count => _layers.Length;

    public Module this[int index] => _layers[index];
}
