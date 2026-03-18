using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// LoRA 包装的全连接层（对应 Appendix E: LinearWithLoRA）
///
/// 将原始 Linear 层冻结，外接 LoRA 旁路:
///   output = original(x) + lora(x)
///          = (x @ W^T + b) + (alpha/rank) * x @ A @ B
///
/// 原始 Linear 的参数不参与训练，只有 LoRA 的 A 和 B 被优化。
/// 这样用极少参数就能适配新任务，大幅减少显存和计算需求。
/// </summary>
public class LinearWithLora : Module
{
    private readonly Linear _original;
    private readonly LoraLayer _lora;

    public Linear Original => _original;
    public LoraLayer Lora => _lora;

    public LinearWithLora(Linear original, int rank, float alpha = 1.0f, Random? rng = null)
    {
        _original = original;

        // 冻结原始层参数
        original.Weight.RequiresGrad = false;
        if (original.Bias != null)
            original.Bias.RequiresGrad = false;

        _lora = new LoraLayer(original.InFeatures, original.OutFeatures, rank, alpha, rng);

        RegisterModule("original", _original);
        RegisterModule("lora", _lora);
    }

    /// <summary>
    /// 前向传播: original(x) + lora(x)
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        var originalOut = _original.Forward(input);
        var loraOut = _lora.Forward(input);
        return TensorOps.Add(originalOut, loraOut);
    }
}
