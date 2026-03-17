using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// 全连接层（对应 PyTorch 的 nn.Linear）
/// 计算: output = input @ Weight^T + Bias
/// Weight 形状: [outFeatures, inFeatures]
/// </summary>
public class Linear : Module
{
    public Tensor Weight { get; }
    public Tensor? Bias { get; }
    public int InFeatures { get; }
    public int OutFeatures { get; }

    public Linear(int inFeatures, int outFeatures, bool bias = true, Random? rng = null)
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;

        // Kaiming 均匀初始化
        var weightData = TensorRandom.KaimingUniform(inFeatures, outFeatures * inFeatures, rng);
        Weight = new Tensor(weightData, [outFeatures, inFeatures], requiresGrad: true);
        RegisterParameter("weight", Weight);

        if (bias)
        {
            var biasData = TensorRandom.KaimingUniform(inFeatures, outFeatures, rng);
            Bias = new Tensor(biasData, [outFeatures], requiresGrad: true);
            RegisterParameter("bias", Bias);
        }
    }

    /// <summary>
    /// 前向传播: output = input @ Weight^T + Bias
    /// input 形状: [..., inFeatures]
    /// output 形状: [..., outFeatures]
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        // input: [..., inFeatures], Weight: [outFeatures, inFeatures]
        // Weight^T: [inFeatures, outFeatures]
        // result: [..., outFeatures]
        var wt = TensorOps.Transpose(Weight, 0, 1); // [inFeatures, outFeatures]

        // 处理高维输入：将 [..., inFeatures] 展平为 [batch, inFeatures]
        int lastDim = input.Shape[^1];
        if (lastDim != InFeatures)
            throw new ArgumentException($"输入最后一维 {lastDim} 与 inFeatures {InFeatures} 不匹配");

        int batchSize = input.Size / lastDim;
        var flatInput = input.Reshape(batchSize, InFeatures);
        var result = TensorOps.MatMul(flatInput, wt); // [batch, outFeatures]

        // 恢复原始批量维度
        var outputShape = new int[input.Ndim];
        Array.Copy(input.Shape, outputShape, input.Ndim - 1);
        outputShape[^1] = OutFeatures;
        result = result.Reshape(outputShape);

        if (Bias != null)
            result = TensorOps.Add(result, Bias);

        return result;
    }
}
