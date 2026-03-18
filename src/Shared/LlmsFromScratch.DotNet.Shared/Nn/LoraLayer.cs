using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// LoRA 低秩分解层（对应 Appendix E: LoRA）
///
/// LoRA (Low-Rank Adaptation) 原理:
/// - 原始权重矩阵 W [d_in, d_out] 被冻结
/// - 学习两个小矩阵: A [d_in, rank] 和 B [rank, d_out]
/// - 前向传播: output = x @ A @ B * (alpha / rank)
/// - 可训练参数从 d_in * d_out 降至 (d_in + d_out) * rank
///
/// 初始化:
/// - A: Kaiming 均匀初始化（与 PyTorch 一致）
/// - B: 零初始化（关键！确保训练开始时 LoRA 不改变原模型输出）
/// </summary>
public class LoraLayer : Module
{
    private readonly Tensor _a; // [inDim, rank]
    private readonly Tensor _b; // [rank, outDim]
    private readonly float _alpha;
    private readonly int _rank;

    public LoraLayer(int inDim, int outDim, int rank, float alpha, Random? rng = null)
    {
        _rank = rank;
        _alpha = alpha;

        // A: Kaiming 均匀初始化
        var aData = TensorRandom.KaimingUniform(inDim, rank * inDim, rng);
        _a = new Tensor(aData, [inDim, rank], requiresGrad: true);
        RegisterParameter("A", _a);

        // B: 零初始化（确保初始时 LoRA 输出为 0）
        var bData = new float[rank * outDim];
        _b = new Tensor(bData, [rank, outDim], requiresGrad: true);
        RegisterParameter("B", _b);
    }

    /// <summary>
    /// 前向传播: (alpha / rank) * x @ A @ B
    /// x: [..., inDim]
    /// output: [..., outDim]
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        // input: [..., inDim]
        int lastDim = input.Shape[^1];
        int batchSize = input.Size / lastDim;

        var flatInput = input.Reshape(batchSize, lastDim);

        // x @ A → [batch, rank]
        var xA = TensorOps.MatMul(flatInput, _a);

        // (x @ A) @ B → [batch, outDim]
        var xAB = TensorOps.MatMul(xA, _b);

        // 缩放
        float scale = _alpha / _rank;
        var result = TensorOps.MulScalar(xAB, scale);

        // 恢复原始批量维度
        var outputShape = new int[input.Ndim];
        Array.Copy(input.Shape, outputShape, input.Ndim - 1);
        outputShape[^1] = _b.Shape[1]; // outDim
        result = result.Reshape(outputShape);

        return result;
    }
}
