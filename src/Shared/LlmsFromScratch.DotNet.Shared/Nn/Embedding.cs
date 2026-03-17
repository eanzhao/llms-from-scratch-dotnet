using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// 嵌入层（对应 PyTorch 的 nn.Embedding）
/// 本质是一个查找表：给定整数索引，返回对应的嵌入向量
/// Weight 形状: [numEmbeddings, embeddingDim]
/// </summary>
public class Embedding : Module
{
    public Tensor Weight { get; }
    public int NumEmbeddings { get; }
    public int EmbeddingDim { get; }

    public Embedding(int numEmbeddings, int embeddingDim, Random? rng = null)
    {
        NumEmbeddings = numEmbeddings;
        EmbeddingDim = embeddingDim;

        var data = TensorRandom.NormalArray(numEmbeddings * embeddingDim, rng);
        Weight = new Tensor(data, [numEmbeddings, embeddingDim], requiresGrad: true);
        RegisterParameter("weight", Weight);
    }

    /// <summary>
    /// 前向传播：根据索引提取嵌入向量
    /// indices 形状: [...] (float 存储的整数索引)
    /// output 形状: [..., embeddingDim]
    /// </summary>
    public override Tensor Forward(Tensor indices)
    {
        return TensorOps.Embedding(Weight, indices);
    }
}
