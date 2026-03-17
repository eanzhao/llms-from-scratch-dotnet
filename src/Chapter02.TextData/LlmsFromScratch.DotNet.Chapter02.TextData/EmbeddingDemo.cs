using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter02.TextData;

/// <summary>
/// 嵌入演示 - 展示 token 嵌入 + 位置嵌入的工作原理
/// 对应书中 Chapter 2 末尾的嵌入概念
///
/// token_embeddings = tok_emb(token_ids)      # [batch, seq, emb_dim]
/// position_embeddings = pos_emb(positions)   # [seq, emb_dim] -> 广播
/// embeddings = token_embeddings + position_embeddings
/// </summary>
public static class EmbeddingDemo
{
    /// <summary>
    /// 演示嵌入流程
    /// </summary>
    public static void Run(int vocabSize = 50257, int contextLength = 1024, int embDim = 768)
    {
        Console.WriteLine("═══ 嵌入层演示 ═══");
        Console.WriteLine($"词汇表大小: {vocabSize}");
        Console.WriteLine($"上下文长度: {contextLength}");
        Console.WriteLine($"嵌入维度: {embDim}");

        // 创建嵌入层
        var tokEmb = new Embedding(vocabSize, embDim);
        var posEmb = new Embedding(contextLength, embDim);

        // 模拟一个小批次: 2个样本，每个4个token
        var tokenIds = Tensor.FromArray(
            [0f, 1f, 2f, 3f, 0f, 1f, 2f, 3f],
            [2, 4]);

        Console.WriteLine($"\n输入 token IDs 形状: {tokenIds}");

        // Token 嵌入
        var tokEmbeddings = tokEmb.Forward(tokenIds);
        Console.WriteLine($"Token 嵌入形状: {tokEmbeddings}");

        // 位置嵌入（位置 0, 1, 2, 3）
        var positions = Tensor.Arange(4);
        var posEmbeddings = posEmb.Forward(positions);
        Console.WriteLine($"位置嵌入形状: {posEmbeddings}");

        // 相加（位置嵌入广播到 batch 维度）
        var embeddings = TensorOps.Add(tokEmbeddings, posEmbeddings);
        Console.WriteLine($"最终嵌入形状: {embeddings}");
        Console.WriteLine($"嵌入前几个值: [{embeddings.Data[0]:F4}, {embeddings.Data[1]:F4}, {embeddings.Data[2]:F4}, ...]");
    }
}
