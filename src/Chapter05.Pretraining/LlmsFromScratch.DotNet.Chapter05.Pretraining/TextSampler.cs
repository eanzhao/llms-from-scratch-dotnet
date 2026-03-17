using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Tensors;
using LlmsFromScratch.DotNet.Shared.Tokenization;

namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// 文本采样器 - 训练过程中定期生成文本以监控模型质量
/// 对应 Python 版 generate_and_print_sample
/// </summary>
public static class TextSampler
{
    /// <summary>将文本编码为带 batch 维度的 tensor</summary>
    public static Tensor TextToTokenIds(string text, SimpleTokenizer tokenizer)
    {
        var ids = tokenizer.Encode(text);
        var data = new float[ids.Length];
        for (int i = 0; i < ids.Length; i++)
            data[i] = ids[i];
        return new Tensor(data, [1, ids.Length]); // [1, seqLen]
    }

    /// <summary>将 tensor 解码为文本</summary>
    public static string TokenIdsToText(Tensor tokenIds, SimpleTokenizer tokenizer)
    {
        var ids = new int[tokenIds.Shape[^1]]; // 取最后一维
        int offset = tokenIds.Size - ids.Length; // 取最后一个样本
        for (int i = 0; i < ids.Length; i++)
            ids[i] = (int)tokenIds.Data[offset + i];
        return tokenizer.Decode(ids);
    }

    /// <summary>生成文本并打印</summary>
    public static void GenerateAndPrint(GptModel model, SimpleTokenizer tokenizer,
        string startContext, int maxNewTokens = 50)
    {
        model.SetTraining(false);
        int contextSize = model.PosEmbWeight.Shape[0];
        var encoded = TextToTokenIds(startContext, tokenizer);

        var tokenIds = TextGenerator.GenerateSimple(model, encoded, maxNewTokens, contextSize);
        var text = TokenIdsToText(tokenIds, tokenizer);

        Console.WriteLine(text.Replace("\n", " "));
        model.SetTraining(true);
    }
}
