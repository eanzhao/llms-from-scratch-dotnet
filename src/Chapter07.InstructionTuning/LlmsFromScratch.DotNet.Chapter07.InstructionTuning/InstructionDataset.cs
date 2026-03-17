using LlmsFromScratch.DotNet.Shared.Tensors;
using LlmsFromScratch.DotNet.Shared.Tokenization;

namespace LlmsFromScratch.DotNet.Chapter07.InstructionTuning;

/// <summary>
/// 指令数据集（对应 Python 版 InstructionDataset）
///
/// 将 InstructionSample 列表预编码为 token ID 序列。
/// 与分类数据集不同，指令数据集的序列长度可变，
/// 填充和裁剪在 InstructionCollator 中处理。
/// </summary>
public class InstructionDataset
{
    public List<int[]> EncodedTexts { get; } = new();
    public int Count => EncodedTexts.Count;

    public InstructionDataset(List<InstructionSample> samples, SimpleTokenizer tokenizer)
    {
        foreach (var sample in samples)
        {
            string fullText = PromptTemplate.FormatFull(sample);
            var encoded = tokenizer.Encode(fullText);
            EncodedTexts.Add(encoded);
        }
    }

    public int[] GetItem(int idx) => EncodedTexts[idx];
}
