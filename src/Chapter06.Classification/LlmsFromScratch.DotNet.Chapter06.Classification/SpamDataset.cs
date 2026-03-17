using LlmsFromScratch.DotNet.Shared.Tensors;
using LlmsFromScratch.DotNet.Shared.Tokenization;

namespace LlmsFromScratch.DotNet.Chapter06.Classification;

/// <summary>
/// 垃圾邮件分类数据集（对应 Python 版 SpamDataset）
///
/// 加载 CSV 格式的短信数据，将文本编码为 token ID 序列，
/// 统一填充/截断到固定长度，返回 (encodedText, label) 对。
/// </summary>
public class SpamDataset
{
    public List<(Tensor encodedText, Tensor label)> Samples { get; } = new();
    public int Count => Samples.Count;

    /// <summary>
    /// 从 (text, label) 列表构建数据集
    /// </summary>
    /// <param name="texts">文本列表</param>
    /// <param name="labels">标签列表 (0=正常, 1=垃圾)</param>
    /// <param name="tokenizer">分词器</param>
    /// <param name="maxLength">最大序列长度</param>
    /// <param name="padTokenId">填充 token 的 ID</param>
    public SpamDataset(List<string> texts, List<int> labels, SimpleTokenizer tokenizer,
        int maxLength, int padTokenId = 0)
    {
        for (int i = 0; i < texts.Count; i++)
        {
            var tokenIds = tokenizer.Encode(texts[i]);

            // 截断到 maxLength
            int actualLen = Math.Min(tokenIds.Length, maxLength);
            var padded = new float[maxLength];
            for (int j = 0; j < actualLen; j++)
                padded[j] = tokenIds[j];
            // 填充
            for (int j = actualLen; j < maxLength; j++)
                padded[j] = padTokenId;

            var encoded = new Tensor(padded, [maxLength]);
            var label = Tensor.FromScalar(labels[i]);

            Samples.Add((encoded, label));
        }
    }

    public (Tensor input, Tensor label) GetItem(int idx) => Samples[idx];
}
