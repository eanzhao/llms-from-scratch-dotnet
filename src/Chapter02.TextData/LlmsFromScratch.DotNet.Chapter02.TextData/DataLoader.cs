using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter02.TextData;

/// <summary>
/// 数据加载器 - 将数据集切分为小批量（mini-batch）
/// 对应 PyTorch 的 DataLoader
///
/// 功能:
/// - 按 batchSize 分批
/// - 可选打乱顺序
/// - 可选丢弃最后不完整的批次
/// </summary>
public class DataLoader
{
    private readonly GptDatasetV1 _dataset;
    private readonly int _batchSize;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly Random _rng;

    public DataLoader(GptDatasetV1 dataset, int batchSize, bool shuffle = true, bool dropLast = true, Random? rng = null)
    {
        _dataset = dataset;
        _batchSize = batchSize;
        _shuffle = shuffle;
        _dropLast = dropLast;
        _rng = rng ?? new Random();
    }

    /// <summary>批次数量</summary>
    public int BatchCount
    {
        get
        {
            int full = _dataset.Count / _batchSize;
            int remainder = _dataset.Count % _batchSize;
            return _dropLast ? full : full + (remainder > 0 ? 1 : 0);
        }
    }

    /// <summary>迭代所有批次，每个批次返回 (inputBatch, targetBatch)</summary>
    public IEnumerable<(Tensor inputs, Tensor targets)> GetBatches()
    {
        // 创建索引列表
        var indices = Enumerable.Range(0, _dataset.Count).ToArray();
        if (_shuffle)
        {
            // Fisher-Yates 洗牌
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = _rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int seqLen = _dataset.InputIds[0].Shape[0];

        for (int start = 0; start + _batchSize <= indices.Length; start += _batchSize)
        {
            int actualBatch = Math.Min(_batchSize, indices.Length - start);
            if (actualBatch < _batchSize && _dropLast)
                break;

            // 将多个样本堆叠为批次
            var inputData = new float[actualBatch * seqLen];
            var targetData = new float[actualBatch * seqLen];

            for (int b = 0; b < actualBatch; b++)
            {
                var (inp, tgt) = _dataset.GetItem(indices[start + b]);
                Array.Copy(inp.Data, 0, inputData, b * seqLen, seqLen);
                Array.Copy(tgt.Data, 0, targetData, b * seqLen, seqLen);
            }

            yield return (
                new Tensor(inputData, [actualBatch, seqLen]),
                new Tensor(targetData, [actualBatch, seqLen])
            );
        }
    }

    /// <summary>创建数据加载器的便捷方法（对应 Python 版 create_dataloader_v1）</summary>
    public static DataLoader Create(
        string text,
        Shared.Tokenization.SimpleTokenizer tokenizer,
        int batchSize = 4,
        int maxLength = 256,
        int stride = 128,
        bool shuffle = true,
        bool dropLast = true)
    {
        var tokenIds = tokenizer.Encode(text);
        var dataset = new GptDatasetV1(tokenIds, maxLength, stride);
        return new DataLoader(dataset, batchSize, shuffle, dropLast);
    }
}
