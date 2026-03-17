using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter02.TextData;

/// <summary>
/// GPT 数据集 V1 - 使用滑动窗口将文本切分为重叠的训练样本
/// 对应 Python 版 GPTDatasetV1
///
/// 工作原理:
/// 给定一个 token ID 序列，以 stride 为步长、maxLength 为窗口大小滑动，
/// 每个窗口产生 (input, target) 对，其中 target 是 input 右移一位的结果。
///
/// 例如 token=[0,1,2,3,4,5], maxLength=4, stride=2:
///   input=[0,1,2,3], target=[1,2,3,4]
///   input=[2,3,4,5], target=[3,4,5,6] (如果有的话)
/// </summary>
public class GptDatasetV1
{
    public List<Tensor> InputIds { get; } = new();
    public List<Tensor> TargetIds { get; } = new();
    public int Count => InputIds.Count;

    /// <summary>
    /// 从 token ID 数组构建数据集
    /// </summary>
    /// <param name="tokenIds">完整文本的 token ID 序列</param>
    /// <param name="maxLength">每个样本的序列长度（上下文窗口大小）</param>
    /// <param name="stride">滑动步长</param>
    public GptDatasetV1(int[] tokenIds, int maxLength, int stride)
    {
        for (int i = 0; i <= tokenIds.Length - maxLength - 1; i += stride)
        {
            var inputChunk = new float[maxLength];
            var targetChunk = new float[maxLength];

            for (int j = 0; j < maxLength; j++)
            {
                inputChunk[j] = tokenIds[i + j];
                targetChunk[j] = tokenIds[i + j + 1];
            }

            InputIds.Add(new Tensor(inputChunk, [maxLength]));
            TargetIds.Add(new Tensor(targetChunk, [maxLength]));
        }
    }

    /// <summary>获取第 idx 个样本</summary>
    public (Tensor input, Tensor target) GetItem(int idx)
    {
        return (InputIds[idx], TargetIds[idx]);
    }
}
