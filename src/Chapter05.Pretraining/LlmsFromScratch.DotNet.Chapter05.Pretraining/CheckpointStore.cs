using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.IO;

namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// 检查点存储器 - 保存和加载模型训练状态
/// 对应 Python 版 torch.save / torch.load
/// </summary>
public static class CheckpointStore
{
    /// <summary>保存模型参数到文件</summary>
    public static void Save(GptModel model, string path)
    {
        ModelSerializer.Save(model, path);
        Console.WriteLine($"模型已保存到: {path}");
    }

    /// <summary>从文件加载模型参数</summary>
    public static void Load(GptModel model, string path)
    {
        ModelSerializer.Load(model, path);
        Console.WriteLine($"模型已从 {path} 加载");
    }
}
