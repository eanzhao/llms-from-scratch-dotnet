using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter05.Pretraining;

/// <summary>
/// GPT-2 预训练权重加载器
/// 加载由 tools/convert_gpt2_weights.py 导出的二进制权重文件
///
/// 使用方法:
///   1. 运行 Python 脚本: python tools/convert_gpt2_weights.py --model_size 124M
///   2. C#: Gpt2WeightLoader.Load("gpt2-weights/gpt2-124M.bin", model);
///
/// 二进制格式:
///   [int32: paramCount]
///   for each param:
///     [int32: nameLen][utf8: name]
///     [int32: ndim][int32 * ndim: shape]
///     [float32 * size: data]
/// </summary>
public static class Gpt2WeightLoader
{
    /// <summary>
    /// 从二进制文件加载 GPT-2 权重到 GptModel
    /// </summary>
    /// <param name="binPath">权重文件路径 (gpt2-{size}.bin)</param>
    /// <param name="model">目标 GptModel 实例</param>
    /// <returns>加载的参数数量</returns>
    public static int Load(string binPath, GptModel model)
    {
        // 读取二进制文件中的命名参数
        var fileParams = ReadBinaryWeights(binPath);

        // 获取模型的命名参数
        var modelParams = model.NamedParameters().ToDictionary(p => p.Name, p => p.Param);

        int loaded = 0;
        int skipped = 0;

        foreach (var (name, shape, data) in fileParams)
        {
            if (modelParams.TryGetValue(name, out var param))
            {
                // 验证形状匹配
                if (!ShapeMatch(param.Shape, shape))
                {
                    Console.WriteLine($"  [警告] 形状不匹配: {name} " +
                        $"模型=[{string.Join(",", param.Shape)}] " +
                        $"文件=[{string.Join(",", shape)}]");
                    skipped++;
                    continue;
                }

                // 复制数据
                Array.Copy(data, param.Data, data.Length);
                loaded++;
            }
            else
            {
                // 尝试模糊匹配（处理命名差异）
                var match = FindBestMatch(name, modelParams.Keys);
                if (match != null && modelParams.TryGetValue(match, out var matchParam))
                {
                    if (ShapeMatch(matchParam.Shape, shape))
                    {
                        Array.Copy(data, matchParam.Data, data.Length);
                        loaded++;
                        continue;
                    }
                }
                skipped++;
            }
        }

        Console.WriteLine($"GPT-2 权重加载完成: {loaded} 个参数已加载, {skipped} 个跳过");
        return loaded;
    }

    /// <summary>读取二进制权重文件</summary>
    private static List<(string Name, int[] Shape, float[] Data)> ReadBinaryWeights(string path)
    {
        var result = new List<(string, int[], float[])>();

        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);

        int paramCount = reader.ReadInt32();

        for (int p = 0; p < paramCount; p++)
        {
            // 读取名称
            int nameLen = reader.ReadInt32();
            string name = System.Text.Encoding.UTF8.GetString(reader.ReadBytes(nameLen));

            // 读取形状
            int ndim = reader.ReadInt32();
            var shape = new int[ndim];
            int size = 1;
            for (int d = 0; d < ndim; d++)
            {
                shape[d] = reader.ReadInt32();
                size *= shape[d];
            }

            // 读取数据
            var bytes = reader.ReadBytes(size * sizeof(float));
            var data = new float[size];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);

            result.Add((name, shape, data));
        }

        return result;
    }

    private static bool ShapeMatch(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>简单的模糊匹配（处理命名约定差异）</summary>
    private static string? FindBestMatch(string name, IEnumerable<string> candidates)
    {
        // 尝试去掉或添加前缀的匹配
        foreach (var candidate in candidates)
        {
            if (candidate.EndsWith(name) || name.EndsWith(candidate))
                return candidate;
        }
        return null;
    }
}
