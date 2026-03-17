using LlmsFromScratch.DotNet.Shared.Nn;

namespace LlmsFromScratch.DotNet.Shared.IO;

/// <summary>
/// 模型序列化器 - 保存和加载模型参数
/// 使用简单的二进制格式: [参数名长度][参数名UTF8][维度数][各维度大小][float数据]
/// </summary>
public static class ModelSerializer
{
    /// <summary>将模型参数保存到文件</summary>
    public static void Save(Module model, string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        var namedParams = model.NamedParameters().ToList();
        writer.Write(namedParams.Count); // 参数数量

        foreach (var (name, param) in namedParams)
        {
            // 写入参数名
            var nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);

            // 写入形状
            writer.Write(param.Shape.Length);
            foreach (int dim in param.Shape)
                writer.Write(dim);

            // 写入数据
            foreach (float val in param.Data)
                writer.Write(val);
        }
    }

    /// <summary>从文件加载模型参数</summary>
    public static void Load(Module model, string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        int paramCount = reader.ReadInt32();
        var modelParams = model.NamedParameters().ToDictionary(p => p.Name, p => p.Param);

        for (int p = 0; p < paramCount; p++)
        {
            // 读取参数名
            int nameLen = reader.ReadInt32();
            var nameBytes = reader.ReadBytes(nameLen);
            string name = System.Text.Encoding.UTF8.GetString(nameBytes);

            // 读取形状
            int ndim = reader.ReadInt32();
            int size = 1;
            for (int d = 0; d < ndim; d++)
            {
                int dimSize = reader.ReadInt32();
                size *= dimSize;
            }

            // 读取数据
            var data = new float[size];
            for (int i = 0; i < size; i++)
                data[i] = reader.ReadSingle();

            // 写入对应参数
            if (modelParams.TryGetValue(name, out var param))
            {
                if (param.Size != size)
                    throw new InvalidOperationException(
                        $"参数 '{name}' 大小不匹配: 文件中 {size}, 模型中 {param.Size}");
                Array.Copy(data, param.Data, size);
            }
        }
    }
}
