namespace LlmsFromScratch.DotNet.Shared.Tensors;

/// <summary>
/// 张量随机数生成器
/// 使用 Box-Muller 变换生成正态分布随机数
/// </summary>
public static class TensorRandom
{
    private static Random _defaultRng = new(42);

    public static void SetSeed(int seed)
    {
        _defaultRng = new Random(seed);
    }

    /// <summary>生成正态分布 N(0,1) 的浮点数组</summary>
    public static float[] NormalArray(int count, Random? rng = null)
    {
        rng ??= _defaultRng;
        var result = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            // Box-Muller 变换
            double u1 = 1.0 - rng.NextDouble(); // 避免 log(0)
            double u2 = rng.NextDouble();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            double z0 = mag * Math.Cos(2.0 * Math.PI * u2);
            double z1 = mag * Math.Sin(2.0 * Math.PI * u2);

            result[i] = (float)z0;
            if (i + 1 < count)
                result[i + 1] = (float)z1;
        }
        return result;
    }

    /// <summary>生成 [0, 1) 均匀分布的浮点数组</summary>
    public static float[] UniformArray(int count, Random? rng = null)
    {
        rng ??= _defaultRng;
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = (float)rng.NextDouble();
        return result;
    }

    /// <summary>使用 Kaiming 均匀初始化（适用于 ReLU/GELU 激活前的线性层）</summary>
    public static float[] KaimingUniform(int fanIn, int count, Random? rng = null)
    {
        rng ??= _defaultRng;
        float limit = MathF.Sqrt(1.0f / fanIn);
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = (float)(rng.NextDouble() * 2 * limit - limit);
        return result;
    }
}
