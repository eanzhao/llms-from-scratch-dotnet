namespace LlmsFromScratch.DotNet.Shared.Tensors;

/// <summary>
/// NumPy 风格广播辅助器
/// 规则：从尾部对齐维度，size-1 的维度可以扩展到任意大小
/// </summary>
internal static class BroadcastHelper
{
    /// <summary>
    /// 计算两个形状广播后的结果形状
    /// 例如: [3, 1, 5] 和 [4, 5] => [3, 4, 5]
    /// </summary>
    public static int[] BroadcastShape(int[] shapeA, int[] shapeB)
    {
        int maxNdim = Math.Max(shapeA.Length, shapeB.Length);
        var result = new int[maxNdim];

        for (int i = 0; i < maxNdim; i++)
        {
            int dimA = i < shapeA.Length ? shapeA[shapeA.Length - 1 - i] : 1;
            int dimB = i < shapeB.Length ? shapeB[shapeB.Length - 1 - i] : 1;

            if (dimA == dimB)
                result[maxNdim - 1 - i] = dimA;
            else if (dimA == 1)
                result[maxNdim - 1 - i] = dimB;
            else if (dimB == 1)
                result[maxNdim - 1 - i] = dimA;
            else
                throw new ArgumentException(
                    $"形状不兼容: {Tensor.ShapeToString(shapeA)} 和 {Tensor.ShapeToString(shapeB)}");
        }

        return result;
    }

    /// <summary>
    /// 将逻辑多维索引（广播后的结果索引）映射到原始张量的一维偏移
    /// 对于 size=1 的维度，索引始终为 0（实现广播语义）
    /// </summary>
    public static int MapIndex(int[] resultIndices, int[] originalShape, int[] originalStrides, int resultNdim)
    {
        int offset = 0;
        int shapeDiff = resultNdim - originalShape.Length;

        for (int i = 0; i < originalShape.Length; i++)
        {
            int resultIdx = resultIndices[i + shapeDiff];
            // 广播：如果原始维度为 1，则索引固定为 0
            int idx = originalShape[i] == 1 ? 0 : resultIdx;
            offset += idx * originalStrides[i];
        }

        return offset;
    }

    /// <summary>
    /// 对逐元素运算执行广播：遍历结果形状的每个元素，
    /// 从两个输入张量中取值，应用运算函数
    /// </summary>
    public static float[] BroadcastElementwise(
        Tensor a, Tensor b, int[] resultShape, Func<float, float, float> op)
    {
        int resultSize = Tensor.ComputeSize(resultShape);
        var resultData = new float[resultSize];
        var resultStrides = Tensor.ComputeStrides(resultShape);
        var indices = new int[resultShape.Length];

        for (int flatIdx = 0; flatIdx < resultSize; flatIdx++)
        {
            // 将平坦索引转为多维索引
            UnravelIndex(flatIdx, resultStrides, indices);

            int offsetA = MapIndex(indices, a.Shape, a.Strides, resultShape.Length);
            int offsetB = MapIndex(indices, b.Shape, b.Strides, resultShape.Length);

            resultData[flatIdx] = op(a.Data[offsetA], b.Data[offsetB]);
        }

        return resultData;
    }

    /// <summary>将一维平坦索引转换为多维索引</summary>
    public static void UnravelIndex(int flatIndex, int[] strides, int[] outIndices)
    {
        int remaining = flatIndex;
        for (int i = 0; i < strides.Length; i++)
        {
            outIndices[i] = remaining / strides[i];
            remaining %= strides[i];
        }
    }

    /// <summary>
    /// 累加反向传播中的梯度：将广播后形状的梯度
    /// 归约回原始形状（对被广播的维度求和）
    /// </summary>
    public static void AccumulateGrad(float[] sourceGrad, int[] sourceShape,
        float[] targetGrad, int[] targetShape)
    {
        int sourceSize = Tensor.ComputeSize(sourceShape);
        var sourceStrides = Tensor.ComputeStrides(sourceShape);
        var targetStrides = Tensor.ComputeStrides(targetShape);
        var indices = new int[sourceShape.Length];

        for (int flatIdx = 0; flatIdx < sourceSize; flatIdx++)
        {
            UnravelIndex(flatIdx, sourceStrides, indices);
            int targetOffset = MapIndex(indices, targetShape, targetStrides, sourceShape.Length);
            targetGrad[targetOffset] += sourceGrad[flatIdx];
        }
    }
}
