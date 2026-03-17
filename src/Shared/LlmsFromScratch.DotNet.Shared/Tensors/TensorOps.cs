namespace LlmsFromScratch.DotNet.Shared.Tensors;

/// <summary>
/// 张量运算 - 所有运算都支持自动微分（反向传播）
/// 每个运算在创建输出张量时注册 BackwardFn 闭包
/// </summary>
public static class TensorOps
{
    // ═══════════════════════════════════════════
    //  逐元素运算（支持广播）
    // ═══════════════════════════════════════════

    /// <summary>逐元素加法: C = A + B</summary>
    public static Tensor Add(Tensor a, Tensor b)
    {
        var resultShape = BroadcastHelper.BroadcastShape(a.Shape, b.Shape);
        var resultData = BroadcastHelper.BroadcastElementwise(a, b, resultShape, (x, y) => x + y);
        bool needsGrad = a.RequiresGrad || b.RequiresGrad;
        var result = new Tensor(resultData, resultShape, needsGrad);

        if (needsGrad)
        {
            result.Parents = [a, b];
            result.BackwardFn = () =>
            {
                if (a.RequiresGrad)
                {
                    Tensor.EnsureGrad(a);
                    BroadcastHelper.AccumulateGrad(result.Grad!, resultShape, a.Grad!, a.Shape);
                }
                if (b.RequiresGrad)
                {
                    Tensor.EnsureGrad(b);
                    BroadcastHelper.AccumulateGrad(result.Grad!, resultShape, b.Grad!, b.Shape);
                }
            };
        }

        return result;
    }

    /// <summary>逐元素减法: C = A - B</summary>
    public static Tensor Sub(Tensor a, Tensor b)
    {
        var resultShape = BroadcastHelper.BroadcastShape(a.Shape, b.Shape);
        var resultData = BroadcastHelper.BroadcastElementwise(a, b, resultShape, (x, y) => x - y);
        bool needsGrad = a.RequiresGrad || b.RequiresGrad;
        var result = new Tensor(resultData, resultShape, needsGrad);

        if (needsGrad)
        {
            result.Parents = [a, b];
            result.BackwardFn = () =>
            {
                if (a.RequiresGrad)
                {
                    Tensor.EnsureGrad(a);
                    BroadcastHelper.AccumulateGrad(result.Grad!, resultShape, a.Grad!, a.Shape);
                }
                if (b.RequiresGrad)
                {
                    Tensor.EnsureGrad(b);
                    // 减法对 b 的梯度取反
                    var negGrad = new float[result.Size];
                    for (int i = 0; i < result.Size; i++)
                        negGrad[i] = -result.Grad![i];
                    BroadcastHelper.AccumulateGrad(negGrad, resultShape, b.Grad!, b.Shape);
                }
            };
        }

        return result;
    }

    /// <summary>逐元素乘法: C = A * B (Hadamard product)</summary>
    public static Tensor Mul(Tensor a, Tensor b)
    {
        var resultShape = BroadcastHelper.BroadcastShape(a.Shape, b.Shape);
        var resultData = BroadcastHelper.BroadcastElementwise(a, b, resultShape, (x, y) => x * y);
        bool needsGrad = a.RequiresGrad || b.RequiresGrad;
        var result = new Tensor(resultData, resultShape, needsGrad);

        if (needsGrad)
        {
            result.Parents = [a, b];
            result.BackwardFn = () =>
            {
                var resultStrides = Tensor.ComputeStrides(resultShape);
                var indices = new int[resultShape.Length];

                if (a.RequiresGrad)
                {
                    Tensor.EnsureGrad(a);
                    // dA = dC * B
                    for (int i = 0; i < result.Size; i++)
                    {
                        BroadcastHelper.UnravelIndex(i, resultStrides, indices);
                        int offsetB = BroadcastHelper.MapIndex(indices, b.Shape, b.Strides, resultShape.Length);
                        int offsetA = BroadcastHelper.MapIndex(indices, a.Shape, a.Strides, resultShape.Length);
                        a.Grad![offsetA] += result.Grad![i] * b.Data[offsetB];
                    }
                }
                if (b.RequiresGrad)
                {
                    Tensor.EnsureGrad(b);
                    // dB = dC * A
                    for (int i = 0; i < result.Size; i++)
                    {
                        BroadcastHelper.UnravelIndex(i, resultStrides, indices);
                        int offsetA = BroadcastHelper.MapIndex(indices, a.Shape, a.Strides, resultShape.Length);
                        int offsetB = BroadcastHelper.MapIndex(indices, b.Shape, b.Strides, resultShape.Length);
                        b.Grad![offsetB] += result.Grad![i] * a.Data[offsetA];
                    }
                }
            };
        }

        return result;
    }

    /// <summary>逐元素除法: C = A / B</summary>
    public static Tensor Div(Tensor a, Tensor b)
    {
        var resultShape = BroadcastHelper.BroadcastShape(a.Shape, b.Shape);
        var resultData = BroadcastHelper.BroadcastElementwise(a, b, resultShape, (x, y) => x / y);
        bool needsGrad = a.RequiresGrad || b.RequiresGrad;
        var result = new Tensor(resultData, resultShape, needsGrad);

        if (needsGrad)
        {
            result.Parents = [a, b];
            result.BackwardFn = () =>
            {
                var resultStrides = Tensor.ComputeStrides(resultShape);
                var indices = new int[resultShape.Length];

                if (a.RequiresGrad)
                {
                    Tensor.EnsureGrad(a);
                    for (int i = 0; i < result.Size; i++)
                    {
                        BroadcastHelper.UnravelIndex(i, resultStrides, indices);
                        int offsetB = BroadcastHelper.MapIndex(indices, b.Shape, b.Strides, resultShape.Length);
                        int offsetA = BroadcastHelper.MapIndex(indices, a.Shape, a.Strides, resultShape.Length);
                        a.Grad![offsetA] += result.Grad![i] / b.Data[offsetB];
                    }
                }
                if (b.RequiresGrad)
                {
                    Tensor.EnsureGrad(b);
                    for (int i = 0; i < result.Size; i++)
                    {
                        BroadcastHelper.UnravelIndex(i, resultStrides, indices);
                        int offsetA = BroadcastHelper.MapIndex(indices, a.Shape, a.Strides, resultShape.Length);
                        int offsetB = BroadcastHelper.MapIndex(indices, b.Shape, b.Strides, resultShape.Length);
                        // d/dB (A/B) = -A/B^2
                        b.Grad![offsetB] += result.Grad![i] * (-a.Data[offsetA] / (b.Data[offsetB] * b.Data[offsetB]));
                    }
                }
            };
        }

        return result;
    }

    /// <summary>标量乘法: C = A * scalar</summary>
    public static Tensor MulScalar(Tensor a, float scalar)
    {
        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
            resultData[i] = a.Data[i] * scalar;

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                for (int i = 0; i < a.Size; i++)
                    a.Grad![i] += result.Grad![i] * scalar;
            };
        }

        return result;
    }

    // ═══════════════════════════════════════════
    //  矩阵运算
    // ═══════════════════════════════════════════

    /// <summary>
    /// 矩阵乘法: C = A @ B
    /// 支持批量矩阵乘法: [..., M, K] @ [..., K, N] => [..., M, N]
    /// </summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Ndim < 2 || b.Ndim < 2)
            throw new ArgumentException("MatMul 需要至少 2 维张量");

        int M = a.Shape[^2]; // 倒数第二维
        int K = a.Shape[^1]; // 最后一维
        int N = b.Shape[^1]; // b 的最后一维

        if (b.Shape[^2] != K)
            throw new ArgumentException($"MatMul 形状不匹配: {Tensor.ShapeToString(a.Shape)} 和 {Tensor.ShapeToString(b.Shape)}");

        // 计算批量维度
        var batchDimsA = a.Shape[..^2];
        var batchDimsB = b.Shape[..^2];
        var batchShape = BroadcastHelper.BroadcastShape(
            batchDimsA.Length > 0 ? batchDimsA : [1],
            batchDimsB.Length > 0 ? batchDimsB : [1]);

        // 结果形状
        var resultShape = new int[batchShape.Length + 2];
        Array.Copy(batchShape, resultShape, batchShape.Length);
        resultShape[^2] = M;
        resultShape[^1] = N;

        int batchSize = Tensor.ComputeSize(batchShape);
        int resultSize = batchSize * M * N;
        var resultData = new float[resultSize];

        var batchStrides = Tensor.ComputeStrides(batchShape);
        var batchIndices = new int[batchShape.Length];

        var batchStridesA = a.Ndim > 2 ? Tensor.ComputeStrides(batchDimsA) : [];
        var batchStridesB = b.Ndim > 2 ? Tensor.ComputeStrides(batchDimsB) : [];

        int aMatStride = M * K; // 每个批次中 A 矩阵的元素数
        int bMatStride = K * N;
        int cMatStride = M * N;

        for (int batch = 0; batch < batchSize; batch++)
        {
            BroadcastHelper.UnravelIndex(batch, batchStrides, batchIndices);

            // 映射到 A 和 B 的批次偏移
            int aOffset = 0;
            int bOffset = 0;

            if (batchDimsA.Length > 0)
            {
                for (int d = 0; d < batchDimsA.Length; d++)
                {
                    int idx = batchIndices[d + batchShape.Length - batchDimsA.Length];
                    idx = batchDimsA[d] == 1 ? 0 : idx;
                    aOffset += idx * batchStridesA[d];
                }
                aOffset *= aMatStride;
            }

            if (batchDimsB.Length > 0)
            {
                for (int d = 0; d < batchDimsB.Length; d++)
                {
                    int idx = batchIndices[d + batchShape.Length - batchDimsB.Length];
                    idx = batchDimsB[d] == 1 ? 0 : idx;
                    bOffset += idx * batchStridesB[d];
                }
                bOffset *= bMatStride;
            }

            int cOffset = batch * cMatStride;

            // 执行矩阵乘法: C[i,j] = sum_k A[i,k] * B[k,j]
            for (int i = 0; i < M; i++)
            {
                for (int k = 0; k < K; k++)
                {
                    float aVal = a.Data[aOffset + i * K + k];
                    for (int j = 0; j < N; j++)
                    {
                        resultData[cOffset + i * N + j] += aVal * b.Data[bOffset + k * N + j];
                    }
                }
            }
        }

        bool needsGrad = a.RequiresGrad || b.RequiresGrad;
        var result = new Tensor(resultData, resultShape, needsGrad);

        if (needsGrad)
        {
            result.Parents = [a, b];
            result.BackwardFn = () =>
            {
                // dC 形状: [..., M, N]
                // dA = dC @ B^T  形状: [..., M, K]
                // dB = A^T @ dC  形状: [..., K, N]
                for (int batch2 = 0; batch2 < batchSize; batch2++)
                {
                    BroadcastHelper.UnravelIndex(batch2, batchStrides, batchIndices);

                    int aOff = 0;
                    int bOff = 0;

                    if (batchDimsA.Length > 0)
                    {
                        for (int d = 0; d < batchDimsA.Length; d++)
                        {
                            int idx = batchIndices[d + batchShape.Length - batchDimsA.Length];
                            idx = batchDimsA[d] == 1 ? 0 : idx;
                            aOff += idx * batchStridesA[d];
                        }
                        aOff *= aMatStride;
                    }

                    if (batchDimsB.Length > 0)
                    {
                        for (int d = 0; d < batchDimsB.Length; d++)
                        {
                            int idx = batchIndices[d + batchShape.Length - batchDimsB.Length];
                            idx = batchDimsB[d] == 1 ? 0 : idx;
                            bOff += idx * batchStridesB[d];
                        }
                        bOff *= bMatStride;
                    }

                    int cOff = batch2 * cMatStride;

                    if (a.RequiresGrad)
                    {
                        Tensor.EnsureGrad(a);
                        // dA[i,k] += sum_j dC[i,j] * B[k,j]
                        for (int i = 0; i < M; i++)
                        {
                            for (int j = 0; j < N; j++)
                            {
                                float dCij = result.Grad![cOff + i * N + j];
                                for (int k = 0; k < K; k++)
                                {
                                    a.Grad![aOff + i * K + k] += dCij * b.Data[bOff + k * N + j];
                                }
                            }
                        }
                    }

                    if (b.RequiresGrad)
                    {
                        Tensor.EnsureGrad(b);
                        // dB[k,j] += sum_i A[i,k] * dC[i,j]
                        for (int i = 0; i < M; i++)
                        {
                            for (int k = 0; k < K; k++)
                            {
                                float aVal = a.Data[aOff + i * K + k];
                                for (int j = 0; j < N; j++)
                                {
                                    b.Grad![bOff + k * N + j] += aVal * result.Grad![cOff + i * N + j];
                                }
                            }
                        }
                    }
                }
            };
        }

        return result;
    }

    /// <summary>
    /// 转置指定两个维度
    /// 实际执行数据复制（非视图），以保持连续存储
    /// </summary>
    public static Tensor Transpose(Tensor a, int dim0, int dim1)
    {
        if (dim0 < 0) dim0 += a.Ndim;
        if (dim1 < 0) dim1 += a.Ndim;

        var newShape = (int[])a.Shape.Clone();
        (newShape[dim0], newShape[dim1]) = (newShape[dim1], newShape[dim0]);

        var newData = new float[a.Size];
        var newStrides = Tensor.ComputeStrides(newShape);
        var indices = new int[a.Ndim];

        for (int i = 0; i < a.Size; i++)
        {
            // 旧张量的多维索引
            BroadcastHelper.UnravelIndex(i, a.Strides, indices);
            // 交换维度后的索引
            (indices[dim0], indices[dim1]) = (indices[dim1], indices[dim0]);
            // 新的平坦偏移
            int newOffset = 0;
            for (int d = 0; d < indices.Length; d++)
                newOffset += indices[d] * newStrides[d];
            newData[newOffset] = a.Data[i];
        }

        var result = new Tensor(newData, newShape, a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // 转置的反向传播就是反向转置
                for (int i = 0; i < a.Size; i++)
                {
                    BroadcastHelper.UnravelIndex(i, a.Strides, indices);
                    (indices[dim0], indices[dim1]) = (indices[dim1], indices[dim0]);
                    int newOff = 0;
                    for (int d = 0; d < indices.Length; d++)
                        newOff += indices[d] * newStrides[d];
                    a.Grad![i] += result.Grad![newOff];
                }
            };
        }

        return result;
    }

    // ═══════════════════════════════════════════
    //  归约运算
    // ═══════════════════════════════════════════

    /// <summary>沿指定维度求和</summary>
    public static Tensor Sum(Tensor a, int dim = -1, bool keepdim = false)
    {
        if (dim < 0) dim += a.Ndim;

        var resultShape = new List<int>();
        for (int i = 0; i < a.Ndim; i++)
        {
            if (i == dim)
            {
                if (keepdim) resultShape.Add(1);
            }
            else
            {
                resultShape.Add(a.Shape[i]);
            }
        }
        if (resultShape.Count == 0) resultShape.Add(1);
        var rShape = resultShape.ToArray();

        int resultSize = Tensor.ComputeSize(rShape);
        var resultData = new float[resultSize];
        var resultStrides = Tensor.ComputeStrides(rShape);
        var indices = new int[a.Ndim];

        for (int i = 0; i < a.Size; i++)
        {
            BroadcastHelper.UnravelIndex(i, a.Strides, indices);

            // 构建结果索引（跳过或压缩 dim 维度）
            int rOffset = 0;
            int ri = 0;
            for (int d = 0; d < a.Ndim; d++)
            {
                if (d == dim)
                {
                    if (keepdim) ri++; // keepdim 时跳过（索引为0）
                    continue;
                }
                rOffset += indices[d] * resultStrides[ri];
                ri++;
            }
            resultData[rOffset] += a.Data[i];
        }

        var result = new Tensor(resultData, rShape, a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // Sum 的梯度就是广播回去
                for (int i = 0; i < a.Size; i++)
                {
                    BroadcastHelper.UnravelIndex(i, a.Strides, indices);
                    int rOff = 0;
                    int ri2 = 0;
                    for (int d = 0; d < a.Ndim; d++)
                    {
                        if (d == dim)
                        {
                            if (keepdim) ri2++;
                            continue;
                        }
                        rOff += indices[d] * resultStrides[ri2];
                        ri2++;
                    }
                    a.Grad![i] += result.Grad![rOff];
                }
            };
        }

        return result;
    }

    /// <summary>沿指定维度求均值</summary>
    public static Tensor Mean(Tensor a, int dim = -1, bool keepdim = false)
    {
        if (dim < 0) dim += a.Ndim;
        int n = a.Shape[dim];
        var sumResult = Sum(a, dim, keepdim);
        return MulScalar(sumResult, 1.0f / n);
    }

    /// <summary>沿指定维度求方差（默认无偏校正，与 PyTorch unbiased=False 一致）</summary>
    public static Tensor Variance(Tensor a, int dim = -1, bool keepdim = false)
    {
        if (dim < 0) dim += a.Ndim;
        var mean = Mean(a, dim, keepdim: true);
        var diff = Sub(a, mean); // 广播减去均值
        var sq = Mul(diff, diff);
        var sumSq = Sum(sq, dim, keepdim);
        int n = a.Shape[dim];
        return MulScalar(sumSq, 1.0f / n);
    }

    /// <summary>沿最后一维取 argmax（不需要梯度）</summary>
    public static Tensor Argmax(Tensor a, int dim = -1)
    {
        if (dim < 0) dim += a.Ndim;

        var resultShape = new List<int>();
        for (int i = 0; i < a.Ndim; i++)
        {
            if (i != dim) resultShape.Add(a.Shape[i]);
        }
        if (resultShape.Count == 0) resultShape.Add(1);

        int outerSize = 1, innerSize = 1;
        for (int i = 0; i < dim; i++) outerSize *= a.Shape[i];
        for (int i = dim + 1; i < a.Ndim; i++) innerSize *= a.Shape[i];
        int dimSize = a.Shape[dim];

        var resultData = new float[outerSize * innerSize];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                float maxVal = float.NegativeInfinity;
                int maxIdx = 0;
                for (int d = 0; d < dimSize; d++)
                {
                    int offset = (outer * dimSize + d) * innerSize + inner;
                    if (a.Data[offset] > maxVal)
                    {
                        maxVal = a.Data[offset];
                        maxIdx = d;
                    }
                }
                resultData[outer * innerSize + inner] = maxIdx;
            }
        }

        return new Tensor(resultData, resultShape.ToArray(), false);
    }

    // ═══════════════════════════════════════════
    //  激活函数
    // ═══════════════════════════════════════════

    /// <summary>Tanh 激活: tanh(x)</summary>
    public static Tensor Tanh(Tensor a)
    {
        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
            resultData[i] = MathF.Tanh(a.Data[i]);

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // d/dx tanh(x) = 1 - tanh(x)^2
                for (int i = 0; i < a.Size; i++)
                    a.Grad![i] += result.Grad![i] * (1.0f - resultData[i] * resultData[i]);
            };
        }

        return result;
    }

    /// <summary>指数函数: exp(x)</summary>
    public static Tensor Exp(Tensor a)
    {
        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
            resultData[i] = MathF.Exp(a.Data[i]);

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // d/dx exp(x) = exp(x)
                for (int i = 0; i < a.Size; i++)
                    a.Grad![i] += result.Grad![i] * resultData[i];
            };
        }

        return result;
    }

    /// <summary>幂运算: x^p</summary>
    public static Tensor Pow(Tensor a, float exponent)
    {
        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
            resultData[i] = MathF.Pow(a.Data[i], exponent);

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // d/dx x^p = p * x^(p-1)
                for (int i = 0; i < a.Size; i++)
                    a.Grad![i] += result.Grad![i] * exponent * MathF.Pow(a.Data[i], exponent - 1);
            };
        }

        return result;
    }

    /// <summary>平方根: sqrt(x)</summary>
    public static Tensor Sqrt(Tensor a)
    {
        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
            resultData[i] = MathF.Sqrt(a.Data[i]);

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // d/dx sqrt(x) = 0.5 / sqrt(x)
                for (int i = 0; i < a.Size; i++)
                    a.Grad![i] += result.Grad![i] * 0.5f / resultData[i];
            };
        }

        return result;
    }

    /// <summary>自然对数: log(x)</summary>
    public static Tensor Log(Tensor a)
    {
        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
            resultData[i] = MathF.Log(a.Data[i]);

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // d/dx log(x) = 1/x
                for (int i = 0; i < a.Size; i++)
                    a.Grad![i] += result.Grad![i] / a.Data[i];
            };
        }

        return result;
    }

    // ═══════════════════════════════════════════
    //  高级运算
    // ═══════════════════════════════════════════

    /// <summary>
    /// Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    /// 沿最后一维计算，数值稳定版本（减去最大值防止溢出）
    /// </summary>
    public static Tensor Softmax(Tensor a, int dim = -1)
    {
        if (dim < 0) dim += a.Ndim;

        int outerSize = 1, innerSize = 1;
        for (int i = 0; i < dim; i++) outerSize *= a.Shape[i];
        for (int i = dim + 1; i < a.Ndim; i++) innerSize *= a.Shape[i];
        int dimSize = a.Shape[dim];

        var resultData = new float[a.Size];

        // 对每个 softmax 切片计算
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // 找最大值（数值稳定性）
                float maxVal = float.NegativeInfinity;
                for (int d = 0; d < dimSize; d++)
                {
                    int idx = (outer * dimSize + d) * innerSize + inner;
                    if (a.Data[idx] > maxVal) maxVal = a.Data[idx];
                }

                // 计算 exp(x - max) 并求和
                float sumExp = 0;
                for (int d = 0; d < dimSize; d++)
                {
                    int idx = (outer * dimSize + d) * innerSize + inner;
                    resultData[idx] = MathF.Exp(a.Data[idx] - maxVal);
                    sumExp += resultData[idx];
                }

                // 归一化
                for (int d = 0; d < dimSize; d++)
                {
                    int idx = (outer * dimSize + d) * innerSize + inner;
                    resultData[idx] /= sumExp;
                }
            }
        }

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                // softmax 反向传播:
                // dInput_i = softmax_i * (dOutput_i - sum_j(dOutput_j * softmax_j))
                for (int outer = 0; outer < outerSize; outer++)
                {
                    for (int inner = 0; inner < innerSize; inner++)
                    {
                        // 计算 sum(dOutput * softmax)
                        float dotProduct = 0;
                        for (int d = 0; d < dimSize; d++)
                        {
                            int idx = (outer * dimSize + d) * innerSize + inner;
                            dotProduct += result.Grad![idx] * resultData[idx];
                        }

                        for (int d = 0; d < dimSize; d++)
                        {
                            int idx = (outer * dimSize + d) * innerSize + inner;
                            a.Grad![idx] += resultData[idx] * (result.Grad![idx] - dotProduct);
                        }
                    }
                }
            };
        }

        return result;
    }

    /// <summary>
    /// 掩码填充：将 mask 为 true 的位置填充为指定值
    /// mask 是一个 bool 条件，形状需与张量兼容
    /// </summary>
    public static Tensor MaskedFill(Tensor a, Tensor mask, float value)
    {
        // mask 是 float 张量，非零值视为 true
        var resultData = (float[])a.Data.Clone();

        // 简单实现：mask 和 a 必须形状兼容
        int size = a.Size;
        for (int i = 0; i < size; i++)
        {
            // 对 mask 做广播映射
            var indices = new int[a.Ndim];
            BroadcastHelper.UnravelIndex(i, a.Strides, indices);
            int maskOffset = BroadcastHelper.MapIndex(indices, mask.Shape, mask.Strides, a.Ndim);
            if (mask.Data[maskOffset] != 0)
                resultData[i] = value;
        }

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                for (int i = 0; i < size; i++)
                {
                    var indices2 = new int[a.Ndim];
                    BroadcastHelper.UnravelIndex(i, a.Strides, indices2);
                    int maskOff = BroadcastHelper.MapIndex(indices2, mask.Shape, mask.Strides, a.Ndim);
                    // 被掩码的位置梯度为 0
                    if (mask.Data[maskOff] == 0)
                        a.Grad![i] += result.Grad![i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// 拼接：沿指定维度连接多个张量
    /// </summary>
    public static Tensor Concat(Tensor[] tensors, int dim = 0)
    {
        if (tensors.Length == 0) throw new ArgumentException("至少需要一个张量");
        if (dim < 0) dim += tensors[0].Ndim;

        // 计算结果形状
        var resultShape = (int[])tensors[0].Shape.Clone();
        int totalDimSize = tensors[0].Shape[dim];
        for (int t = 1; t < tensors.Length; t++)
        {
            totalDimSize += tensors[t].Shape[dim];
            for (int d = 0; d < resultShape.Length; d++)
            {
                if (d != dim && tensors[t].Shape[d] != resultShape[d])
                    throw new ArgumentException("非拼接维度的大小必须一致");
            }
        }
        resultShape[dim] = totalDimSize;

        int resultSize = Tensor.ComputeSize(resultShape);
        var resultData = new float[resultSize];
        var resultStrides = Tensor.ComputeStrides(resultShape);

        // 逐个张量复制数据
        int dimOffset = 0;
        for (int t = 0; t < tensors.Length; t++)
        {
            var src = tensors[t];
            var srcIndices = new int[src.Ndim];
            for (int i = 0; i < src.Size; i++)
            {
                BroadcastHelper.UnravelIndex(i, src.Strides, srcIndices);
                var dstIndices = (int[])srcIndices.Clone();
                dstIndices[dim] += dimOffset;
                int dstOffset = 0;
                for (int d = 0; d < dstIndices.Length; d++)
                    dstOffset += dstIndices[d] * resultStrides[d];
                resultData[dstOffset] = src.Data[i];
            }
            dimOffset += src.Shape[dim];
        }

        bool needsGrad = tensors.Any(t => t.RequiresGrad);
        var result = new Tensor(resultData, resultShape, needsGrad);

        if (needsGrad)
        {
            result.Parents = [.. tensors.Where(t => t.RequiresGrad)];
            result.BackwardFn = () =>
            {
                int dOff = 0;
                for (int t = 0; t < tensors.Length; t++)
                {
                    if (tensors[t].RequiresGrad)
                    {
                        Tensor.EnsureGrad(tensors[t]);
                        var sIndices = new int[tensors[t].Ndim];
                        for (int i = 0; i < tensors[t].Size; i++)
                        {
                            BroadcastHelper.UnravelIndex(i, tensors[t].Strides, sIndices);
                            var rIndices = (int[])sIndices.Clone();
                            rIndices[dim] += dOff;
                            int rOff = 0;
                            for (int d = 0; d < rIndices.Length; d++)
                                rOff += rIndices[d] * resultStrides[d];
                            tensors[t].Grad![i] += result.Grad![rOff];
                        }
                    }
                    dOff += tensors[t].Shape[dim];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// 嵌入查找：根据整数索引从权重矩阵中提取行
    /// weight: [vocabSize, embDim], indices: [...] => output: [..., embDim]
    /// </summary>
    public static Tensor Embedding(Tensor weight, Tensor indices)
    {
        int embDim = weight.Shape[1];
        var outputShape = new int[indices.Ndim + 1];
        Array.Copy(indices.Shape, outputShape, indices.Ndim);
        outputShape[^1] = embDim;

        int numIndices = indices.Size;
        var resultData = new float[numIndices * embDim];

        for (int i = 0; i < numIndices; i++)
        {
            int idx = (int)indices.Data[i];
            Array.Copy(weight.Data, idx * embDim, resultData, i * embDim, embDim);
        }

        var result = new Tensor(resultData, outputShape, weight.RequiresGrad);

        if (weight.RequiresGrad)
        {
            result.Parents = [weight];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(weight);
                // scatter-add: 将输出梯度累加回对应行
                for (int i = 0; i < numIndices; i++)
                {
                    int idx = (int)indices.Data[i];
                    for (int j = 0; j < embDim; j++)
                        weight.Grad![idx * embDim + j] += result.Grad![i * embDim + j];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Dropout: 训练时随机置零，推理时直通
    /// 置零的元素按 1/(1-rate) 缩放以保持期望不变
    /// </summary>
    public static Tensor Dropout(Tensor a, float rate, bool training, Random? rng = null)
    {
        if (!training || rate <= 0)
        {
            // 推理模式或不需要 dropout，直接返回副本
            return new Tensor((float[])a.Data.Clone(), (int[])a.Shape.Clone(), a.RequiresGrad)
            {
                Parents = a.RequiresGrad ? [a] : null,
                BackwardFn = a.RequiresGrad ? () =>
                {
                    Tensor.EnsureGrad(a);
                    for (int i = 0; i < a.Size; i++)
                        a.Grad![i] += a.Grad![i]; // identity
                } : null
            };
        }

        rng ??= new Random();
        float scale = 1.0f / (1.0f - rate);
        var mask = new bool[a.Size];
        var resultData = new float[a.Size];

        for (int i = 0; i < a.Size; i++)
        {
            mask[i] = rng.NextDouble() >= rate; // true = 保留
            resultData[i] = mask[i] ? a.Data[i] * scale : 0;
        }

        var result = new Tensor(resultData, (int[])a.Shape.Clone(), a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                for (int i = 0; i < a.Size; i++)
                {
                    if (mask[i])
                        a.Grad![i] += result.Grad![i] * scale;
                }
            };
        }

        return result;
    }

    /// <summary>
    /// 切片操作：提取张量的子区域
    /// 例如: tensor[:, -contextSize:, :] 对应 Slice(tensor, [(.., ..), (-contextSize, ..), (.., ..)])
    /// </summary>
    public static Tensor Slice(Tensor a, (int start, int end)[] ranges)
    {
        if (ranges.Length != a.Ndim)
            throw new ArgumentException("Slice 范围数量必须与张量维度数一致");

        var resolvedRanges = new (int start, int end)[a.Ndim];
        var resultShape = new int[a.Ndim];

        for (int d = 0; d < a.Ndim; d++)
        {
            int s = ranges[d].start;
            int e = ranges[d].end;
            if (s < 0) s += a.Shape[d];
            if (e <= 0) e += a.Shape[d];
            if (e > a.Shape[d]) e = a.Shape[d];
            resolvedRanges[d] = (s, e);
            resultShape[d] = e - s;
        }

        int resultSize = Tensor.ComputeSize(resultShape);
        var resultData = new float[resultSize];
        var resultStrides = Tensor.ComputeStrides(resultShape);
        var indices = new int[a.Ndim];

        for (int i = 0; i < resultSize; i++)
        {
            BroadcastHelper.UnravelIndex(i, resultStrides, indices);
            int srcOffset = 0;
            for (int d = 0; d < a.Ndim; d++)
                srcOffset += (indices[d] + resolvedRanges[d].start) * a.Strides[d];
            resultData[i] = a.Data[srcOffset];
        }

        var result = new Tensor(resultData, resultShape, a.RequiresGrad);

        if (a.RequiresGrad)
        {
            result.Parents = [a];
            result.BackwardFn = () =>
            {
                Tensor.EnsureGrad(a);
                for (int i = 0; i < resultSize; i++)
                {
                    BroadcastHelper.UnravelIndex(i, resultStrides, indices);
                    int srcOff = 0;
                    for (int d = 0; d < a.Ndim; d++)
                        srcOff += (indices[d] + resolvedRanges[d].start) * a.Strides[d];
                    a.Grad![srcOff] += result.Grad![i];
                }
            };
        }

        return result;
    }
}
