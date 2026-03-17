namespace LlmsFromScratch.DotNet.Shared.Tensors;

/// <summary>
/// 核心张量类 - 支持自动微分的多维数组
/// 使用一维 float[] 存储数据，通过 Shape 和 Strides 实现多维索引
/// </summary>
public class Tensor
{
    /// <summary>行优先存储的一维浮点数组</summary>
    public float[] Data { get; internal set; }

    /// <summary>张量形状，例如 [2, 3, 768]</summary>
    public int[] Shape { get; internal set; }

    /// <summary>步长数组，用于将多维索引映射到一维偏移</summary>
    public int[] Strides { get; }

    /// <summary>张量维度数</summary>
    public int Ndim => Shape.Length;

    /// <summary>张量元素总数</summary>
    public int Size { get; }

    /// <summary>梯度数组，与 Data 等长，惰性分配</summary>
    public float[]? Grad { get; set; }

    /// <summary>是否需要计算梯度</summary>
    public bool RequiresGrad { get; set; }

    /// <summary>反向传播函数（闭包），由运算注册</summary>
    internal Action? BackwardFn { get; set; }

    /// <summary>计算图中的父节点</summary>
    internal List<Tensor>? Parents { get; set; }

    // ─── 构造函数 ───

    public Tensor(float[] data, int[] shape, bool requiresGrad = false)
    {
        if (shape.Length == 0)
            throw new ArgumentException("Shape 不能为空");

        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize *= shape[i];

        if (data.Length != totalSize)
            throw new ArgumentException($"数据长度 {data.Length} 与形状 {ShapeToString(shape)} 不匹配（期望 {totalSize}）");

        Data = data;
        Shape = shape;
        Size = totalSize;
        Strides = ComputeStrides(shape);
        RequiresGrad = requiresGrad;
    }

    // ─── 工厂方法 ───

    public static Tensor Zeros(params int[] shape)
    {
        int size = ComputeSize(shape);
        return new Tensor(new float[size], (int[])shape.Clone());
    }

    public static Tensor Ones(params int[] shape)
    {
        int size = ComputeSize(shape);
        var data = new float[size];
        Array.Fill(data, 1.0f);
        return new Tensor(data, (int[])shape.Clone());
    }

    public static Tensor Randn(int[] shape, Random? rng = null)
    {
        int size = ComputeSize(shape);
        var data = TensorRandom.NormalArray(size, rng);
        return new Tensor(data, (int[])shape.Clone());
    }

    public static Tensor FromArray(float[] data, int[] shape)
    {
        return new Tensor((float[])data.Clone(), (int[])shape.Clone());
    }

    public static Tensor FromScalar(float value)
    {
        return new Tensor([value], [1]);
    }

    public static Tensor Arange(int count)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = i;
        return new Tensor(data, [count]);
    }

    /// <summary>创建上三角矩阵（对角线上方为 1，其余为 0）</summary>
    public static Tensor Triu(int rows, int cols, int diagonal = 0)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (j >= i + diagonal)
                    data[i * cols + j] = 1.0f;
        return new Tensor(data, [rows, cols]);
    }

    // ─── 索引 ───

    public float this[params int[] indices]
    {
        get
        {
            int offset = ComputeOffset(indices);
            return Data[offset];
        }
        set
        {
            int offset = ComputeOffset(indices);
            Data[offset] = value;
        }
    }

    private int ComputeOffset(int[] indices)
    {
        if (indices.Length != Shape.Length)
            throw new ArgumentException($"索引维度 {indices.Length} 与张量维度 {Shape.Length} 不匹配");

        int offset = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indices[i];
            if (idx < 0) idx += Shape[i]; // 支持负索引
            if (idx < 0 || idx >= Shape[i])
                throw new IndexOutOfRangeException($"维度 {i} 索引 {indices[i]} 超出范围 [0, {Shape[i]})");
            offset += idx * Strides[i];
        }
        return offset;
    }

    // ─── 形状操作 ───

    public Tensor Reshape(params int[] newShape)
    {
        // 处理 -1 维度（自动推断）
        int unknownIdx = -1;
        int knownProduct = 1;
        for (int i = 0; i < newShape.Length; i++)
        {
            if (newShape[i] == -1)
            {
                if (unknownIdx != -1) throw new ArgumentException("Reshape 中只能有一个 -1");
                unknownIdx = i;
            }
            else
            {
                knownProduct *= newShape[i];
            }
        }

        var resolvedShape = (int[])newShape.Clone();
        if (unknownIdx != -1)
        {
            resolvedShape[unknownIdx] = Size / knownProduct;
        }

        var result = new Tensor(Data, resolvedShape, RequiresGrad);

        if (RequiresGrad)
        {
            result.Parents = [this];
            result.BackwardFn = () =>
            {
                EnsureGrad(this);
                for (int i = 0; i < Size; i++)
                    this.Grad![i] += result.Grad![i];
            };
        }

        return result;
    }

    public Tensor Transpose(int dim0, int dim1)
    {
        return TensorOps.Transpose(this, dim0, dim1);
    }

    public Tensor Unsqueeze(int dim)
    {
        if (dim < 0) dim += Ndim + 1;
        var newShape = new int[Ndim + 1];
        int si = 0;
        for (int i = 0; i < newShape.Length; i++)
        {
            if (i == dim)
                newShape[i] = 1;
            else
                newShape[i] = Shape[si++];
        }
        return Reshape(newShape);
    }

    public Tensor Squeeze(int dim)
    {
        if (dim < 0) dim += Ndim;
        if (Shape[dim] != 1)
            throw new ArgumentException($"维度 {dim} 的大小为 {Shape[dim]}，无法 squeeze");

        var newShape = new int[Ndim - 1];
        int si = 0;
        for (int i = 0; i < Ndim; i++)
        {
            if (i != dim)
                newShape[si++] = Shape[i];
        }
        return Reshape(newShape);
    }

    // ─── 自动微分 ───

    /// <summary>执行反向传播，从当前张量（通常是标量损失）回溯计算所有梯度</summary>
    public void Backward()
    {
        Autograd.Backward(this);
    }

    public void ZeroGrad()
    {
        if (Grad != null)
            Array.Clear(Grad);
    }

    // ─── 实用方法 ───

    /// <summary>返回数据的副本，不追踪梯度</summary>
    public Tensor Detach()
    {
        return new Tensor((float[])Data.Clone(), (int[])Shape.Clone(), false);
    }

    /// <summary>获取标量值（仅限 size=1 的张量）</summary>
    public float ToScalar()
    {
        if (Size != 1)
            throw new InvalidOperationException("ToScalar() 仅适用于包含单个元素的张量");
        return Data[0];
    }

    public override string ToString()
    {
        return $"Tensor(shape={ShapeToString(Shape)}, requiresGrad={RequiresGrad})";
    }

    // ─── 内部辅助 ───

    internal static void EnsureGrad(Tensor t)
    {
        t.Grad ??= new float[t.Size];
    }

    internal static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    internal static int ComputeSize(int[] shape)
    {
        int size = 1;
        for (int i = 0; i < shape.Length; i++)
            size *= shape[i];
        return size;
    }

    internal static string ShapeToString(int[] shape)
    {
        return $"[{string.Join(", ", shape)}]";
    }
}
