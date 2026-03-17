namespace LlmsFromScratch.DotNet.Shared.Tensors;

/// <summary>
/// 自动微分引擎
/// 通过拓扑排序遍历计算图，反向调用每个节点的 BackwardFn
///
/// 工作原理:
/// 1. 从输出张量（通常是标量损失）出发
/// 2. DFS 遍历所有父节点，构建拓扑排序
/// 3. 反向遍历，依次调用 BackwardFn 传播梯度
/// </summary>
internal static class Autograd
{
    /// <summary>
    /// 执行反向传播
    /// </summary>
    /// <param name="root">起始张量（通常是损失函数的输出）</param>
    public static void Backward(Tensor root)
    {
        // 初始化根节点梯度为 1.0
        root.Grad = new float[root.Size];
        Array.Fill(root.Grad, 1.0f);

        // 拓扑排序
        var sorted = TopologicalSort(root);

        // 反向遍历，调用每个节点的 BackwardFn
        for (int i = sorted.Count - 1; i >= 0; i--)
        {
            sorted[i].BackwardFn?.Invoke();
        }
    }

    /// <summary>
    /// 对计算图做拓扑排序（后序 DFS）
    /// 保证每个节点在其所有子节点之后被处理
    /// </summary>
    private static List<Tensor> TopologicalSort(Tensor root)
    {
        var visited = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        var sorted = new List<Tensor>();

        void Dfs(Tensor node)
        {
            if (!visited.Add(node))
                return;

            if (node.Parents != null)
            {
                foreach (var parent in node.Parents)
                    Dfs(parent);
            }

            sorted.Add(node);
        }

        Dfs(root);
        return sorted;
    }
}
