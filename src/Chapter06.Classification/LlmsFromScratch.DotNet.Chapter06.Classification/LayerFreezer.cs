using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Nn;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter06.Classification;

/// <summary>
/// 参数冻结/解冻器（对应 Python 版的 requires_grad = False 操作）
///
/// 微调策略:
/// 1. 先冻结所有参数（不参与训练）
/// 2. 解冻最后一个 Transformer 块和 final_norm
/// 3. 替换 out_head 为分类头（vocabSize → numClasses）
///
/// 这样只训练少量参数，既节省计算又防止过拟合。
/// </summary>
public static class LayerFreezer
{
    /// <summary>冻结模型所有参数</summary>
    public static void FreezeAll(Module model)
    {
        foreach (var param in model.Parameters())
            param.RequiresGrad = false;
    }

    /// <summary>解冻指定名称前缀的参数</summary>
    public static void UnfreezeByPrefix(Module model, string prefix)
    {
        foreach (var (name, param) in model.NamedParameters())
        {
            if (name.StartsWith(prefix))
                param.RequiresGrad = true;
        }
    }

    /// <summary>
    /// 为分类任务准备模型:
    /// 1. 冻结所有参数
    /// 2. 解冻最后一个 Transformer 块
    /// 3. 解冻 final_norm
    /// 4. 替换输出头
    /// </summary>
    public static void PrepareForClassification(GptModel model, int numClasses, Random? rng = null)
    {
        // 冻结全部
        FreezeAll(model);

        // 解冻最后一个 Transformer 块
        int lastBlockIdx = model.Config.NLayers - 1;
        UnfreezeByPrefix(model, $"trf_blocks.layer_{lastBlockIdx}.");

        // 解冻 final_norm
        UnfreezeByPrefix(model, "final_norm.");

        // 替换输出头为分类头
        var classHead = new Linear(model.Config.EmbDim, numClasses, bias: true, rng: rng);
        model.ReplaceOutHead(classHead);
    }
}
