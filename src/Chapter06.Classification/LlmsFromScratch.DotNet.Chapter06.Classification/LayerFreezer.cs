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
/// LoRA 策略（Appendix E）:
/// 1. 冻结所有参数
/// 2. 在目标 Linear 层上添加 LoRA 旁路
/// 3. 只有 LoRA 的 A/B 矩阵参与训练
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

    /// <summary>
    /// 在模型的注意力层 Linear 上应用 LoRA（Appendix E）
    ///
    /// 遍历模型所有 TransformerBlock，将注意力层的 Q/V 投影
    /// 替换为 LinearWithLora。这是最常见的 LoRA 应用方式:
    /// - 只在 W_query 和 W_value 上加 LoRA（原论文推荐）
    /// - W_key 和 out_proj 保持冻结
    ///
    /// 用法:
    ///   LayerFreezer.FreezeAll(model);
    ///   LayerFreezer.ApplyLora(model, rank: 8, alpha: 16);
    /// </summary>
    public static void ApplyLora(GptModel model, int rank, float alpha = 16.0f, Random? rng = null)
    {
        // 先冻结所有参数
        FreezeAll(model);

        // 通过 NamedModules 找到所有注意力层中的 W_query 和 W_value
        var replacements = new List<(Module parent, string childName, Linear linear)>();

        foreach (var (name, module) in model.NamedModules())
        {
            // 找 att.W_query 和 att.W_value
            if ((name.EndsWith(".W_query") || name.EndsWith(".W_value")) && module is Linear linear)
            {
                // 找到父模块（注意力层）
                string parentName = name.Substring(0, name.LastIndexOf('.'));
                foreach (var (pName, pModule) in model.NamedModules())
                {
                    if (pName == parentName)
                    {
                        string childName = name.Substring(name.LastIndexOf('.') + 1);
                        replacements.Add((pModule, childName, linear));
                        break;
                    }
                }
            }
        }

        // 执行替换
        foreach (var (parent, childName, linear) in replacements)
        {
            var loraWrapper = new LinearWithLora(linear, rank, alpha, rng);
            parent.ReplaceSubModule(childName, loraWrapper);
        }
    }

    /// <summary>
    /// 统计模型可训练参数数量（用于验证 LoRA 效果）
    /// </summary>
    public static (int Total, int Trainable) CountParameters(Module model)
    {
        int total = 0;
        int trainable = 0;
        foreach (var param in model.Parameters())
        {
            total += param.Size;
            if (param.RequiresGrad)
                trainable += param.Size;
        }
        return (total, trainable);
    }
}
