namespace LlmsFromScratch.DotNet.Chapter07.InstructionTuning;

/// <summary>
/// 提示词模板（对应 Python 版 format_input）
///
/// 将 InstructionSample 格式化为模型输入的文本格式:
///
/// Below is an instruction that describes a task. Write a response that
/// appropriately completes the request.
///
/// ### Instruction:
/// {instruction}
///
/// ### Input:
/// {input}        (仅当 input 非空时包含)
///
/// ### Response:
/// {output}
/// </summary>
public static class PromptTemplate
{
    /// <summary>
    /// 格式化指令（不含 response，用于推理时的输入）
    /// </summary>
    public static string FormatInput(InstructionSample sample)
    {
        string prompt = "Below is an instruction that describes a task. " +
            "Write a response that appropriately completes the request.\n\n" +
            $"### Instruction:\n{sample.Instruction}";

        if (!string.IsNullOrWhiteSpace(sample.Input))
            prompt += $"\n\n### Input:\n{sample.Input}";

        return prompt;
    }

    /// <summary>
    /// 格式化完整训练样本（含 response，用于 SFT 训练）
    /// </summary>
    public static string FormatFull(InstructionSample sample)
    {
        return FormatInput(sample) + $"\n\n### Response:\n{sample.Output}";
    }
}
