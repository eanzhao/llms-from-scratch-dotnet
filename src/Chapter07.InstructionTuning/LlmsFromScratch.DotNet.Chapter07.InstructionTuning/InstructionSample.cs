namespace LlmsFromScratch.DotNet.Chapter07.InstructionTuning;

/// <summary>
/// 指令样本数据结构（对应 Python 版 instruction-data.json 中的条目）
///
/// 每个样本包含:
/// - Instruction: 任务指令（"将以下句子翻译成英文"）
/// - Input: 可选的输入内容（"你好世界"）
/// - Output: 期望的输出（"Hello World"）
/// </summary>
public record InstructionSample(
    string Instruction,
    string Input,
    string Output);
