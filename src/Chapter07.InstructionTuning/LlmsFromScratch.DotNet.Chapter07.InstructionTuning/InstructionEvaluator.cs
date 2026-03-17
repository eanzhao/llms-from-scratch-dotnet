using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Chapter05.Pretraining;
using LlmsFromScratch.DotNet.Shared.Tokenization;

namespace LlmsFromScratch.DotNet.Chapter07.InstructionTuning;

/// <summary>
/// 指令评估器 - 为测试集中的每个样本生成响应
/// 对应 Python 版 gpt_instruction_finetuning.py 的评估部分
/// </summary>
public static class InstructionEvaluator
{
    /// <summary>
    /// 为指令样本列表生成响应
    /// </summary>
    public static List<(InstructionSample sample, string response)> Evaluate(
        GptModel model, SimpleTokenizer tokenizer,
        List<InstructionSample> samples, int maxNewTokens = 256,
        float temperature = 0.0f, int? topK = null)
    {
        var results = new List<(InstructionSample, string)>();
        int contextSize = model.PosEmbWeight.Shape[0];

        foreach (var sample in samples)
        {
            string prompt = PromptTemplate.FormatInput(sample) + "\n\n### Response:\n";
            var encoded = TextSampler.TextToTokenIds(prompt, tokenizer);

            // 获取 endoftext token ID
            int? eosId = tokenizer.TokenToId("<|endoftext|>");

            var output = AdvancedTextGenerator.Generate(
                model, encoded, maxNewTokens, contextSize,
                temperature: temperature, topK: topK, eosId: eosId);

            string fullText = TextSampler.TokenIdsToText(output, tokenizer);

            // 提取 Response 部分
            int responseStart = fullText.IndexOf("### Response:\n");
            string response = responseStart >= 0
                ? fullText[(responseStart + "### Response:\n".Length)..].Trim()
                : fullText;

            results.Add((sample, response));
        }

        return results;
    }

    /// <summary>打印评估结果</summary>
    public static void PrintResults(List<(InstructionSample sample, string response)> results, int maxShow = 5)
    {
        Console.WriteLine("\n═══ 指令微调评估结果 ═══\n");

        for (int i = 0; i < Math.Min(maxShow, results.Count); i++)
        {
            var (sample, response) = results[i];
            Console.WriteLine($"[{i + 1}] 指令: {sample.Instruction}");
            if (!string.IsNullOrWhiteSpace(sample.Input))
                Console.WriteLine($"    输入: {sample.Input}");
            Console.WriteLine($"    期望: {sample.Output}");
            Console.WriteLine($"    生成: {response}");
            Console.WriteLine();
        }
    }
}
