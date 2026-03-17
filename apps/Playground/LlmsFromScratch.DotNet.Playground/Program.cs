using LlmsFromScratch.DotNet.Chapter02.TextData;
using LlmsFromScratch.DotNet.Chapter03.Attention;
using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Playground;

internal static class Program
{
    private static void Main()
    {
        const string title = "LLMs from Scratch — C# Playground";
        Console.WriteLine(title);
        Console.WriteLine(new string('=', title.Length));
        Console.WriteLine();

        // ═══ Chapter 02 演示: 文本数据处理 ═══
        Console.WriteLine("═══ Chapter 02: 文本数据处理 ═══");
        DemoChapter02();

        // ═══ Chapter 03 演示: 注意力机制 ═══
        Console.WriteLine("\n═══ Chapter 03: 注意力机制 ═══");
        DemoChapter03();

        // ═══ Chapter 04 演示: GPT 模型 ═══
        Console.WriteLine("\n═══ Chapter 04: GPT 模型 ═══");
        DemoChapter04();
    }

    static void DemoChapter02()
    {
        string text = "Hello, I am a language model. I can generate text based on the input I receive.";
        var tokenizer = new LlmsFromScratch.DotNet.Shared.Tokenization.SimpleTokenizer(text);
        Console.WriteLine($"  词汇表大小: {tokenizer.VocabSize}");

        var ids = tokenizer.Encode(text);
        Console.WriteLine($"  编码后长度: {ids.Length}");
        Console.WriteLine($"  前5个ID: [{string.Join(", ", ids.Take(5))}]");

        var decoded = tokenizer.Decode(ids);
        Console.WriteLine($"  解码: {decoded}");

        // 数据集
        int maxLength = 4;
        var dataset = new GptDatasetV1(ids, maxLength, stride: 2);
        Console.WriteLine($"  数据集样本数 (maxLen={maxLength}, stride=2): {dataset.Count}");

        if (dataset.Count > 0)
        {
            var (inp, tgt) = dataset.GetItem(0);
            Console.WriteLine($"  样本0 input: [{string.Join(", ", inp.Data.Take(maxLength).Select(x => (int)x))}]");
            Console.WriteLine($"  样本0 target: [{string.Join(", ", tgt.Data.Take(maxLength).Select(x => (int)x))}]");
        }
    }

    static void DemoChapter03()
    {
        // 小型自注意力测试
        int batch = 2, seq = 4, dIn = 8, dOut = 8, numHeads = 2;
        var input = Tensor.Randn([batch, seq, dIn]);
        Console.WriteLine($"  输入形状: {input}");

        var mha = new MultiHeadAttention(dIn, dOut, contextLength: 16, numHeads: numHeads);
        var output = mha.Forward(input);
        Console.WriteLine($"  MultiHeadAttention 输出形状: {output}");
        Console.WriteLine($"  输出前3个值: [{output.Data[0]:F4}, {output.Data[1]:F4}, {output.Data[2]:F4}]");
    }

    static void DemoChapter04()
    {
        // 使用微型配置快速验证
        var cfg = GptConfig.Tiny;
        Console.WriteLine($"  配置: vocab={cfg.VocabSize}, ctx={cfg.ContextLength}, " +
            $"emb={cfg.EmbDim}, heads={cfg.NHeads}, layers={cfg.NLayers}");

        var model = new GptModel(cfg);
        var paramCount = model.Parameters().Sum(p => p.Size);
        Console.WriteLine($"  参数总数: {paramCount:N0}");

        // 前向传播
        var tokenIds = Tensor.FromArray([1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f], [2, 4]);
        var logits = model.Forward(tokenIds);
        Console.WriteLine($"  输入形状: {tokenIds} -> 输出形状: {logits}");
        Console.WriteLine($"  logits 期望: [2, 4, {cfg.VocabSize}]");

        // 文本生成
        var startTokens = Tensor.FromArray([1f, 2f], [1, 2]);
        var generated = TextGenerator.GenerateSimple(model, startTokens, maxNewTokens: 5, contextSize: cfg.ContextLength);
        Console.Write($"  生成的 token IDs: [");
        Console.Write(string.Join(", ", generated.Data.Select(x => (int)x)));
        Console.WriteLine("]");
    }
}
