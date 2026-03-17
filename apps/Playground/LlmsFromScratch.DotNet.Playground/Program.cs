using LlmsFromScratch.DotNet.Chapter02.TextData;
using LlmsFromScratch.DotNet.Chapter03.Attention;
using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Chapter05.Pretraining;
using LlmsFromScratch.DotNet.Chapter06.Classification;
using LlmsFromScratch.DotNet.Chapter07.InstructionTuning;
using LlmsFromScratch.DotNet.Shared;

namespace LlmsFromScratch.DotNet.Playground;

internal static class Program
{
    private static void Main()
    {
        var modules = new (string Name, string Goal)[]
        {
            (Chapter02TextDataModule.Name, Chapter02TextDataModule.Goal),
            (Chapter03AttentionModule.Name, Chapter03AttentionModule.Goal),
            (Chapter04GptModule.Name, Chapter04GptModule.Goal),
            (Chapter05PretrainingModule.Name, Chapter05PretrainingModule.Goal),
            (Chapter06ClassificationModule.Name, Chapter06ClassificationModule.Goal),
            (Chapter07InstructionTuningModule.Name, Chapter07InstructionTuningModule.Goal)
        };

        Console.WriteLine(ProjectMetadata.Title);
        Console.WriteLine(new string('=', ProjectMetadata.Title.Length));
        Console.WriteLine();

        foreach (var module in modules)
        {
            Console.WriteLine($"- {module.Name}");
            Console.WriteLine($"  {module.Goal}");
        }

        Console.WriteLine();
        Console.WriteLine("Docs site lives in ./web");
        Console.WriteLine("Roadmap starts at ./web/src/content/docs/roadmap.md");
    }
}
