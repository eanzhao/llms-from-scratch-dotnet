using LlmsFromScratch.DotNet.Chapter04.Gpt;
using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Chapter06.Classification;

/// <summary>
/// 分类指标计算（对应 Python 版 calc_accuracy_loader）
/// </summary>
public static class ClassificationMetrics
{
    /// <summary>
    /// 计算分类准确率
    /// 取最后一个 token 的 logits，argmax 得到预测类别，与真实标签对比
    /// </summary>
    public static float CalcAccuracy(GptModel model, List<(Tensor input, Tensor label)> data, int batchSize)
    {
        model.SetTraining(false);
        int correct = 0;
        int total = 0;

        for (int i = 0; i < data.Count; i += batchSize)
        {
            int actual = Math.Min(batchSize, data.Count - i);
            int seqLen = data[0].input.Shape[0];

            var inputData = new float[actual * seqLen];
            for (int j = 0; j < actual; j++)
                Array.Copy(data[i + j].input.Data, 0, inputData, j * seqLen, seqLen);

            var inputBatch = new Tensor(inputData, [actual, seqLen]);
            var logits = model.Forward(inputBatch); // [batch, seq, numClasses]

            int numClasses = logits.Shape[2];
            int seq = logits.Shape[1];

            // 取最后 token 的 logits 并 argmax
            for (int j = 0; j < actual; j++)
            {
                float maxVal = float.NegativeInfinity;
                int predicted = 0;
                for (int c = 0; c < numClasses; c++)
                {
                    float val = logits.Data[(j * seq + seq - 1) * numClasses + c];
                    if (val > maxVal) { maxVal = val; predicted = c; }
                }

                int trueLabel = (int)data[i + j].label.Data[0];
                if (predicted == trueLabel) correct++;
                total++;
            }
        }

        model.SetTraining(true);
        return total > 0 ? (float)correct / total : 0;
    }
}
