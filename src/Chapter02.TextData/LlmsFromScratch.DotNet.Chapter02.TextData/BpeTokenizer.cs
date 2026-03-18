using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LlmsFromScratch.DotNet.Chapter02.TextData;

/// <summary>
/// BPE (Byte Pair Encoding) 分词器
/// 对应 ch02/05_bpe-from-scratch 的 BPETokenizerSimple
///
/// 支持两种使用方式:
/// 1. Train() — 从语料训练 BPE 词表
/// 2. LoadFromOpenAI() — 加载 GPT-2 的 encoder.json + vocab.bpe
///
/// BPE 算法核心:
/// - 从 256 个字节 token 开始
/// - 反复找出最高频的相邻 token 对，合并为新 token
/// - 直到达到目标词表大小
/// </summary>
public class BpeTokenizer
{
    // token_id → 字符串
    private Dictionary<int, string> _vocab = new();
    // 字符串 → token_id
    private Dictionary<string, int> _inverseVocab = new();
    // (id1, id2) → merged_id（训练模式）
    private Dictionary<(int, int), int> _merges = new();
    // (str1, str2) → rank（GPT-2 模式，rank 越小优先级越高）
    private Dictionary<(string, string), int> _bpeRanks = new();

    private bool _useRanks; // true = GPT-2 模式，false = 训练模式

    public int VocabSize => _vocab.Count;

    // ═══════════════════════════════════════════
    // 方式 1: 从语料训练 BPE
    // ═══════════════════════════════════════════

    /// <summary>
    /// 从文本语料训练 BPE 词表
    /// </summary>
    public void Train(string text, int vocabSize, HashSet<string>? specialTokens = null)
    {
        _useRanks = false;
        specialTokens ??= new HashSet<string> { "<|endoftext|>" };

        // 初始化 256 个字节 token
        _vocab.Clear();
        _inverseVocab.Clear();
        _merges.Clear();

        for (int i = 0; i < 256; i++)
        {
            string ch = ((char)i).ToString();
            _vocab[i] = ch;
            _inverseVocab[ch] = i;
        }

        // 添加特殊 token
        int nextId = 256;
        foreach (var st in specialTokens)
        {
            _vocab[nextId] = st;
            _inverseVocab[st] = nextId;
            nextId++;
        }

        // 预分词: 按空格/标点切分
        var preTokens = PreTokenize(text);

        // 将每个 pre-token 转为字节 ID 序列
        var sequences = new List<List<int>>();
        foreach (var pt in preTokens)
        {
            var seq = new List<int>();
            foreach (char c in pt)
                seq.Add((int)c < 256 ? (int)c : (int)'?');
            if (seq.Count > 0) sequences.Add(seq);
        }

        // 迭代合并
        while (nextId < vocabSize)
        {
            // 统计所有相邻 pair 的频率
            var pairCounts = new Dictionary<(int, int), int>();
            foreach (var seq in sequences)
            {
                for (int i = 0; i < seq.Count - 1; i++)
                {
                    var pair = (seq[i], seq[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + 1;
                }
            }

            if (pairCounts.Count == 0) break;

            // 找频率最高的 pair
            var bestPair = pairCounts.MaxBy(kv => kv.Value).Key;

            // 创建新 token
            string merged = _vocab[bestPair.Item1] + _vocab[bestPair.Item2];
            _vocab[nextId] = merged;
            _inverseVocab[merged] = nextId;
            _merges[bestPair] = nextId;

            // 替换所有序列中的 pair
            for (int s = 0; s < sequences.Count; s++)
                sequences[s] = ReplacePair(sequences[s], bestPair, nextId);

            nextId++;
        }
    }

    // ═══════════════════════════════════════════
    // 方式 2: 加载 GPT-2 词表
    // ═══════════════════════════════════════════

    /// <summary>
    /// 加载 OpenAI GPT-2 的 encoder.json 和 vocab.bpe
    /// </summary>
    public void LoadFromOpenAI(string encoderJsonPath, string vocabBpePath)
    {
        _useRanks = true;
        _vocab.Clear();
        _inverseVocab.Clear();
        _bpeRanks.Clear();

        // 加载 encoder.json (token_string → id)
        var json = File.ReadAllText(encoderJsonPath);
        var encoder = JsonSerializer.Deserialize<Dictionary<string, int>>(json)!;

        foreach (var (token, id) in encoder)
        {
            _vocab[id] = token;
            _inverseVocab[token] = id;
        }

        // 处理特殊字符映射
        // GPT-2 中 Ċ (U+010A, ID 198) = 换行符
        if (_inverseVocab.ContainsKey("\u010a") && !_inverseVocab.ContainsKey("\n"))
            _inverseVocab["\n"] = _inverseVocab["\u010a"];

        // 加载 vocab.bpe (merge 规则)
        var lines = File.ReadAllLines(vocabBpePath);
        int rank = 0;
        foreach (var line in lines)
        {
            if (line.StartsWith("#version") || string.IsNullOrWhiteSpace(line))
                continue;

            var parts = line.Split(' ');
            if (parts.Length == 2)
            {
                _bpeRanks[(parts[0], parts[1])] = rank;
                rank++;
            }
        }
    }

    // ═══════════════════════════════════════════
    // 编码 / 解码
    // ═══════════════════════════════════════════

    /// <summary>将文本编码为 token ID 序列</summary>
    public int[] Encode(string text, HashSet<string>? allowedSpecial = null)
    {
        allowedSpecial ??= new HashSet<string> { "<|endoftext|>" };
        var result = new List<int>();

        // 处理特殊 token
        var specialPattern = string.Join("|", allowedSpecial.Select(Regex.Escape));
        var segments = allowedSpecial.Count > 0
            ? Regex.Split(text, $"({specialPattern})")
            : new[] { text };

        foreach (var segment in segments)
        {
            if (string.IsNullOrEmpty(segment)) continue;

            if (_inverseVocab.TryGetValue(segment, out int specialId))
            {
                result.Add(specialId);
                continue;
            }

            // 预分词
            var preTokens = PreTokenize(segment);

            foreach (var pt in preTokens)
            {
                if (_inverseVocab.TryGetValue(pt, out int directId))
                {
                    result.Add(directId);
                }
                else if (_useRanks)
                {
                    result.AddRange(BpeEncodeWithRanks(pt));
                }
                else
                {
                    result.AddRange(BpeEncodeWithMerges(pt));
                }
            }
        }

        return result.ToArray();
    }

    /// <summary>将 token ID 序列解码为文本</summary>
    public string Decode(int[] tokenIds)
    {
        var sb = new StringBuilder();
        foreach (var id in tokenIds)
        {
            if (_vocab.TryGetValue(id, out string? token))
            {
                // GPT-2 空格前缀: Ġ (U+0120) → 空格
                token = token.Replace("\u0120", " ");
                // GPT-2 换行: Ċ (U+010A) → 换行
                token = token.Replace("\u010a", "\n");
                sb.Append(token);
            }
        }
        return sb.ToString();
    }

    // ═══════════════════════════════════════════
    // 内部方法
    // ═══════════════════════════════════════════

    /// <summary>GPT-2 风格的预分词（按空白/标点切分，空格用 Ġ 前缀表示）</summary>
    private static List<string> PreTokenize(string text)
    {
        // GPT-2 正则: 匹配连续字母、连续数字、标点、空白+字母组合等
        var tokens = new List<string>();
        // 简化版正则，覆盖主要 pattern
        var pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";
        var matches = Regex.Matches(text, pattern);

        foreach (Match m in matches)
        {
            string tok = m.Value;
            // 空格前缀 → Ġ 前缀
            if (tok.StartsWith(' '))
                tok = "\u0120" + tok.Substring(1);
            tokens.Add(tok);
        }

        return tokens;
    }

    /// <summary>使用训练得到的 merges 进行 BPE 编码</summary>
    private List<int> BpeEncodeWithMerges(string token)
    {
        // 转为字符级 ID 序列
        var ids = new List<int>();
        foreach (char c in token)
        {
            if (_inverseVocab.TryGetValue(c.ToString(), out int id))
                ids.Add(id);
            else
                ids.Add((int)c < 256 ? (int)c : (int)'?');
        }

        // 反复应用 merges
        while (ids.Count >= 2)
        {
            // 找第一个可 merge 的 pair
            int bestIdx = -1;
            int bestNewId = -1;
            for (int i = 0; i < ids.Count - 1; i++)
            {
                var pair = (ids[i], ids[i + 1]);
                if (_merges.TryGetValue(pair, out int newId))
                {
                    bestIdx = i;
                    bestNewId = newId;
                    break;
                }
            }

            if (bestIdx < 0) break;

            ids[bestIdx] = bestNewId;
            ids.RemoveAt(bestIdx + 1);
        }

        return ids;
    }

    /// <summary>使用 GPT-2 的 rank 进行 BPE 编码</summary>
    private List<int> BpeEncodeWithRanks(string token)
    {
        // 转为单字符 symbol 列表
        var symbols = new List<string>();
        foreach (char c in token)
            symbols.Add(c.ToString());

        while (symbols.Count >= 2)
        {
            // 找 rank 最小（优先级最高）的 pair
            int bestRank = int.MaxValue;
            int bestIdx = -1;

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);
                if (_bpeRanks.TryGetValue(pair, out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0) break;

            // 合并
            symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
        }

        // 转为 ID
        var ids = new List<int>();
        foreach (var sym in symbols)
        {
            if (_inverseVocab.TryGetValue(sym, out int id))
                ids.Add(id);
            // 未知 token 跳过（GPT-2 通常不会出现）
        }

        return ids;
    }

    /// <summary>替换序列中所有匹配的 pair</summary>
    private static List<int> ReplacePair(List<int> sequence, (int, int) pair, int newId)
    {
        var result = new List<int>();
        int i = 0;
        while (i < sequence.Count)
        {
            if (i < sequence.Count - 1 && sequence[i] == pair.Item1 && sequence[i + 1] == pair.Item2)
            {
                result.Add(newId);
                i += 2;
            }
            else
            {
                result.Add(sequence[i]);
                i++;
            }
        }
        return result;
    }
}
