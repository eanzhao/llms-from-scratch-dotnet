namespace LlmsFromScratch.DotNet.Shared.Tokenization;

/// <summary>
/// 简单分词器 - 基于正则表达式的词级分词
/// 教学用途，模拟书中 Chapter 2 的基础分词器
///
/// 分词规则:
/// 1. 按空白和标点符号分割
/// 2. 保留标点符号作为独立 token
/// 3. 支持特殊 token（如 &lt;|endoftext|&gt;）
/// </summary>
public class SimpleTokenizer
{
    private readonly Dictionary<string, int> _strToInt;
    private readonly Dictionary<int, string> _intToStr;

    public int VocabSize => _strToInt.Count;

    /// <summary>
    /// 从文本语料构建词汇表
    /// </summary>
    public SimpleTokenizer(string text, IEnumerable<string>? specialTokens = null)
    {
        // 分词并收集唯一 token
        var tokens = Tokenize(text);
        var uniqueTokens = new SortedSet<string>(tokens);

        // 添加特殊 token
        if (specialTokens != null)
        {
            foreach (var st in specialTokens)
                uniqueTokens.Add(st);
        }

        _strToInt = new Dictionary<string, int>();
        _intToStr = new Dictionary<int, string>();

        int idx = 0;
        foreach (var token in uniqueTokens)
        {
            _strToInt[token] = idx;
            _intToStr[idx] = token;
            idx++;
        }
    }

    /// <summary>从已有词汇表构建</summary>
    public SimpleTokenizer(Dictionary<string, int> vocab)
    {
        _strToInt = new Dictionary<string, int>(vocab);
        _intToStr = new Dictionary<int, string>();
        foreach (var (str, id) in vocab)
            _intToStr[id] = str;
    }

    /// <summary>将文本编码为 token ID 序列</summary>
    public int[] Encode(string text)
    {
        var tokens = Tokenize(text);
        var ids = new List<int>();

        foreach (var token in tokens)
        {
            if (_strToInt.TryGetValue(token, out int id))
                ids.Add(id);
            // 未知 token 被跳过（简单实现）
        }

        return ids.ToArray();
    }

    /// <summary>将 token ID 序列解码为文本</summary>
    public string Decode(int[] ids)
    {
        var tokens = new List<string>();
        foreach (var id in ids)
        {
            if (_intToStr.TryGetValue(id, out string? token))
                tokens.Add(token);
        }

        // 简单的空格连接，标点前不加空格
        var result = new System.Text.StringBuilder();
        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            if (i > 0 && !IsPunctuation(token))
                result.Append(' ');
            result.Append(token);
        }
        return result.ToString();
    }

    /// <summary>获取 token 对应的 ID</summary>
    public int? TokenToId(string token)
    {
        return _strToInt.TryGetValue(token, out int id) ? id : null;
    }

    /// <summary>获取 ID 对应的 token</summary>
    public string? IdToToken(int id)
    {
        return _intToStr.TryGetValue(id, out string? token) ? token : null;
    }

    // ─── 内部方法 ───

    /// <summary>将文本分割为 token 列表</summary>
    private static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        var current = new System.Text.StringBuilder();

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];

            if (char.IsWhiteSpace(c))
            {
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
            }
            else if (IsPunctuationChar(c))
            {
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
                tokens.Add(c.ToString());
            }
            else
            {
                current.Append(c);
            }
        }

        if (current.Length > 0)
            tokens.Add(current.ToString());

        return tokens;
    }

    private static bool IsPunctuationChar(char c)
    {
        return c is '.' or ',' or '!' or '?' or ';' or ':' or '\'' or '"'
            or '(' or ')' or '[' or ']' or '{' or '}' or '-' or '—';
    }

    private static bool IsPunctuation(string s)
    {
        return s.Length == 1 && IsPunctuationChar(s[0]);
    }
}
