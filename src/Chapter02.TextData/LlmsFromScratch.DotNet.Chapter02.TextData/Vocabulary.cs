namespace LlmsFromScratch.DotNet.Chapter02.TextData;

/// <summary>
/// 词汇表 - 管理 token 与整数 ID 之间的双向映射
/// 对应书中 Chapter 2 的词汇表构建过程
/// </summary>
public class Vocabulary
{
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<int, string> _idToToken;

    public int Size => _tokenToId.Count;

    /// <summary>从 token 列表构建词汇表（按字典序排列后编号）</summary>
    public Vocabulary(IEnumerable<string> tokens, IEnumerable<string>? specialTokens = null)
    {
        var uniqueTokens = new SortedSet<string>(tokens);
        if (specialTokens != null)
            foreach (var st in specialTokens)
                uniqueTokens.Add(st);

        _tokenToId = new Dictionary<string, int>();
        _idToToken = new Dictionary<int, string>();

        int idx = 0;
        foreach (var token in uniqueTokens)
        {
            _tokenToId[token] = idx;
            _idToToken[idx] = token;
            idx++;
        }
    }

    /// <summary>从已有映射构建</summary>
    public Vocabulary(Dictionary<string, int> tokenToId)
    {
        _tokenToId = new Dictionary<string, int>(tokenToId);
        _idToToken = new Dictionary<int, string>();
        foreach (var (token, id) in tokenToId)
            _idToToken[id] = token;
    }

    public int Encode(string token) =>
        _tokenToId.TryGetValue(token, out int id) ? id : throw new KeyNotFoundException($"未知 token: '{token}'");

    public string Decode(int id) =>
        _idToToken.TryGetValue(id, out string? token) ? token : throw new KeyNotFoundException($"未知 ID: {id}");

    public bool Contains(string token) => _tokenToId.ContainsKey(token);

    public int? TryEncode(string token) =>
        _tokenToId.TryGetValue(token, out int id) ? id : null;
}
