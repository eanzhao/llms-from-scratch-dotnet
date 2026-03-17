---
title: 交互式 Playground
order: 99
summary: 在浏览器中体验 LLM 核心组件！交互式演示分词、注意力计算、文本生成等关键步骤。
status: done
tags:
  - playground
  - interactive
  - demo
---

## 在线演示

以下是可交互的 LLM 核心组件演示，无需运行 C# 代码即可直观理解各步骤。

---

### 1. 分词器 (Tokenizer)

输入文本，观察分词结果和 Token ID 映射：

<div id="tokenizer-demo" class="playground-widget">
  <textarea id="tok-input" rows="2" placeholder="输入文本，例如：Hello, I am a language model." style="width:100%;font-family:monospace;padding:8px;border-radius:8px;border:1px solid #ddd;">Hello, I am a language model.</textarea>
  <button id="tok-btn" style="margin:8px 0;padding:6px 16px;border-radius:8px;border:1px solid #316dca;background:#316dca;color:#fff;cursor:pointer;">分词</button>
  <div id="tok-output" style="font-family:monospace;font-size:0.85rem;"></div>
</div>

---

### 2. 滑动窗口数据集

调整窗口大小和步长，观察训练样本的生成方式：

<div id="dataset-demo" class="playground-widget">
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;">
    <label>窗口大小 <input id="ds-maxlen" type="number" value="4" min="2" max="8" style="width:60px;padding:4px;border-radius:6px;border:1px solid #ddd;"></label>
    <label>步长 <input id="ds-stride" type="number" value="2" min="1" max="4" style="width:60px;padding:4px;border-radius:6px;border:1px solid #ddd;"></label>
  </div>
  <button id="ds-btn" style="margin:4px 0 8px;padding:6px 16px;border-radius:8px;border:1px solid #316dca;background:#316dca;color:#fff;cursor:pointer;">生成样本</button>
  <div id="ds-output" style="font-family:monospace;font-size:0.85rem;"></div>
</div>

---

### 3. 注意力分数可视化

输入序列长度，查看缩放点积注意力的计算过程和因果掩码效果：

<div id="attention-demo" class="playground-widget">
  <label>序列长度 <input id="attn-seqlen" type="number" value="4" min="2" max="8" style="width:60px;padding:4px;border-radius:6px;border:1px solid #ddd;"></label>
  <button id="attn-btn" style="margin:4px 0 8px;padding:6px 16px;border-radius:8px;border:1px solid #316dca;background:#316dca;color:#fff;cursor:pointer;">计算注意力</button>
  <div id="attn-output" style="font-family:monospace;font-size:0.85rem;overflow-x:auto;"></div>
</div>

---

### 4. Softmax 温度实验

调整 temperature 参数，观察概率分布的变化：

<div id="temperature-demo" class="playground-widget">
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:8px;">
    <label>Temperature <input id="temp-val" type="range" min="0.1" max="3.0" step="0.1" value="1.0" style="width:160px;"> <span id="temp-display">1.0</span></label>
    <label>Top-k <input id="topk-val" type="number" value="0" min="0" max="10" style="width:60px;padding:4px;border-radius:6px;border:1px solid #ddd;"> <small>(0=关闭)</small></label>
  </div>
  <div id="temp-output" style="font-family:monospace;font-size:0.85rem;"></div>
</div>

---

### 5. GPT 架构总览

<div class="playground-widget" style="font-family:monospace;font-size:0.82rem;line-height:1.6;white-space:pre;overflow-x:auto;">
输入 Token IDs: [batch, seq_len]
       │
  Token Embedding + Positional Embedding + Dropout
       │ → [batch, seq_len, emb_dim]
       │
  ┌────┴────┐
  │ Block×N │  Pre-Norm TransformerBlock:
  │         │    LayerNorm → MultiHeadAttention → +residual
  │         │    LayerNorm → FeedForward(GELU)  → +residual
  └────┬────┘
       │ → [batch, seq_len, emb_dim]
       │
  Final LayerNorm → Linear(vocab_size)
       │
  Logits: [batch, seq_len, vocab_size]
       │
  argmax / temperature+top-k → 下一个 token
</div>

---

### 6. 贪心文本生成模拟

模拟 GPT 文本生成过程（使用随机 logits 演示流程）：

<div id="generate-demo" class="playground-widget">
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;">
    <label>起始 Token <input id="gen-start" type="text" value="The" style="width:100px;padding:4px;border-radius:6px;border:1px solid #ddd;"></label>
    <label>生成步数 <input id="gen-steps" type="number" value="8" min="1" max="20" style="width:60px;padding:4px;border-radius:6px;border:1px solid #ddd;"></label>
    <label>Temperature <input id="gen-temp" type="number" value="0.8" min="0.1" max="2.0" step="0.1" style="width:80px;padding:4px;border-radius:6px;border:1px solid #ddd;"></label>
  </div>
  <button id="gen-btn" style="margin:4px 0 8px;padding:6px 16px;border-radius:8px;border:1px solid #316dca;background:#316dca;color:#fff;cursor:pointer;">生成</button>
  <div id="gen-output" style="font-family:monospace;font-size:0.85rem;"></div>
</div>

<style>
.playground-widget {
  margin: 12px 0 20px;
  padding: 16px;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #f9fafb;
}
</style>

<script type="module">
// ═══ 1. Tokenizer Demo ═══
function simpleTokenize(text) {
  return text.match(/[\w]+|[^\s\w]/g) || [];
}

document.getElementById('tok-btn')?.addEventListener('click', () => {
  const text = document.getElementById('tok-input').value;
  const tokens = simpleTokenize(text);
  const vocab = {};
  let nextId = 0;
  tokens.forEach(t => { if (!(t in vocab)) vocab[t] = nextId++; });
  const ids = tokens.map(t => vocab[t]);

  let html = `<div style="margin:4px 0"><b>Tokens (${tokens.length}):</b></div>`;
  html += `<div style="display:flex;flex-wrap:wrap;gap:4px;margin:4px 0">`;
  tokens.forEach((t, i) => {
    html += `<span style="padding:2px 8px;border-radius:6px;background:#dbeafe;border:1px solid #93c5fd;font-size:0.82rem">${t}<sub style="color:#6b7280;font-size:0.7rem">${ids[i]}</sub></span>`;
  });
  html += `</div>`;
  html += `<div style="margin:8px 0"><b>Token IDs:</b> [${ids.join(', ')}]</div>`;
  html += `<div><b>词汇表大小:</b> ${Object.keys(vocab).length}</div>`;
  document.getElementById('tok-output').innerHTML = html;
});

// ═══ 2. Dataset Demo ═══
document.getElementById('ds-btn')?.addEventListener('click', () => {
  const text = document.getElementById('tok-input').value;
  const tokens = simpleTokenize(text);
  const vocab = {};
  let nextId = 0;
  tokens.forEach(t => { if (!(t in vocab)) vocab[t] = nextId++; });
  const ids = tokens.map(t => vocab[t]);

  const maxLen = parseInt(document.getElementById('ds-maxlen').value);
  const stride = parseInt(document.getElementById('ds-stride').value);

  let html = `<div style="margin-bottom:8px"><b>Token序列:</b> [${ids.join(', ')}] (长度 ${ids.length})</div>`;
  html += `<table style="border-collapse:collapse;font-size:0.82rem;width:100%">`;
  html += `<tr style="background:#e5e7eb"><th style="padding:4px 8px;text-align:left">样本</th><th style="padding:4px 8px;text-align:left">Input</th><th style="padding:4px 8px;text-align:left">Target</th></tr>`;

  let count = 0;
  for (let i = 0; i + maxLen < ids.length; i += stride) {
    const inp = ids.slice(i, i + maxLen);
    const tgt = ids.slice(i + 1, i + 1 + maxLen);
    if (tgt.length < maxLen) break;
    html += `<tr style="border-top:1px solid #e5e7eb"><td style="padding:4px 8px">#${count}</td><td style="padding:4px 8px">[${inp.join(', ')}]</td><td style="padding:4px 8px">[${tgt.join(', ')}]</td></tr>`;
    count++;
  }
  html += `</table>`;
  html += `<div style="margin-top:8px"><b>样本数:</b> ${count}</div>`;
  document.getElementById('ds-output').innerHTML = html;
});

// ═══ 3. Attention Demo ═══
function randomMatrix(rows, cols) {
  return Array.from({length: rows}, () => Array.from({length: cols}, () => (Math.random() * 2 - 1)));
}

function matmul(a, bT) {
  const rows = a.length, cols = bT.length;
  return a.map(row => bT.map(col => row.reduce((s, v, i) => s + v * col[i], 0)));
}

function softmaxRow(row) {
  const max = Math.max(...row);
  const exps = row.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function fmtMatrix(mat, label) {
  let html = `<div style="margin:6px 0"><b>${label}:</b></div><table style="border-collapse:collapse;font-size:0.78rem">`;
  mat.forEach(row => {
    html += '<tr>' + row.map(v => `<td style="padding:2px 6px;text-align:right;border:1px solid #e5e7eb;${v === 0 ? 'color:#d1d5db' : ''}">${v.toFixed(3)}</td>`).join('') + '</tr>';
  });
  return html + '</table>';
}

document.getElementById('attn-btn')?.addEventListener('click', () => {
  const n = parseInt(document.getElementById('attn-seqlen').value);
  const dk = 4;
  const Q = randomMatrix(n, dk);
  const K = randomMatrix(n, dk);
  const scale = Math.sqrt(dk);
  const scores = matmul(Q, K).map(row => row.map(v => v / scale));

  // causal mask
  const masked = scores.map((row, i) => row.map((v, j) => j > i ? -1e9 : v));
  const weights = masked.map(softmaxRow);

  let html = fmtMatrix(scores, `QK<sup>T</sup>/√d<sub>k</sub> (d<sub>k</sub>=${dk})`);
  html += fmtMatrix(masked.map((row, i) => row.map((v, j) => j > i ? -Infinity : v)).map(row => row.map(v => isFinite(v) ? v : -99)), '因果掩码后 (-99 = -∞)');
  html += fmtMatrix(weights, 'Softmax 注意力权重');
  html += `<div style="margin-top:8px;color:#6b7280;font-size:0.78rem">注意：上三角位置 softmax 后权重为 ≈0（因果掩码效果）</div>`;
  document.getElementById('attn-output').innerHTML = html;
});

// ═══ 4. Temperature Demo ═══
const rawLogits = [2.1, 1.5, 0.8, 0.3, -0.2, -0.5, -1.0, -1.5, -2.0, -3.0];
const vocabWords = ['the', 'a', 'is', 'in', 'to', 'and', 'of', 'it', 'for', 'on'];

function renderTempDemo() {
  const temp = parseFloat(document.getElementById('temp-val').value);
  const topk = parseInt(document.getElementById('topk-val').value);
  document.getElementById('temp-display').textContent = temp.toFixed(1);

  let logits = rawLogits.map(v => v / temp);

  if (topk > 0 && topk < logits.length) {
    const sorted = [...logits].sort((a, b) => b - a);
    const threshold = sorted[topk - 1];
    logits = logits.map(v => v >= threshold ? v : -1e9);
  }

  const probs = softmaxRow(logits);
  const maxProb = Math.max(...probs);

  let html = `<table style="border-collapse:collapse;font-size:0.82rem;width:100%">`;
  html += `<tr style="background:#e5e7eb"><th style="padding:4px 8px">Token</th><th style="padding:4px 8px">原始Logit</th><th style="padding:4px 8px">缩放后</th><th style="padding:4px 8px">概率</th><th style="padding:4px 8px">分布</th></tr>`;
  probs.forEach((p, i) => {
    const barW = Math.round(p / maxProb * 120);
    const isFiltered = topk > 0 && logits[i] < -1e8;
    html += `<tr style="border-top:1px solid #e5e7eb;${isFiltered ? 'color:#d1d5db' : ''}">`;
    html += `<td style="padding:3px 8px;font-weight:600">${vocabWords[i]}</td>`;
    html += `<td style="padding:3px 8px;text-align:right">${rawLogits[i].toFixed(1)}</td>`;
    html += `<td style="padding:3px 8px;text-align:right">${isFiltered ? '-∞' : (rawLogits[i]/temp).toFixed(2)}</td>`;
    html += `<td style="padding:3px 8px;text-align:right">${(p*100).toFixed(1)}%</td>`;
    html += `<td style="padding:3px 8px"><div style="height:14px;width:${barW}px;background:${isFiltered ? '#e5e7eb' : '#316dca'};border-radius:3px"></div></td>`;
    html += `</tr>`;
  });
  html += `</table>`;
  document.getElementById('temp-output').innerHTML = html;
}

document.getElementById('temp-val')?.addEventListener('input', renderTempDemo);
document.getElementById('topk-val')?.addEventListener('change', renderTempDemo);
renderTempDemo();

// ═══ 6. Generate Demo ═══
const demoVocab = ['the','a','is','was','in','on','to','and','of','it','that','for','with','as','at','by','from','or','an','be','this','have','had','not','but','what','all','were','we','when','your','can','said','there','each','which','do','how','if','will','up','about','out','many','then','them','her','like','so','these','its','would','make','just','over','such','know','more','some','time','very','could','no','him','one','come','than','first','been','who','now','people','my','made','after','did','back','see','only','way','where','get','much','go','well','new','also','into','because','good','day','here','most','find','still','between','name','should','being','under','long','right','down','too','own','say','small','every','another','around','want','help','thing','again','does','ask','try','late','big','never','end','while','part','last','run','off','put','old','take','place','keep','let','begin','since','work','may','eat','little','use','home','even','hand','high','year','give','look','thought','might','next','tell','open','along','seem','need'];

function sampleFromLogits(logits, temperature) {
  const scaled = logits.map(v => v / temperature);
  const probs = softmaxRow(scaled);
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

document.getElementById('gen-btn')?.addEventListener('click', () => {
  const startWord = document.getElementById('gen-start').value.toLowerCase();
  const steps = parseInt(document.getElementById('gen-steps').value);
  const temp = parseFloat(document.getElementById('gen-temp').value);

  let sequence = [startWord];
  let html = `<div style="margin-bottom:8px"><b>生成过程:</b></div>`;

  for (let step = 0; step < steps; step++) {
    const logits = demoVocab.map(() => Math.random() * 4 - 2);
    // bias toward common words
    logits[0] += 1; logits[4] += 0.5; logits[7] += 0.5;
    const idx = sampleFromLogits(logits, temp);
    const word = demoVocab[idx];
    sequence.push(word);
    html += `<div style="color:#6b7280;font-size:0.78rem">步骤 ${step+1}: logits → softmax(T=${temp}) → 采样 → "<b style="color:#316dca">${word}</b>"</div>`;
  }

  html += `<div style="margin-top:12px;padding:8px 12px;background:#dbeafe;border-radius:8px;font-size:0.9rem"><b>生成结果:</b> ${sequence.join(' ')}</div>`;
  html += `<div style="margin-top:8px;color:#9ca3af;font-size:0.75rem">注：此演示使用随机 logits 模拟流程，非真实模型输出。运行 C# Playground 可使用实际 GPT 模型。</div>`;
  document.getElementById('gen-output').innerHTML = html;
});
</script>
