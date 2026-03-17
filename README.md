# LLMs from Scratch in C# / .NET

这个仓库把 `LLMs-from-scratch` 的学习路线迁移到 `C# / .NET` 语境里，并额外配了一个 Astro 文档站，用 Markdown 来组织学习内容。

## 目标

- 用你熟悉的 `C# / .NET` 重走一次从零构建 LLM 的主线。
- 保留原书的章节顺序，避免一上来就陷入框架细节。
- 让代码、实验、学习笔记和前端展示在同一个仓库里协同演进。

## 仓库结构

```text
.
├── LlmsFromScratch.DotNet.sln
├── apps/
│   └── Playground/
├── src/
│   ├── Shared/
│   ├── Chapter02.TextData/
│   ├── Chapter03.Attention/
│   ├── Chapter04.Gpt/
│   ├── Chapter05.Pretraining/
│   ├── Chapter06.Classification/
│   └── Chapter07.InstructionTuning/
└── web/
    └── src/content/docs/
```

## 学习节奏

1. 先读 `web/src/content/docs/roadmap.md`，理解全局路线。
2. 再按章节读文档，并把每章的 C# 代码落在对应的 `src/ChapterXX.*` 项目里。
3. 用 `apps/Playground` 做最小验证和实验。
4. 需要可视化回顾时，启动 Astro 站点浏览 Markdown 文档。

## 本地命令

```bash
dotnet restore
dotnet build
dotnet run --project apps/Playground/LlmsFromScratch.DotNet.Playground
```

```bash
cd web
pnpm install
pnpm dev
```

## 参考来源

- 官方仓库：<https://github.com/rasbt/LLMs-from-scratch>
- 中文仓库：<https://github.com/MLNLP-World/LLMs-from-scratch-CN>
- 本机参考副本：
  - `/Users/zhaoyiqi/Code/LLMs-from-scratch`
  - `/Users/zhaoyiqi/Code/LLMs-from-scratch-CN`
