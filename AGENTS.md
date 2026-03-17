# AGENTS.md

## Project Background

This repository exists to study and reimplement the learning path from **LLMs-from-scratch** in a `C# / .NET` context.

The original tutorial and its Chinese translation are both Python-first. This project is intentionally different:

- it uses `C# / .NET` as the primary implementation stack for the LLM code
- it keeps the chapter-by-chapter learning flow from the source material
- it uses `Markdown` to organize study notes and roadmap content
- it uses `Astro` only as the documentation and presentation layer

The main goal is to rebuild the concepts and code path of the tutorial with a pure `.NET` engineering mindset, instead of wrapping or calling Python implementations.

## Related Repositories

### 1. Original repository

- Name: `LLMs-from-scratch`
- Role: the main source tutorial and chapter structure for this project
- Local reference path: `/Users/zhaoyiqi/Code/LLMs-from-scratch`
- Upstream repository: <https://github.com/rasbt/LLMs-from-scratch>

### 2. Chinese repository

- Name: `LLMs-from-scratch-CN`
- Role: Chinese translation and localized learning companion for the same tutorial flow
- Local reference path: `/Users/zhaoyiqi/Code/LLMs-from-scratch-CN`
- Upstream repository: <https://github.com/MLNLP-World/LLMs-from-scratch-CN>

## Implementation Direction

This project is intended to implement the tutorial using a **pure `C# / .NET` stack** for the LLM system itself.

That means:

- core model code should be written in `C#`
- training, tokenization, attention, GPT structure, and finetuning experiments should be expressed in `.NET`
- Python code from the source repositories should be treated as reference material, not as runtime dependency
- the web layer is for reading and navigation, not for implementing the model logic

## Repository Intent

This repository is a learning and implementation workspace for:

1. understanding the full chapter flow of LLM construction
2. translating the tutorial ideas into `C# / .NET`
3. keeping notes, roadmap, and implementation progress in one place

## Guidance For Future Contributors And Agents

- preserve the chapter ordering from the original tutorial unless there is a strong reason to change it
- prefer `C# / .NET` implementations over external language bridges
- keep documentation aligned with the original tutorial concepts and the Chinese translation
- treat the two related repositories as learning references, not as code to copy blindly
