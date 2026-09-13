# Katali Lab

A pure-C, CPU-only inference engine for large MoE and hybrid-attention LLMs, plus a Win32 chat UI on top of it. This repo ships the Windows binaries only — no source.

## What it does

Katali Lab runs quantized LLM checkpoints that have been converted to the **EQS1** pack format. It's built for machines where the full model doesn't fit in RAM:

- **Memory-mapped dense weights** — attention, norms, embeddings, router gates, and dense FFN blocks are loaded lazily, layer by layer.
- **Streamed expert weights** — routed MoE experts (often 100+ GB) are read on demand into a bounded, byte-LRU cache with background prefetching, instead of requiring everything resident in RAM.
- **Quantized kernels** — Q8_0 for attention/dense, Q4_0 for MLP/expert blocks, with runtime AVX2/FMA/F16C dispatch so one binary runs on both modern and older CPUs.
- **CPU-only** — no GPU required, no CUDA/driver setup. The forward pass runs entirely on the CPU with a small Win32 thread pool for parallelism.

It supports two model families through one driver:

| Family | Attention | FFN |
|---|---|---|
| Qwen3-MoE / dense | Grouped-query attention (optional sliding window, partial RoPE) | Dense or expert MoE |
| GLM-5.3-Flash | KDA linear + MLA/DSA hybrid | Dense (first layers) then sigmoid-gated MoE |

Performance scales with storage speed — an NVMe-backed pack streams experts noticeably faster than a SATA one, since expert weights are pulled from disk on demand rather than fully preloaded.

## Tested with

Katali Lab has been run end-to-end against real, large MoE checkpoints, including:

- **GLM-5.3-Flash** (~167 GB pack)
- **Qwen3-235B-A22B** and **Qwen3-30B-A3B**

Both families load their EQS1 pack, stream experts from disk, and produce coherent answers on ordinary prompts (e.g. "What is the capital of the Philippines?" → "The capital of the Philippines is **Manila**."). Generation speed varies a lot with hardware, mainly disk speed for the expert stream and available RAM for the cache — a big pack on an NVMe drive is noticeably faster than the same pack on a SATA drive.

## Models

EQS packs are hosted on Hugging Face: **[huggingface.co/katalidevai](https://huggingface.co/katalidevai)**

- [Qwen3-30B-A3B (Katali Lab EQS, q4)](https://huggingface.co/katalidevai/qwen3-30b-katali-lab-eqs) — ~17 GB, ready to download and run
- GLM-5.3-Flash and Qwen3-235B-A22B packs are too large to host (167 GB+) and are not published; convert your own with the packing tools if you need them

## Contents

- `katali-lab.exe` — CLI: `inspect` (open a pack), `bench` (stream expert blocks), `run` (generate), `verify` (static layout/shape checks)
- `katali-lab-chat.exe` — Win32 chat GUI: multi-turn conversations, session history, resumable state (re-sending the same history skips re-processing the shared prefix). Spawns `katali-lab.exe` as a subprocess, so keep both files in the same folder.

## Quick start

In PowerShell (the default on Windows 11), prefix with `.\` to run from the current folder:

```
.\katali-lab.exe --help
.\katali-lab.exe run --model PATH_TO_PACK -p "Hi" --n 32 --verbose
```

In Command Prompt (cmd.exe), the plain name works too:

```
katali-lab.exe --help
```

Or just double-click `katali-lab-chat.exe`, point it at a model pack directory, and start chatting.

## Requirements

- Windows, x86-64 CPU
- A model pack already converted to the EQS1 format (`dense.eqs` + optional `experts.eqs`)
- Enough free RAM for the expert LRU cache; enough disk throughput (NVMe recommended) if the pack is larger than RAM

## Support

Developed by: Joan Apita
Support: katali.dev.ai@gmail.com
