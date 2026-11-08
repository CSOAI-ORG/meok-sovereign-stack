---
title: MEOK adapter preview
emoji: 🐝
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
models:
  - Qwen/Qwen3-0.6B
---

This is an **untested, undeployed serving wrapper candidate** for the actual MEOK
bootstrap LoRA adapter. It is separate from the static browser demonstration.
The training/checkpoint-reload evidence does not establish this wrapper's runtime
behaviour, useful improvement or end-user acceptance. Quality remains UNMEASURED.

Required configuration has no default; startup stops until both values are provided:

- `MEOK_ADAPTER_REPO`: the explicit Hub owner/repository containing the adapter.
- `MEOK_ADAPTER_REVISION`: its exact 40-character lowercase hexadecimal commit.
- `HF_TOKEN`: a Space secret with read access if the selected repository is private.

The repository is supplied privately by the owner; `owner/MEOK-candidate` is an
illustrative placeholder. Configure it only after upload and revision readback.
The owner also supplies the exact commit privately. Never use `main` as the revision.
Credentials belong in Space Secrets. Files, UI fields, public configuration and output metadata contain no tokens.

The adapter file must hash to
`0c197a14fa771718028a0ea151d30457fb0dd9d55755d510c45a2862ca6d4f13`.
Its configuration is checked against Qwen3-0.6B, causal-LM LoRA rank 8, alpha 16 and
q/v projections. The base is pinned to
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` and loaded using native
Transformers and safetensors with remote code disabled. The loaded adapter is
frozen. Only the two named adapter files are downloaded; executable Hub code is
not imported. A changed checkpoint requires a source review and new digest.

CPU uses float32. CUDA uses bfloat16 only when supported and otherwise float32,
including T4-class GPUs without native bfloat16 support. No automatic GPU upgrade
is requested. Memory, latency and output still need a real serving test. The first
request downloads the pinned model/adapter on the serving host.

The UI has three explicit prompt profiles sharing the same adapter. Generation is
serialized, prompts are limited to 2,000 characters and 2,048 input tokens without
silent truncation, and output is capped at 128 tokens. Generated text is displayed
as text and never executed. Errors expose their class only. The app does not save
prompt logs; Gradio necessarily handles submitted text in server memory, and the
hosting platform has its own access/retention controls. No analytics or public
Gradio share tunnel is enabled. Receipts contain hashes and metadata, not prompt
or answer text, and are unsigned server reports.

Activation is limited to an **existing authorised CPU/free allocation**, after M4
release review and an explicit confirmation of the hosting account's current
eligibility. [Current HF documentation](https://huggingface.co/docs/hub/spaces-overview)
requires a paid plan for creating new Gradio/Docker compute Spaces; CPU Basic's
free CPU allocation does not waive that requirement. This source creates no Space,
changes no hardware and spends nothing. Keep any private adapter-serving preview
private until its intended access and release state are reviewed.

Package existence/direct constraints were read from primary PyPI metadata on
8 November 2026, including [Gradio 5.49.1](https://pypi.org/pypi/gradio/5.49.1/json),
[Hub 0.36.0](https://pypi.org/pypi/huggingface-hub/0.36.0/json) and
[PyTorch 2.9.0](https://pypi.org/pypi/torch/2.9.0/json). These are candidate pins,
not a fully locked or tested deployment environment. Gradio queue/click APIs were
checked against its [5.49.1 source](https://github.com/gradio-app/gradio/tree/gradio%405.49.1/gradio).
PEFT's [pinned-version API](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model)
documents frozen adapter loading. No dependency install or model download was
performed when authoring this wrapper.
