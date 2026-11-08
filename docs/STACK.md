# MEOK first build: component boundaries

Observed 8 November 2026. This document separates source candidates from executed integrations. The component catalogue does not install packages, run models, start paid hardware or change production.

One conversation entry point routes to named specialists. A specialist may reference a pinned base model, an independently versioned adapter, allowed tools and its evidence record. Configured groups are not proof of trained specialists, a learned mixture-of-experts model or inter-agent consensus.

## Free first release

The first executable paths are explicit browser inference, an existing inference endpoint, and a bounded private Kaggle training experiment when the account's free GPU quota permits it. A Hugging Face static Space can display the app, model card and receipts without provisioning an hourly GPU. Static hosting does not supply server-side inference: browser inference downloads weights into the visitor's device after an explicit load action. Model and dataset repositories hold artifacts; an authenticated write-capable credential is still required to publish them.

A local Ollama response proves inference for the actual runtime/model digest used. A two-step LoRA run proves training execution and changed adapter bytes. Neither proves improved model quality. Improvement needs a frozen held-out evaluation against the unchanged base, item-level results, contamination checks and sufficient sample size. The own-product results remain engineering evidence rather than a Council ranking of MEOK.

Do not combine unrelated weight architectures or ingest every public dataset. Track each base revision, licence, source, data rights and train/validation/test split. Keep evaluation banks out of training. Source availability does not imply permission to train, redistribute or use a model commercially.

## Model selection

| Role | Exact model and revision | Boundary |
|---|---|---|
| Small trainable base | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0; native Transformers; first bounded LoRA candidate |
| Larger small specialist | `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0; independently test runtime and memory |
| NVIDIA candidate | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16@dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` | NVIDIA Nemotron Open Model License; custom model code; about 7.95 GB of BF16 parameter bytes before runtime overhead |
| Existing owned gate artifact | `csoai/csoai-gate-ft2@0be3ead219b62e6b204ddb61831873d24c72c2d2` | Qwen2-0.5B-derived GGUF; Apache-2.0; training lineage still needs inspection |
| Existing owned model artifact | `csoai/csoai-unified-33@69a2bfbf0b2ac1bc14ce21f82b4a7ca700089499` | Llama-3.2-3B-Instruct-derived GGUF; Llama 3.2 licence |

The [Qwen card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md) documents Transformers >=4.51.0. Keep `trust_remote_code=False` for the generic loader. The [Nano4B card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/README.md) uses custom code and documents vLLM >=0.15.1; review the exact code revision before enabling it. The 30B-A3B NVIDIA checkpoint has about 31.6B total parameters: active-parameter naming does not make its full weights fit like a 3B dense model.

Nemotron-Flash-1B and 3B are excluded from the default commercial base selection because their inspected cards declare CC-BY-NC-4.0. This is a different licence from Nano4B's. No NVIDIA checkpoint is asserted to be trained, integrated or runtime-tested by this catalogue.

The selected browser base is the official `HuggingFaceTB/SmolLM2-360M-Instruct@a10cc1512eabd3dde888204e902eca88bddb4951`, which supplies ONNX weights and an Apache-2.0 model card. It is a small model with limited capability. Browser downloads, memory fit and output must be tested separately from source/unit tests. A larger conversion candidate, `onnx-community/Qwen3-0.6B-ONNX@58ca03a5e7b354b480b1a82b34bde8dc7b854cc0`, identifies Qwen3-0.6B as its base and demonstrates Transformers.js WebGPU inference in its [README](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/blob/58ca03a5e7b354b480b1a82b34bde8dc7b854cc0/README.md), but has no licence field or LICENSE file at that revision. Its base is Apache-2.0; its conversion-repository licence is undeclared, so it was not selected as the default.

## NVIDIA candidates

All rows below are `CATALOGUED`, with `integrated:false` and `runtime_tested:false` in [components.json](../configs/components.json). Source commit and licence-file URLs are pinned there. These source pins are not installation commands or a tested environment lock.

| Component | Purpose | Qualification needed |
|---|---|---|
| NeMo AutoModel | NVIDIA SFT/LoRA path for Nano4B | GPU fit, exact recipe and successful checkpoint reload |
| NeMo Guardrails library | Optional input/output policy checks | Tests of the selected policies, latency and false positives |
| NeMo Curator | Corpus filtering and deduplication at scale | Data rights, split isolation and processing receipts |
| NeMo Evaluator | Repeatable offline model evaluation | Pinned method, held-out items and raw results |
| TensorRT-LLM | Accelerated NVIDIA serving | GPU compatibility and parity against original checkpoint |
| Model Optimizer | Quantization/distillation experiments | Before/after quality, memory and throughput evidence |

[NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) provides Nano4B SFT and LoRA examples; the shown launch uses eight processes, so it is not proof of single-GPU fit. [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html) lists NemotronH and Qwen3 support; support depends on the specific version, model and hardware. The [open-source Guardrails library](https://docs.nvidia.com/nemo/guardrails/latest/home) is distinct from the enterprise Guardrails microservice. No paid microservice is part of the free build.

## Linux Foundation ecosystem candidates

These four projects have a verified primary foundation source. They are optional future integrations rather than new dependencies. Their inclusion does not establish a relationship between MEOK and a foundation. LFX is an ecosystem of project services, not one runtime library.

| Project | Verified home | Purpose |
|---|---|---|
| MLflow | [Linux Foundation](https://www.linuxfoundation.org/press/press-release/the-mlflow-project-joins-linux-foundation) | Local experiment tracking and artifact registry |
| ONNX | [LF AI & Data](https://lfaidata.foundation/projects/) | Portable model interchange when export/parity is supported |
| OpenTelemetry | [CNCF](https://www.cncf.io/projects/opentelemetry/) | Request/run traces without exporting prompts or credentials by default |
| KServe | [CNCF](https://www.cncf.io/projects/kserve/) | Future Kubernetes inference only if an existing cluster warrants it |

KServe's current source identifies CNCF, so it is not labelled as a current LF AI & Data project here. Ray was not assigned an LF affiliation because this bounded review did not establish one from a primary source. No Kubernetes cluster, MLflow server or telemetry exporter is provisioned.

## Hugging Face capacity observed

The public `csoai` author API returned 7 model repositories and 32 Spaces. Five model sibling lists contain GGUF files. The first dataset page returned 100 entries and was capped; that is not a complete dataset count. Listings establish availability of metadata, not model quality or successful runtime jobs.

Of the 32 Spaces, 25 have `sdk:static` and Hub runtime `RUNNING`. Seven have compute SDKs:

| Space | SDK | Hub stage | Current / requested hardware |
|---|---|---|---|
| `csoai/gspc-mcp` | Gradio | RUNNING | cpu-basic / cpu-basic |
| `csoai/gspc-node` | Docker | RUNNING | cpu-basic / cpu-basic |
| `csoai/gspc-auto-measure` | Gradio | RUNNING | zero-a10g / zero-a10g |
| `csoai/gspc-governance-leaderboard` | Gradio | SLEEPING | null / cpu-basic |
| `csoai/gspc-mill` | Docker | SLEEPING | null / cpu-basic |
| `csoai/public-surface-watch` | Docker | PAUSED | null / cpu-basic |
| `csoai/gspc-council-os` | Gradio | PAUSED | null / cpu-basic |

The static Space IDs are `gspc-gov`, `gspc-agi`, `gspc-prv`, `gspc-asi`, `gspc-oss`, `gspc-mach`, `gspc-care`, `gspc-xr`, `gspc-det`, `gspc-art5`, `gspc-swarm`, `gspc-board`, `council-space`, `council-os`, `ras-assess`, `claimguard`, `games-catalog`, `gspc-interop`, `gspc-affect`, `living-catalog`, `council-coliseum`, `gspc-verify`, `gspc-explorer`, `gspc-pulse-dashboard`, and `gspc-sidebar`, all under `csoai/`. These are Hub-reported states; this inventory did not call model inference.

No existing Space in that public inventory is named MEOK. The inspected `gspc-auto-measure` source rotates a hardcoded set of third-party models over GSPC banks; it is not a general MEOK inference/training endpoint. Its ZeroGPU metadata is worth retaining, but it does not authorize repurposing a public measurement service or prove free training capacity.

[Current Spaces documentation](https://huggingface.co/docs/hub/spaces-overview) says Static Spaces are free; creating new Gradio or Docker Spaces requires a paid plan even when CPU Basic has no hourly fee. Existing space allocations are observed separately from new-space eligibility. The Hugging Face connector's discovery tools returned `UNAVAILABLE`; direct public Hub API reads succeeded. Authentication and repository write permission require their own readback.

The [Cloudflare receipt adapter](../cloudflare/README.md) now has source and Node mock tests for authenticated content-addressed R2 storage. It is not deployed. A configuration or passing R2 mock does not establish cloud durability, free quota or a working inference route. Integration still goes through the owning GitHub release workflow; no Cloudflare deploy is performed by this candidate.
