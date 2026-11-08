# MEOK · Light

A small, inspectable first build of the MEOK OOWM entry point: explicit specialist
routing, pinned model execution, an experimental LoRA adapter and receipts.
This is a development candidate, not a claim of frontier capability.

One conversation routes to named specialists. This build configures three
**prompt profiles** (general,
evidence, code). They are not independently trained specialist models. Historical
hive and clan counts have not been reconciled and are not asserted here.

## Contents

| Path | Purpose | Evidence boundary |
| --- | --- | --- |
| `meok_light/` | Standard-library Ollama runtime and metadata receipts | Requires matching pre/postflight digests and an actual response |
| `space/` | Free static Hugging Face UI, optional browser CPU/WebGPU inference | Uses pinned existing SmolLM2, not the experimental Qwen adapter |
| `serve/` | Gradio wrapper for the actual saved adapter | Syntax checked; wrapper not runtime tested or deployed |
| `training/` | Private Kaggle CUDA LoRA smoke, verifier and saved-weight reload | Synthetic development exercise, not an independent capability benchmark |
| `cloudflare/` | Authenticated R2 receipt storage candidate | Mock-tested; not deployed or proven durable in Cloudflare |
| `configs/components.json` | Pinned NVIDIA/HF/LF component register | Catalogued components are not claimed as integrated |
| `scripts/archive_candidate.py` | Integrity-checked private HF artifact retention | Does not publish a public model or deploy a Space |

## First experiment — 8 November 2026

Private Kaggle run `nicktempleman/meok-private-cuda-adapter-smoke` version 1
completed 20 LoRA steps on `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`, using NVIDIA Tesla T4 / CUDA 13.0.
Only GPU 0 was used. The 1,146,880-parameter adapter is 4,602,248 bytes, with
SHA-256 `0c197a14fa771718028a0ea151d30457fb0dd9d55755d510c45a2862ca6d4f13`.
All 112 saved tensors passed the local integrity/finite-value check.

There were 16 original synthetic training examples and six disjoint development
examples. Development target-token loss changed from 5.39024 to 4.64985. This
does not establish improvement on real tasks. Captured answers still contain
confused or incomplete reasoning. No Council score or independent quality result
is assigned to this own product.

## Try the source

Requires Python 3.9+ and an existing Ollama model matching `configs/routes.json`.
No runtime package install or model download occurs in this command:

```sh
python3 -m meok_light 'Explain what a signed receipt proves.' --specialist evidence
```

The endpoint defaults to loopback. HTTPS is required for non-loopback origins.
Unknown routes, identity changes, missing outputs and provider failures cannot
become successful inference receipts. Receipts store prompt/output hashes;
the CLI prints the response to the terminal. Receipts are unsigned and do not
establish correctness. Model alias/digest identities remain provider-reported.

Preview the static Space, then explicitly choose **Load model**:

```sh
python3 -m http.server 8769 --bind 127.0.0.1 --directory space
```

It downloads several hundred MB into the visitor's browser, with no API key,
hosted GPU or server-side prompt store. UI controls were inspected; actual browser
model inference needs a machine with sufficient storage. The M2 Ollama probe
timed out on 8 November; configured routes are not reported as live.

## Validate and reproduce

```sh
python3 -m unittest discover -s tests -v
node --test tests/*.test.mjs
node --check space/worker.js
```

See [training/README.md](training/README.md) for bounded private GPU training and
reload recipes. Jobs require an available free Kaggle GPU quota; there is no
automatic paid fallback. Separate reviewed training data, a sealed holdout and
independent acceptance are needed before promotion.

## Release and alignment

Codex contributes a source branch and PR. Main Claude on M4 owns merge/close and
production release. No laptop/pod Cloudflare deployment is provided. The R2
example has no active account, bucket or public route. Existing Council measurement
Spaces remain intact. Private HF artifact retention is not an inference deployment.
See [docs/STACK.md](docs/STACK.md) for current platform and licence boundaries.

The June vault, identity and competition files are historical source and
are not invoked by this runtime. Legacy default signing seeds and LLM judge
outputs must not be treated as trusted measurement or identity.

Repository code: MIT. Models and component dependencies retain their own licences.
