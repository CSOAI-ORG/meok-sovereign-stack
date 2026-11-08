# Kaggle development and MEOK artifact handoff

This source connects the MEOK Hugging Face build's `training/kaggle_smoke.py`
output contract to an offline receiving check and a complete GSPC axis inventory.
It does not start a job, train a model, publish a notebook, load weights, sign a
card, admit a result or change the serving route.

## Training handoff

1. Retain the exact reviewed training script before dispatch. Generate its
   contract **outside** the eventual notebook output directory:

   ```sh
   python3 -m kaggle_bridge contract --source training/kaggle_smoke.py \
     --output /private/reviewed-source-contract.json
   ```

2. Use the Hugging Face lane's private kernel builder and existing authorised
   Kaggle quota. Record the actual kernel version and quota. `kernels push`
   normally starts execution; `--no-run` is the staging option. A request to
   build this bridge is not a new paid-compute budget.
3. Download the finished private notebook output into a fresh, stable directory.
   Keep the full source, run ID/version, log, receipt, data and adapter files.
   Do not derive the expected contract from a script supplied by that output.
4. Verify the output against the independently retained contract:

   ```sh
   python3 -m kaggle_bridge verify --run /private/download/meok-artifacts \
     --contract /private/reviewed-source-contract.json \
     --output /private/adapter-check.json
   ```

The check rejects incomplete steps, unknown schemas, source/model/dependency
drift, altered or overlapping data splits, false improvement flags, missing or
changed weights, extra adapter files, symlinks, path traversal, duplicate JSON
keys and nonfinite numbers. Streaming hashes are bounded; JSON files have size
limits. The input must be a completed export without concurrent writers.

`ARTIFACTS_VERIFIED` means exported bytes match the supplied contract. It is
**not** a signature check or proof of execution. Notebook metrics remain
producer-reported. The next independent steps are PEFT loading with the same
base revision, a new held-out evaluation, base-versus-adapter comparison under
the same budget, and review before promotion. The 16 training and 6 development
fixtures are a software smoke test, not independent evidence of useful learning.

No model weights, notebook secrets or raw production conversations belong in
Git. No external benchmark answer bank is a training source. Add open models
through a registry of exact revisions, licences, source-code review, model
architecture, memory requirements and measured task value; a public Hub listing
does not grant unrestricted training or redistribution rights.

## NVIDIA activation order

Run `python3 -m kaggle_bridge.nvidia_probe --exercise` **inside the target
Kaggle image**. It records package versions and `nvidia-smi` output, then attempts
bounded CUDA arithmetic and cuDF arithmetic in subprocesses. It installs nothing.
Missing packages or a failed operation remain UNCHECKABLE; packages alone are
not working integrations. Keep the emitted JSON with each run.

| Component | Concrete role | Activation requirement |
|---|---|---|
| CUDA + PyTorch SDPA | Existing bounded LoRA smoke and inference | Actual GPU, CUDA and dependency receipt |
| RAPIDS cuDF/cuML | Corpus deduplication, analysis and classical baselines | Target-image import + operation; preserve source rights and split boundaries |
| NeMo Curator | Larger corpus preparation | Reviewed source/data rights and demonstrated benefit over simple CPU processing |
| NeMo AutoModel | NVIDIA-native adaptation experiments | Exact supported model, precision, GPU memory and runtime proof |
| Model Optimizer | Quantisation/distillation experiments | Compare the derived model with its exact base on disjoint data |
| NeMo Evaluator | External benchmark harness integration | Frozen task adapter, permitted bank and independent replay |
| NeMo Guardrails | Optional input/output controls | Tests of the configured rules; no unsupported assurance claim |
| TensorRT-LLM | Optimised GPU serving | Exact GPU/model support and output/performance comparison; not assumed on T4 |

Use the other lane's `configs/components.json` for component revision and
licence pins. Catalogue entries remain CATALOGUED until the intended operation
has a captured result. Prefer NVIDIA where it demonstrably helps; the light
runtime does not require every heavy framework to be installed.

## Every axis remains visible

Capture the public board, retaining source URL and capture time. Generate a
coverage inventory from those exact bytes:

```sh
python3 -m kaggle_bridge coverage --board /private/gspc.json \
  --observed-at 2026-11-08T04:00:00+00:00 --output /private/coverage.json
```

The timestamp above is an example; use the actual capture time. The current
schema is checked explicitly. Every axis in `axes` appears once; new or unknown
kinds remain visible as UNSUPPORTED. Counts are derived, never hardcoded to 34.
Board status is distinct from this run's measurement and execution states. The
CLI starts every run UNMEASURED/NOT_RUN, copying no board score into a new result.
The Python API accepts an independently supplied receipt validator; a caller
flag alone cannot establish replay. That validator must establish the claimed
axis, subject, method, inputs, outputs and replay status. Admission stays separate.

At the 8 November capture the categories are 14 model-comparison, 9 deterministic
facts and 11 measured-subject axes. Only model-comparison axes belong in a generic
LLM inference queue. Cryptography, ledger/public-series facts and subject-family
observations require their own methods. None becomes an aggregate MEOK score.

Reuse Council's canonical grader (`harness/gspc-top100/mill_hub_queue.py`),
`harness/bench` Kaggle/Inspect adapters and unsigned intake. Do not implement a
new grader here. Kaggle's hosted benchmark proxy and private GPU notebooks are
different execution modes. A MEOK HF model is not automatically available in
the hosted proxy catalogue. MEOK engineering outputs are excluded from Council
self-ranking and do not enter public third-party comparisons.

## Release boundaries and checks

The HF download repair is merged as Council PR #4049; all-14 runner expansion
in PR #3883 was open at review. Neither status proves a successful new GPU run.
The prior GovBench replay (36 responses, 19 label matches) equalled the constant
COMPLY baseline and is not evidence of useful model improvement or independent
label validation. Preserve that counter-signal when selecting the next experiment.

Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v`.
This bridge uses Python 3.10+ standard library only. The separate Kaggle SDK
requires its own qualified Python/dependency environment. Repository changes go
through CSOAI-ORG PRs; only main M4 Claude merges/closes. Publication needs the
outward gate and sent log. No public Kaggle task is published by this bridge.

Official platform sources and their current limitations are recorded in
`kaggle-research.md`. NVIDIA hardware support:
https://nvidia.github.io/TensorRT-LLM/supported-hardware.html
