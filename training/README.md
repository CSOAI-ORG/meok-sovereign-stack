# MEOK private CUDA adapter smoke

`kaggle_smoke.py` is a self-contained 20-step LoRA optimisation run for a private
Kaggle GPU notebook. It uses the notebook's NVIDIA CUDA PyTorch build and SDPA,
a commit-pinned Qwen3-0.6B base, pinned Transformers/PEFT dependencies, and 16
original synthetic training examples. Six disjoint development examples provide
before/after loss; three also retain both generations.

This is a bootstrap software smoke test, **not a capability benchmark or evidence
of general improvement**. The development set is not a sealed independent
holdout. Do not promote this adapter to a default route based on its loss alone.
No external benchmark labels, private conversations or scraped source text are
included. Fixture rights and source digests are in the receipt.

Build a private script bundle (no submission):

```sh
python3 training/build_kaggle_bundle.py --owner YOUR_KAGGLE_USER \
  --slug meok-private-cuda-smoke --output /path/to/new/private-bundle
```

Submit only through the owner's existing free Kaggle quota after inspecting
`kernel-metadata.json`; keep `is_private: true`. The builder cannot establish
account quota or platform billing policy. It does not select paid hardware,
upload a model, publish a notebook or contact Hugging Face with a write token.

The script installs pinned text-model libraries into the disposable notebook
environment, retains its CUDA-enabled torch, and removes unused torchvision and
librosa packages to avoid text-import conflicts. GPU and exact torch/CUDA versions
are observed at runtime and recorded. Without CUDA the run stops with an error. On older GPUs
it uses float32; supported GPUs use bfloat16. The whole process has a 20-minute
deadline, adjustable from 3 to 30 minutes, and 10–30 optimiser steps.

Outputs under `/kaggle/working/meok-artifacts`:

- `receipt.json`: terminal status, exact model and source identity, device,
  versions, sample counts, losses, original outputs, settings and limitations.
- `adapter/adapter_model.safetensors` and PEFT config, with SHA-256 manifest.
- `data.json`: exact synthetic examples and rights metadata.

Only LoRA weights are saved. A fresh output directory is required for every run.
An incomplete or interrupted run is not evidence that an adapter is usable; require
`COMPLETED`, finite losses, matching steps and matching manifest digests.

After downloading outputs, run the dependency-free integrity validator using the
SHA-256 of the exact script submitted to Kaggle:

```sh
python3 training/verify_artifacts.py /path/to/meok-artifacts \
  --expected-source-sha256 EXACT_SUBMITTED_SCRIPT_SHA256
```

It checks the pinned base identity, completed steps, recorded CUDA device,
finite losses, data-split digests, adapter configuration, all manifest hashes,
112 LoRA tensors and finite weights. Traversal paths, symlinks, extra adapter
files and overlapping tensor offsets are rejected. `INTEGRITY_CHECKED` means
these local bytes satisfy that contract; it does not authenticate their origin,
execute the saved adapter or establish capability improvement.

`build_replay_bundle.py` prepares a second private Kaggle process that mounts the
completed training kernel's outputs, verifies the exact adapter SHA-256, loads
the saved adapter over the same pinned base, and captures three new generations.
It performs no optimisation and has a 15-minute deadline:

```sh
python3 training/build_replay_bundle.py --owner nicktempleman \
  --slug meok-private-adapter-reload --output /path/to/new/replay-bundle
```

This builder is deliberately pinned to the first completed training run's source
and adapter hashes. A different run requires review and new explicit pins. The
generated script embeds SHA-checked local helpers and writes
`/kaggle/working/meok-replay/receipt.json`. Successful reload establishes that
the saved adapter can produce nonempty outputs in another process on the same
platform. It is not an independent capability evaluation or evidence of better
answers.

References checked 2026-11-08:

- [Pinned base card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md)
- [PEFT quicktour](https://huggingface.co/docs/peft/v0.17.0/en/quicktour)
- [Kaggle kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
