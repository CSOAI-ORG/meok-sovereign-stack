# Kaggle integration evidence — 8 November 2026

Kaggle is a bounded development and reproducible replay environment. MEOK's
private engineering evaluations remain separate from independent third-party
GSPC admission. Every axis needs its own method and valid evidence; an LLM task
cannot replace deterministic facts or subject-family observations.

## Verified platform behavior

- [Benchmark CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md):
  push, run, status, logs, downloads and model discovery exist. Proxy inference
  quota is separate from GPU/TPU hours. A local token can see fewer models than
  the hosted executor. Re-push must repeat dataset attachments. Publication also
  publishes the backing notebook by default; CLI unpublishing is unsupported.
- [Kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md):
  push runs the notebook unless `--no-run` is passed. T4, A100, L4, H100 and RTX
  Pro 6000 IDs are listed, but some require competition or administrator access.
  Actual account hardware and quotas require a fresh authenticated inventory.
- [KaggleHub](https://github.com/Kaggle/kagglehub): datasets, models and notebook
  outputs have explicit version handles. Retain these plus file digests. Inside
  Kaggle, attached resources use the shared cache rather than local VM storage.
- [Notebook documentation](https://www.kaggle.com/docs/notebooks): dataset and
  notebook collaborator permissions are separate. Notebook versions retain
  code, outputs, logs, sources and their Docker environment version.
- [NVIDIA 26.10 Kaggle guide](https://docs.nvidia.com/datascience/deployment/26.10/platforms/kaggle/index.html):
  the latest Kaggle environment includes cuDF and cuML. They suit data cleaning,
  deduplication, feature analysis and baselines. Freeze the resolved environment
  for evaluation; the exploratory `always latest` setting is not a version pin.
- [SDK user guide](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/user_guide.md):
  failures can be collected separately and retried. Retain expected item IDs,
  completed/error counts and attempt provenance; do not silently shrink the
  denominator. [SDK requirements](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/pyproject.toml)
  require Python 3.11+. A newer SDK needs qualification before replacing the
  previously qualified adapter.

## Integration contract

Record source commit; method/bank hashes; training/development/holdout identity;
rights decision; model revision; tokenizer and prompt template hashes; generation
settings; actual GPU/driver/CUDA/package versions; task/run versions; budget;
outputs and hashes. Never train on private holdouts or their adjudications.
Proxy execution and HF-model kernel execution are different modes. A new HF
model is not automatically included in Kaggle's proxy catalog.

Start with one pinned public development bank, third-party baseline comparisons,
canonical/Kaggle/Inspect replay parity and tamper/missing-output controls. Replay
parity does not independently establish label correctness. Exported task inputs,
traces, logs and output files need disclosure review before publication.

## Fresh source observations

The public board fetched through curl reported `csoai.gspc-axes/0.5` with an
`axes` list of 34 entries: 14 `model-comparison`, 9 `deterministic-facts`, and 11
`measured-subject`. These are captured board declarations, not new Kaggle runs.
Python urllib returned HTTP 403 on the same route; preserve transport failures
as observed rather than claiming the route universally works.

GitHub API branch heads observed on 8 November 2026:

- Kaggle/kaggle-benchmarks `ci`: `5ffefc57696e26b172fc3626392731ebecbb1195`.
- Kaggle/kaggle-cli `main`: `fdc788faffe3f29ae317480325c44080a00a5434`.
- Kaggle/kagglehub `main`: `03c1f544fb5e87019f5d77e4fc129455aa573278`.

The SDK local-development page still describes internal staging while the CLI
documents token initialization. Use the supported CLI/account discovery route;
do not copy staging URLs. Account authentication, quotas, GPU entitlement,
Kaggle acceptance, production installation and end-to-end execution are not
established by this research.
