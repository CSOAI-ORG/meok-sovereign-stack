---
title: MEOK · Light
emoji: 🐝
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - HuggingFaceTB/SmolLM2-360M-Instruct
---

One OOWM interface with three explicit prompt profiles. This static Space runs
an existing open model in the visitor's browser after an explicit model-load click.
No hosted GPU or inference token is required. Browser model data can occupy several
hundred MB; storage is controlled by the visitor's browser.

Default browser model: `HuggingFaceTB/SmolLM2-360M-Instruct` at
`a10cc1512eabd3dde888204e902eca88bddb4951` (Apache-2.0), ONNX q4.
Transformers.js 3.8.1, using ONNX Runtime WASM or WebGPU.

These profiles are not independently trained specialist weights. The separate Kaggle
LoRA smoke artifact is not loaded by this browser client. No independent capability
measurement or frontier-level performance is claimed. Model output is untrusted text.

The source review includes the interface, runtime and training recipes.
Public release follows source review and M4 release authority.
