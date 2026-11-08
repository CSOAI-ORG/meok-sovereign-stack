# MEOK bootstrap acceptance — 8 November 2026

**Outcome: experimental trained artifact retained; product release not accepted.**

The private Kaggle training run completed20 optimisation steps on NVIDIA Tesla T4.
The saved adapter has112 finite tensors and matched the captured SHA-256 manifest.
A second Kaggle process verified those bytes, reloaded the pinned base and saved
adapter, and produced3 nonempty responses. This establishes a bounded training,
save and reload path. It is not an independent capability evaluation.

An explicit counterexample remains in reload-03: asked what a successful reload
demonstrates, the model said it demonstrates an ability to adapt and improve over
time. That conclusion is unsupported. The complete output is retained in the
private model's `evidence/reload-receipt.json`. This failure blocks any claim that
the adapter reliably maintains evidence boundaries.

| Surface | Observed state | Remaining acceptance |
| --- | --- | --- |
| Kaggle adapter training | Completed, recorded CUDA runtime and fixed model/source pins | Larger rights-reviewed training corpus and useful task evaluation |
| Saved adapter | Integrity checked; second-process reload produced3 responses | Independent replay, sealed holdout and human acceptance |
| Hugging Face archive | Private experimental artifact under the owner's account | csoai organisation private storage is full; public release held for M4 |
| M2/Ollama route | Model catalogue read succeeded earlier; actual router preflight timed out | Fresh completed inference through exact candidate code |
| Static Space | UI/profile/input controls inspected; source tests passed | Real browser model load and generation on a machine with storage headroom |
| Cloudflare R2 candidate | Auth, bounds and readback semantics tested with a mock bucket | Existing-bucket configuration and deployed readback via authorised release |
| NVIDIA/LF register | Pinned sources/licences recorded | Each planned integration needs its own runtime qualification |

No paid compute was started. The two Kaggle jobs used the existing free quota.
No base-model weights were downloaded to the Mac; only the4.6MB adapter and small receipts
were retained. The Mac remains below its5GiB floor; additional model downloads must
stay on hosted compute. Existing production services were not replaced.

Main Claude on M4 owns source merge, public release and any production deployment.
The Busabase proposal is draft-only (`Codex`, `autoMerge:false`) on the interim M4
copy. RunPod primary restoration is not established by that proposal.
