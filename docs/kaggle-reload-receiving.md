# Receiving a saved-adapter reload

The `verify-reload` command links a producer's saved-adapter reload receipt to
an original training export that this receiver checks again. It verifies the
three expected prompt identities and outputs against separately retained source
and generation pins. It does not run the model or grade the answers.

First retain a contract from the reviewed replay script outside run outputs:

```sh
python3 -m kaggle_bridge reload-contract --source /reviewed/replay.py \
  --output /reviewed/reload-contract.json
```

The extractor parses literal source constants without importing or executing
Python. Ambiguous receipt definitions, duplicate keys and dynamic unpacking
are rejected. The resulting contract pins source bytes, training-source digest,
adapter digest, exact prompts and generation settings. It is a declared review
input, not an approval or a signed execution record. Do not construct expected
pins from the receipt being checked.

Check the actual export and saved-adapter reload together:

```sh
python3 -m kaggle_bridge verify-reload \
  --run /private/original-kaggle-export \
  --contract /reviewed/training-source-contract.json \
  --receipt /private/reload-receipt.json \
  --replay-contract /reviewed/reload-contract.json \
  --output /private/reload-check.json
```

The receiver recomputes training-file hashes, binds the supplied training
receipt through raw and canonical digests, checks the reload's source/model/
adapter identity, checks the complete expected prompt set, and recomputes each
UTF-8 prompt/output hash. Token counts and times must be finite and within the
stated token budget. A changed, missing, duplicate or failed sample cannot be
hidden by an overall COMPLETED flag. False effect and improvement claims fail.

`RELOAD_RECEIPT_VERIFIED` reports consistency of unsigned producer evidence.
Execution authentication, independent replay, answer correctness and general
improvement remain unestablished. A recorded fluent but incorrect answer still
passes a byte-consistency check; an independent task-specific quality evaluation
is a different required step.

## Preserve packaging history

The original training receipt includes the adapter README. If a Hugging Face
model card replaces that file, keep the original export intact and record the
packaging transformation separately. Never change the training receipt to make
its hashes match a new README. Use immutable archive versions while receiving;
a mutable package being assembled can change filenames between reads.

## Receiving observation, 8 November 2026

The actual 20-step training export passed this receiver against the previously
retained source contract: three adapter files, 4,608,318 total bytes. The separate
saved-adapter receipt passed source/identity and all three output checks.
Kaggle independently reported both jobs COMPLETE. These are bounded execution
and integrity observations, not proof of useful learning. One recorded reload
answer incorrectly asserted that successful reload demonstrates improvement;
that remains a quality limitation rather than a passing capability result.

A separate private Kaggle probe performed CUDA and cuDF arithmetic on Tesla T4
hardware with no internet, package installation, model download or training.
Both returned the expected sum. CUDA 13.0 / PyTorch 2.11.0+cu130 and cuDF 26.6
were observed. cuML was installed but not exercised. NeMo, Model Optimizer and
TensorRT-LLM were absent. Image-default Transformers/PEFT differed from the
qualified training pins, so retain explicit dependency pins for reproduction.
No source change here publishes or promotes the adapter or creates a GSPC cell.
