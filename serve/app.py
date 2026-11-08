"""Untested Gradio wrapper candidate; checkpoint reload evidence is separate."""
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
BASE = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ADAPTER_SHA256 = "0c197a14fa771718028a0ea151d30457fb0dd9d55755d510c45a2862ca6d4f13"
REPO = os.environ.get("MEOK_ADAPTER_REPO", "")
REVISION = os.environ.get("MEOK_ADAPTER_REVISION", "")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", REPO):
    raise RuntimeError("Explicit MEOK_ADAPTER_REPO is required")
if not re.fullmatch(r"[0-9a-f]{40}", REVISION):
    raise RuntimeError("Exact MEOK_ADAPTER_REVISION is required")

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROFILES = {
    "general": "Help with the request directly and concisely.",
    "evidence": "Separate observations, unknowns and proposed checks. Never invent sources or signed receipts.",
    "code": "Explain code clearly. Do not claim code ran unless an execution result was supplied.",
}
SYSTEM = "You are MEOK, a bounded assistant. State uncertainty. Offer suggestions only; execution, official assessments and approvals require separate evidence and authority. "
LOCK = threading.Lock()
MODEL = None


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    files = {name: Path(hf_hub_download(REPO, name, revision=REVISION))
             for name in ("adapter_config.json", "adapter_model.safetensors")}
    with files["adapter_model.safetensors"].open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ADAPTER_SHA256:
            raise ValueError("Adapter digest mismatch")
    config = json.loads(files["adapter_config.json"].read_text())
    if (config.get("base_model_name_or_path") != BASE or config.get("peft_type") != "LORA"
            or config.get("task_type") != "CAUSAL_LM" or config.get("r") != 8
            or config.get("lora_alpha") != 16 or config.get("auto_mapping") is not None
            or set(config.get("target_modules", [])) != {"q_proj", "v_proj"}):
        raise ValueError("Unexpected adapter configuration")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION,
        trust_remote_code=False, use_safetensors=True, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(files["adapter_config.json"].parent),
        is_trainable=False, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    MODEL = (model, tokenizer, device, str(dtype))
    return MODEL


def generate(prompt, profile):
    try:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000 or profile not in PROFILES:
            raise ValueError("Invalid input")
        with LOCK:
            started = time.monotonic()
            model, tokenizer, device, dtype = load_model()
            messages = [{"role": "system", "content": SYSTEM + PROFILES[profile]},
                        {"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
            length = inputs["input_ids"].shape[-1]
            if length > 2048:
                raise ValueError("Input token limit exceeded")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
            if not answer:
                raise ValueError("Empty model output")
            receipt = {"schema": "meok.adapter-inference-receipt/0.1", "id": str(uuid.uuid4()),
                "observed_at": datetime.now(timezone.utc).isoformat(), "status": "RUNTIME_OBSERVED",
                "base_model": BASE, "base_revision": BASE_REVISION, "adapter_repo": REPO,
                "adapter_revision": REVISION, "adapter_sha256": ADAPTER_SHA256,
                "specialist": profile, "device": device, "dtype": dtype, "prompt_tokens": length,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "output_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "generation": {"max_new_tokens": 128, "do_sample": False, "enable_thinking": False},
                "runtime": {name: version(name) for name in ("torch", "transformers", "peft", "gradio")},
                "duration_ms": round((time.monotonic() - started) * 1000),
                "quality": "UNMEASURED", "signature": None,
                "limitations": ["Unsigned server-reported inference, not independent evaluation.",
                                "Three prompt profiles share one bootstrap adapter."]}
            return answer, receipt
    except Exception as error:
        # Do not return provider error text, prompts, private URLs or credentials.
        name = type(error).__name__
        return f"Request could not be completed: {name}", {"status": "UNCHECKABLE", "error_type": name}


with gr.Blocks(title="MEOK adapter preview", analytics_enabled=False) as demo:
    gr.Markdown("# MEOK adapter preview\nUntested serving wrapper candidate. Quality UNMEASURED. "
                "This server receives your prompt; no prompt log is saved by this app.")
    profile = gr.Radio(list(PROFILES), value="general", label="Prompt profile — shared adapter")
    prompt = gr.Textbox(label="Request (maximum 2,000 characters)", lines=4)
    button = gr.Button("Ask MEOK")
    answer = gr.Textbox(label="Model output — may be incorrect", interactive=False)
    metadata = gr.JSON(label="Unsigned inference metadata")
    button.click(generate, [prompt, profile], [answer, metadata], concurrency_limit=1, api_name=False)
demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
if __name__ == "__main__":
    demo.launch(share=False, show_error=False)
