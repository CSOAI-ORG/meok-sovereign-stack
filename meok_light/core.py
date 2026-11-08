"""Small standard-library runtime. Model text never becomes executable actions."""
import hashlib
import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_BYTES = 1_048_576
SYSTEM = ("You are MEOK, a SOVOS-light assistant. State uncertainty plainly. "
          "Never invent evidence, sources, model capabilities, completed actions or scores. "
          "You cannot execute tools, approve changes, or certify systems. "
          "Distinguish model-generated suggestions from observed facts. ")

def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Provider redirects are not allowed")

def endpoint_url(base):
    u = urllib.parse.urlsplit(base)
    if u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
        raise ValueError("Endpoint must be an origin with no credentials, query or path")
    try:
        local = ipaddress.ip_address(u.hostname or "").is_loopback
    except ValueError:
        local = u.hostname == "localhost"
    if u.scheme != "https" and not (u.scheme == "http" and local):
        raise ValueError("Use HTTPS or an HTTP loopback endpoint")
    return base.rstrip("/")

def http_json(url, payload=None, timeout=90):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Provider response exceeds limit")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Provider returned a non-object")
    return parsed

def load_routes(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "meok.routes/0.1":
        raise ValueError("Unsupported route schema")
    routes = data.get("routes", {})
    if not routes or "general" not in routes:
        raise ValueError("A general route is required")
    for name, route in routes.items():
        if route.get("provider") != "ollama" or not isinstance(route.get("model"), str):
            raise ValueError("Unsupported route provider/model")
        pin = route.get("digest", "")
        if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            raise ValueError("Every runnable route requires a model digest")
        for key in ("hive", "clan", "instruction"):
            if not isinstance(route.get(key), str) or not route[key]:
                raise ValueError("Missing route metadata: " + key)
    return data

def run(prompt, specialist, config, endpoint="http://127.0.0.1:11434", transport=http_json,
        max_tokens=128, timeout=90):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8_000:
        raise ValueError("Prompt must contain 1–8000 characters")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
        raise ValueError("Token limit must be 1–512")
    if not 1 <= timeout <= 180:
        raise ValueError("Timeout must be 1–180 seconds")
    routes = config["routes"]
    if specialist not in routes:
        raise ValueError("Unknown specialist; choose a configured route")
    route = routes[specialist]
    endpoint = endpoint_url(endpoint)
    started = time.monotonic()
    receipt = {
        "schema": "meok.inference-receipt/0.1", "id": str(uuid.uuid4()),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "UNCHECKABLE", "specialist": specialist,
        "hive": route["hive"], "clan": route["clan"], "provider": route["provider"],
        "model": route["model"], "expected_digest": route["digest"],
        "observed_digest": None, "prompt_sha256": digest(prompt), "output_sha256": None,
        "config_sha256": digest(json.dumps(config, sort_keys=True, separators=(",", ":"))),
        "max_tokens": max_tokens, "generation": {"seed": 42, "temperature": 0, "num_ctx": 2048},
        "quality": "UNMEASURED", "signature": None,
        "limitations": ["Provider-reported identity; endpoint operator is trusted.",
                         "Prompt profiles are not separately trained OWEM weights.",
                         "Successful inference does not establish answer correctness."],
    }
    output = None
    try:
        tags = transport(endpoint + "/api/tags", timeout=min(timeout, 10))
        matches = [m for m in tags.get("models", []) if m.get("name") == route["model"]]
        if len(matches) != 1 or matches[0].get("digest") != route["digest"]:
            raise ValueError("MODEL_IDENTITY_MISMATCH")
        receipt["observed_digest"] = matches[0]["digest"]
        response = transport(endpoint + "/api/chat", {
            "model": route["model"], "stream": False, "think": False,
            "keep_alive": "0s",
            "messages": [{"role": "system", "content": SYSTEM + route["instruction"]},
                         {"role": "user", "content": prompt}],
            "options": {"num_predict": max_tokens, "seed": 42, "temperature": 0, "num_ctx": 2048},
        }, timeout=max(1, timeout - (time.monotonic() - started)))
        if response.get("model") != route["model"] or response.get("done") is not True:
            raise ValueError("INCOMPLETE_OR_WRONG_MODEL_RESPONSE")
        output = response.get("message", {}).get("content")
        if not isinstance(output, str) or not output.strip():
            output = None
            raise ValueError("EMPTY_RESPONSE")
        receipt.update(status="RUNTIME_OBSERVED", output_sha256=digest(output),
                       finish_reason=response.get("done_reason"),
                       generated_tokens=response.get("eval_count"))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # Do not persist response bodies, endpoints, credential-bearing errors or prompts.
        receipt["error_type"] = type(exc).__name__
        if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
            receipt["status"] = "UNREACHABLE"
        output = None
    receipt["duration_ms"] = round((time.monotonic() - started) * 1000)
    return {"text": output, "receipt": receipt}

def save_receipt(receipt, folder):
    """Write only metadata by default, exclusively; never silently overwrite evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (str(uuid.UUID(receipt["id"])) + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return path
