"""Bounded NVIDIA runtime observations; optional libraries are never installed.

Run inside the target Kaggle image with --exercise for a tiny CUDA and RAPIDS
operation. A package listing is not runtime integration or model qualification.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import subprocess

PACKAGES = ("torch", "cudf-cu12", "cudf-cu13", "cuml-cu12", "cuml-cu13",
            "nemo-curator", "nemo-automodel", "nemoguardrails", "nemo-evaluator",
            "nvidia-modelopt", "tensorrt-llm", "transformers", "peft")


def probe(exercise=False):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"schema": "meok.nvidia-runtime/1", "observed_at": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "packages": versions,
              "hardware": {"status": "UNCHECKABLE"}, "torch_cuda": {"status": "NOT_RUN"},
              "rapids_cudf": {"status": "NOT_RUN"}, "model_inference": "NOT_RUN",
              "limitations": ["Package versions are inventory, not proof of compatibility.",
                              "Tiny arithmetic probes do not qualify a model, trainer or serving engine.",
                              "No install, model download, service start or publication is performed."]}
    try:
        info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        if info.returncode == 0 and info.stdout.strip():
            result["hardware"] = {"status": "RUNTIME_OBSERVED", "nvidia_smi_csv": info.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired):
        pass
    if exercise:
        # Subprocess deadlines prevent GPU imports/driver initialization from
        # hanging the notebook or a calling automation indefinitely.
        programs = {
            "torch_cuda": "import json,torch; assert torch.cuda.is_available(); x=torch.tensor([1,2,3],device='cuda'); v=x.sum().item(); torch.cuda.synchronize(); assert v==6; print(json.dumps({'sum':v,'device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'capability':list(torch.cuda.get_device_capability(0)),'bf16_supported':torch.cuda.is_bf16_supported()}))",
            "rapids_cudf": "import json,cudf; x=cudf.DataFrame({'a':[1,2,3]}); v=int(x.a.sum()); assert v==6; print(json.dumps({'sum':v,'cudf':cudf.__version__}))",
        }
        import sys
        for name, code in programs.items():
            try:
                done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=45)
                if done.returncode == 0:
                    result[name] = {"status": "RUNTIME_OBSERVED", "observation": json.loads(done.stdout.strip())}
                else:
                    result[name] = {"status": "UNCHECKABLE", "exit_code": done.returncode}
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                result[name] = {"status": "UNCHECKABLE", "error_type": type(error).__name__}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", action="store_true")
    args = parser.parse_args()
    print(json.dumps(probe(args.exercise), indent=2))


if __name__ == "__main__": main()
