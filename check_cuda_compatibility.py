#!/usr/bin/env python3
"""check_cuda_compatibility.py

Utility script that inspects the system's NVIDIA GPU, driver version, CUDA runtime, and PyTorch build
to determine whether they are compatible. It prints a concise report and, if mismatches are detected,
provides actionable suggestions (e.g., reinstall a suitable PyTorch wheel).

Usage:
python check_cuda_compatibility.py
"""

import subprocess
import sys
import re

def run_cmd(cmd: list[str]) -> str:
"""Run a command and return its stdout as a stripped string.
Errors are captured and returned as an empty string.
"""
try:
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
return result.stdout.strip()
except Exception as e:
return ""

def get_nvidia_smi_info() -> dict:
"""Parse `nvidia-smi` output for driver version and GPU compute capability.
Returns a dict with keys: driver_version, gpu_name, compute_capability.
"""
out = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,compute_capability", "--format=csv,noheader"])
if not out:
return {}
# Example line: "GeForce GTX 1050 Ti, 525.105.17, 6.1"
parts = [p.strip() for p in out.split(",")]
if len(parts) != 3:
return {}
gpu_name, driver_version, compute_capability = parts
return {
"gpu_name": gpu_name,
"driver_version": driver_version,
"compute_capability": compute_capability,
}

def get_torch_info() -> dict:
"""Collect PyTorch version, CUDA version, and supported compute capabilities.
"""
try:
import torch
except ImportError:
return {"torch_installed": False}
info = {
"torch_installed": True,
"torch_version": torch.__version__,
"cuda_available": torch.cuda.is_available(),
}
if info["cuda_available"]:
info["cuda_version"] = torch.version.cuda
# Torch reports supported architectures via TORCH_CUDA_ARCH_LIST env var or defaults.
# We can inspect the compiled arch list via torch.cuda.get_arch_list() (available in recent releases).
try:
archs = torch.cuda.get_arch_list()
info["torch_arches"] = archs
except Exception:
info["torch_arches"] = []
info["device_name"] = torch.cuda.get_device_name(0)
info["device_capability"] = "{}.{}".format(*torch.cuda.get_device_capability(0))
return info

def main():
gpu_info = get_nvidia_smi_info()
torch_info = get_torch_info()

print("=== System Compatibility Report ===\n")
if not gpu_info:
print("[WARNING] Could not retrieve GPU info via nvidia-smi. Ensure the NVIDIA driver is installed and nvidia-smi is on PATH.")
else:
print(f"GPU: {gpu_info.get('gpu_name')}")
print(f"Driver version: {gpu_info.get('driver_version')}")
print(f"Compute Capability: {gpu_info.get('compute_capability')}\n")

if not torch_info.get("torch_installed"):
print("[ERROR] PyTorch is not installed in the current environment.")
sys.exit(1)
else:
print(f"PyTorch version: {torch_info.get('torch_version')}")
if torch_info.get("cuda_available"):
print(f"CUDA runtime (torch): {torch_info.get('cuda_version')}")
print(f"Torch sees device: {torch_info.get('device_name')} (CC {torch_info.get('device_capability')})")
arches = torch_info.get("torch_arches", [])
if arches:
print(f"Torch compiled for architectures: {', '.join(arches)}")
else:
print("[WARNING] Could not determine compiled CUDA architectures in the torch wheel.")
else:
print("[WARNING] CUDA not available to PyTorch. This may be due to a missing or incompatible CUDA toolkit.")

# Compatibility check
compatible = True
if gpu_info and torch_info.get("cuda_available"):
gpu_cc = gpu_info.get("compute_capability")
torch_cc = torch_info.get("device_capability")
if gpu_cc != torch_cc:
compatible = False
print("\n[ERROR] Compute capability mismatch between driver (" + gpu_cc + ") and torch (" + torch_cc + ").")
print(" Suggested fix: reinstall a torch wheel that includes sm_" + gpu_cc.replace('.', '') + ".")
if not torch_info.get("cuda_available"):
compatible = False
print("\n[ERROR] PyTorch cannot access CUDA. Ensure the driver matches the CUDA version used to build the torch wheel.")

if compatible:
print("\n[OK] All checks passed. Your GPU, driver, CUDA runtime, and PyTorch appear compatible.")
else:
print("\n[WARNING] Compatibility issues detected. Review the messages above for remediation steps.")

if __name__ == "__main__":
main()
