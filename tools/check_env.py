#!/usr/bin/env python3
"""Report the local environment required by SparseAttn without changing it."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_version(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "not found"


def module_state(name: str) -> str:
    return "installed" if importlib.util.find_spec(name) else "missing"


def main() -> int:
    print(f"repository: {ROOT}")
    print(f"python:     {sys.executable} ({sys.version.split()[0]})")
    print(f"cmake:      {command_version(['cmake', '--version'])}")
    print(f"nvcc:       {command_version(['nvcc', '--version'])}")
    print(f"nvidia-smi: {'found' if shutil.which('nvidia-smi') else 'not found'}")

    try:
        import torch
    except ImportError:
        print("torch:      missing (install a CUDA-enabled PyTorch before building)")
        return 1

    print(f"torch:      {torch.__version__}; CUDA build {torch.version.cuda}")
    print(f"cuda ready: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        print(f"gpu:        {torch.cuda.get_device_name(device)} (sm_{capability[0]}{capability[1]})")
        if capability[0] < 8:
            print("warning:    SPTC currently accepts NVIDIA compute capability 8.x or 9.0.")
    else:
        print("warning:    SPTC cannot run until CUDA is available to PyTorch.")

    for module in ("sgl_kernel", "minference", "matplotlib", "triton"):
        print(f"{module:11}{module_state(module)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
