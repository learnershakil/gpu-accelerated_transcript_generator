"""Hardware and GPU detection utilities for transcript generation.
Optimized for NVIDIA GeForce RTX 3050 and CUDA acceleration.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def setup_cuda_environment() -> None:
    """Ensure CUDA 12 and cuDNN library paths are available in LD_LIBRARY_PATH
    and pre-loaded if installed via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12).
    """
    venv_site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    nvidia_dir = venv_site / "nvidia"

    cuda_lib_paths = []
    if nvidia_dir.is_dir():
        for sub in ["cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"]:
            lib_path = nvidia_dir / sub / "lib"
            if lib_path.is_dir():
                cuda_lib_paths.append(str(lib_path))

    # Also check system CUDA
    for sys_cuda in ["/opt/cuda/lib64", "/usr/local/cuda/lib64", "/usr/lib"]:
        if os.path.isdir(sys_cuda):
            cuda_lib_paths.append(sys_cuda)

    # Preload essential shared libraries into process space if present
    for lib_dir in cuda_lib_paths:
        p = Path(lib_dir)
        for lib_name in ["libcublas.so.12", "libcudnn.so.9", "libcudnn.so.8"]:
            candidate = p / lib_name
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

    # Update LD_LIBRARY_PATH
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [p for p in cuda_lib_paths if p not in current_ld.split(":")]
    if new_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + (f":{current_ld}" if current_ld else "")


# Initialize environment on module import
setup_cuda_environment()


def get_gpu_info() -> Dict[str, Any]:
    """Retrieve detailed information about available NVIDIA GPU(s) via CUDA runtime."""
    setup_cuda_environment()
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_name": "None",
        "total_memory_gb": 0.0,
        "compute_capability": None,
        "compute_types_supported": [],
        "driver_warning": None,
    }

    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            info["cuda_available"] = True
            supported_types = list(ctranslate2.get_supported_compute_types("cuda", 0))
            info["compute_types_supported"] = supported_types

            # Try to get detailed device name via CUDA driver API
            try:
                cuda = ctypes.CDLL("libcuda.so.1")
                cuda.cuInit(0)
                device = ctypes.c_int()
                cuda.cuDeviceGet(ctypes.byref(device), 0)

                name_buf = ctypes.create_string_buffer(256)
                cuda.cuDeviceGetName(name_buf, 256, device)
                info["device_name"] = name_buf.value.decode("utf-8")

                mem = ctypes.c_size_t()
                cuda.cuDeviceTotalMem(ctypes.byref(mem), device)
                info["total_memory_gb"] = round(mem.value / (1024**3), 2)
            except Exception:
                info["device_name"] = "NVIDIA CUDA Device (RTX Series)"
                info["total_memory_gb"] = 4.0
    except Exception as e:
        err_msg = str(e)
        if "Driver/library version mismatch" in err_msg:
            info["driver_warning"] = (
                "NVIDIA kernel module and library version mismatch. "
                "A system reboot is recommended to refresh driver modules."
            )
        else:
            info["driver_warning"] = err_msg

    return info


def get_recommended_settings() -> Tuple[str, str, str]:
    """Return recommended (device, compute_type, model) for current machine."""
    gpu = get_gpu_info()
    if gpu["cuda_available"]:
        device = "cuda"
        # RTX 3050 supports float16 and int8_float16
        # If memory <= 4GB, float16 works with large-v3 or large-v3-turbo
        compute_type = "float16" if "float16" in gpu["compute_types_supported"] else "int8_float16"
        model = "large-v3"
        return device, compute_type, model
    else:
        return "cpu", "int8", "medium"
