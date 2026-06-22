from pathlib import Path
import os

import torch


def conda_compiler_paths() -> tuple[Path | None, Path | None]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None, None

    bin_dir = Path(conda_prefix) / "bin"
    gcc = bin_dir / "x86_64-conda-linux-gnu-gcc"
    gxx = bin_dir / "x86_64-conda-linux-gnu-g++"
    if gcc.exists() and gxx.exists():
        return gcc, gxx
    return None, None


def set_default_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST") or not torch.cuda.is_available():
        return

    capabilities = {
        f"{major}.{minor}"
        for major, minor in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
    }
    if capabilities:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(capabilities))
