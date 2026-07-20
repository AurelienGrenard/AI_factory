"""Load rebuilt native libraries without reusing a stale dynamic-loader handle."""

from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_cpp_library() -> ctypes.CDLL:
    configured = os.environ.get("AI_FACTORY_CPP_LIBRARY")
    candidates = [
        Path(configured) if configured else None,
        Path("/tmp/ai_factory_cpp_build/libai_factory_cuda.so"),
        PROJECT_ROOT / "src_cpp" / "build" / "libai_factory_cuda.so",
        PROJECT_ROOT / "build" / "libai_factory_cuda.so",
        PROJECT_ROOT / "build" / "src_cpp" / "libai_factory_cuda.so",
        PROJECT_ROOT / "cmake-build-release" / "libai_factory_cuda.so",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            library = ctypes.CDLL(_versioned_library_copy(candidate))
            library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
            return library
    raise FileNotFoundError("Could not find libai_factory_cuda.so. Build src_cpp first.")


def _versioned_library_copy(library_path: Path) -> str:
    resolved = library_path.resolve()
    stat = resolved.stat()
    load_dir = Path("/tmp") / "ai_factory_loaded_libraries"
    load_dir.mkdir(parents=True, exist_ok=True)
    copy_path = (
        load_dir
        / f"{resolved.stem}_{stat.st_mtime_ns}_{stat.st_size}{resolved.suffix}"
    )
    if not copy_path.exists() or copy_path.stat().st_size != stat.st_size:
        temporary_path = (
            load_dir
            / f"{resolved.stem}_{stat.st_mtime_ns}_{stat.st_size}_{os.getpid()}.tmp"
        )
        shutil.copy2(resolved, temporary_path)
        os.replace(temporary_path, copy_path)
    return str(copy_path)
