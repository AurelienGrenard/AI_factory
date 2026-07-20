import ctypes
import os
import subprocess
from pathlib import Path

import pytest


class RoughHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double),
        ("hurst", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
    ]


def _build_cuda_library(tmp_path: Path) -> Path:
    build_dir = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", "src_cpp", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "ai_factory_cuda_shared"],
        check=True,
        text=True,
        capture_output=True,
    )
    return build_dir / "libai_factory_cuda.so"


def _load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    row_ptr = ctypes.POINTER(RoughHestonRow)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    library.ai_factory_cpu_generate_rough_heston_spot_paths.argtypes = [
        row_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_ptr,
    ]
    library.ai_factory_cpu_generate_rough_heston_spot_paths.restype = ctypes.c_int
    library.ai_factory_generate_rough_heston_spot_paths.argtypes = [
        row_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_ptr,
        double_ptr,
    ]
    library.ai_factory_generate_rough_heston_spot_paths.restype = ctypes.c_int
    library.ai_factory_cpu_generate_rough_heston_state_paths.argtypes = [
        row_ptr, ctypes.c_size_t, ctypes.c_size_t, double_ptr, double_ptr,
    ]
    library.ai_factory_cpu_generate_rough_heston_state_paths.restype = ctypes.c_int
    library.ai_factory_generate_rough_heston_state_paths.argtypes = [
        row_ptr, ctypes.c_size_t, ctypes.c_size_t,
        double_ptr, double_ptr, double_ptr,
    ]
    library.ai_factory_generate_rough_heston_state_paths.restype = ctypes.c_int
    return library


def _last_error(library: ctypes.CDLL) -> str:
    raw = library.ai_factory_cuda_last_error()
    return raw.decode("utf-8") if raw else "unknown C++/CUDA error"


@pytest.mark.skipif(
    not Path("/dev/dxg").exists() and not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="CUDA device is not visible in this environment.",
)
def test_rough_heston_cpu_gpu_spot_paths_match_pathwise(tmp_path: Path) -> None:
    library = _load_library(_build_cuda_library(tmp_path))
    num_paths = 256
    num_steps = 64
    row = RoughHestonRow(
        1.0,
        0.025,
        0.01,
        0.04,
        1.7,
        0.05,
        0.45,
        0.12,
        -0.7,
        1.0,
        1.25,
        900000001,
    )
    value_count = num_paths * (num_steps + 1)
    cpu = (ctypes.c_double * value_count)()
    gpu = (ctypes.c_double * value_count)()
    kernel_seconds = ctypes.c_double(0.0)

    cpu_status = library.ai_factory_cpu_generate_rough_heston_spot_paths(
        ctypes.byref(row), num_paths, num_steps, cpu
    )
    assert cpu_status == 0, _last_error(library)

    gpu_status = library.ai_factory_generate_rough_heston_spot_paths(
        ctypes.byref(row), num_paths, num_steps, gpu, ctypes.byref(kernel_seconds)
    )
    assert gpu_status == 0, _last_error(library)
    assert kernel_seconds.value > 0.0

    max_abs_diff = max(abs(cpu[index] - gpu[index]) for index in range(value_count))
    assert max_abs_diff <= 2.0e-13, max_abs_diff

    factor_count = value_count * 8
    cpu_spots = (ctypes.c_double * value_count)()
    cpu_factors = (ctypes.c_double * factor_count)()
    gpu_spots = (ctypes.c_double * value_count)()
    gpu_factors = (ctypes.c_double * factor_count)()
    kernel_seconds = ctypes.c_double(0.0)
    assert library.ai_factory_cpu_generate_rough_heston_state_paths(
        ctypes.byref(row), num_paths, num_steps, cpu_spots, cpu_factors
    ) == 0, _last_error(library)
    assert library.ai_factory_generate_rough_heston_state_paths(
        ctypes.byref(row), num_paths, num_steps,
        gpu_spots, gpu_factors, ctypes.byref(kernel_seconds),
    ) == 0, _last_error(library)
    assert kernel_seconds.value > 0.0
    assert max(
        abs(cpu_spots[index] - gpu_spots[index]) for index in range(value_count)
    ) <= 2.0e-13
    assert max(
        abs(cpu_factors[index] - gpu_factors[index]) for index in range(factor_count)
    ) <= 2.0e-13
