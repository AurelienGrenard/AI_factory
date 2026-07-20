import ctypes
import os
import subprocess
from pathlib import Path

import pytest


class HestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
        ("scheme", ctypes.c_int),
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
    row_ptr = ctypes.POINTER(HestonRow)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    library.ai_factory_cpu_generate_heston_terminal_spots.argtypes = [
        ctypes.POINTER(HestonRow),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.ai_factory_cpu_generate_heston_terminal_spots.restype = ctypes.c_int
    library.ai_factory_generate_heston_terminal_spots.argtypes = [
        ctypes.POINTER(HestonRow),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.ai_factory_generate_heston_terminal_spots.restype = ctypes.c_int
    library.ai_factory_cpu_generate_heston_state_paths.argtypes = [
        row_ptr, ctypes.c_size_t, ctypes.c_size_t, double_ptr, double_ptr,
    ]
    library.ai_factory_cpu_generate_heston_state_paths.restype = ctypes.c_int
    library.ai_factory_generate_heston_state_paths.argtypes = [
        row_ptr, ctypes.c_size_t, ctypes.c_size_t,
        double_ptr, double_ptr, double_ptr,
    ]
    library.ai_factory_generate_heston_state_paths.restype = ctypes.c_int
    return library


def _last_error(library: ctypes.CDLL) -> str:
    raw = library.ai_factory_cuda_last_error()
    return raw.decode("utf-8") if raw else "unknown C++/CUDA error"


@pytest.mark.skipif(
    not Path("/dev/dxg").exists() and not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="CUDA device is not visible in this environment.",
)
def test_heston_cpu_gpu_terminal_spots_match_pathwise(tmp_path: Path) -> None:
    library = _load_library(_build_cuda_library(tmp_path))
    num_paths = 512
    num_steps = 64
    for scheme in [0, 1, 2]:
        row = HestonRow(
            1.0,
            0.012,
            0.004,
            0.01,
            1.4,
            0.0121,
            0.12,
            -0.2,
            0.9376148755637311,
            1.0,
            900000001,
            scheme,
        )
        cpu = (ctypes.c_double * num_paths)()
        gpu = (ctypes.c_double * num_paths)()
        kernel_seconds = ctypes.c_double(0.0)

        cpu_status = library.ai_factory_cpu_generate_heston_terminal_spots(
            ctypes.byref(row),
            num_paths,
            num_steps,
            cpu,
        )
        assert cpu_status == 0, _last_error(library)

        gpu_status = library.ai_factory_generate_heston_terminal_spots(
            ctypes.byref(row),
            num_paths,
            num_steps,
            gpu,
            ctypes.byref(kernel_seconds),
        )
        assert gpu_status == 0, _last_error(library)

        max_abs_diff = max(abs(cpu[index] - gpu[index]) for index in range(num_paths))
        assert max_abs_diff <= 2.0e-14, (scheme, max_abs_diff)

    row.scheme = 2
    value_count = num_paths * (num_steps + 1)
    cpu_spots = (ctypes.c_double * value_count)()
    cpu_variances = (ctypes.c_double * value_count)()
    gpu_spots = (ctypes.c_double * value_count)()
    gpu_variances = (ctypes.c_double * value_count)()
    kernel_seconds = ctypes.c_double(0.0)
    assert library.ai_factory_cpu_generate_heston_state_paths(
        ctypes.byref(row), num_paths, num_steps, cpu_spots, cpu_variances
    ) == 0, _last_error(library)
    assert library.ai_factory_generate_heston_state_paths(
        ctypes.byref(row), num_paths, num_steps,
        gpu_spots, gpu_variances, ctypes.byref(kernel_seconds),
    ) == 0, _last_error(library)
    assert kernel_seconds.value > 0.0
    assert max(
        abs(cpu_spots[index] - gpu_spots[index]) for index in range(value_count)
    ) <= 2.0e-14
    assert max(
        abs(cpu_variances[index] - gpu_variances[index])
        for index in range(value_count)
    ) <= 2.0e-14
