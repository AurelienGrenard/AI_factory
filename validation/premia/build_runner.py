"""Cross-compile the small Windows executable that calls libpremia.dll."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def project_root() -> Path:
    """Return the AI_factory repository owning this module."""

    return Path(__file__).resolve().parents[2]


def runner_path(root: Path | None = None) -> Path:
    """Return the untracked build location of the Premia runner."""

    repository = root or project_root()
    return repository / "build" / "validation" / "premia" / "premia_runner.exe"


def build_runner(root: Path | None = None, force: bool = False) -> Path:
    """Compile and return the Premia runner, rebuilding only when necessary."""

    repository = root or project_root()
    source = repository / "validation" / "premia" / "runner" / "premia_runner.c"
    package = repository / "validation" / "premia" / "premia-19-win64"
    output = runner_path(repository)
    dependencies = (
        source,
        package / "include" / "optype.h",
        package / "include" / "premia_obj.h",
        package / "lib" / "libpremia.lib",
    )
    if not force and output.is_file() and all(
        output.stat().st_mtime >= dependency.stat().st_mtime
        for dependency in dependencies
    ):
        return output

    compiler = shutil.which("x86_64-w64-mingw32-gcc-posix")
    if compiler is None:
        compiler = shutil.which("x86_64-w64-mingw32-gcc")
    if compiler is None:
        raise RuntimeError(
            "MinGW-w64 is required; install gcc-mingw-w64-x86-64-posix."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    command = (
        compiler,
        "-O2",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-I",
        str(package / "include"),
        str(source),
        str(package / "lib" / "libpremia.lib"),
        "-o",
        str(output),
    )
    subprocess.run(command, cwd=repository, check=True)
    return output


def main() -> int:
    """Build the runner from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(build_runner(force=arguments.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
