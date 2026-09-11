"""Check source-driven CMake links and labels on incremental host-only builds."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def cmake_function(module, name):
    """Exercise the production function without configuring the CUDA catalogue."""
    text = (ROOT / "cmake" / module).read_text()
    match = re.search(
        rf"^[ \t]*function\({name}\b.*?^[ \t]*endfunction\(\)",
        text, re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"Missing production CMake function: {name}")
    return match.group()


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class InferredDependenciesTest(unittest.TestCase):
    def test_include_addition_and_removal_reconfigure_links_and_labels(self):
        functions = "\n".join([
            cmake_function("AIFactoryTargets.cmake", "ai_factory_collect_source_dependencies"),
            cmake_function("AIFactoryCatalog.cmake", "ai_factory_collect_generation_dependencies"),
            cmake_function("AIFactoryCatalog.cmake", "add_parameter_generator"),
            cmake_function("AIFactoryTests.cmake", "add_cuda_workbench_test"),
        ])
        with tempfile.TemporaryDirectory(prefix="ai-factory-cmake-") as temporary:
            source = Path(temporary)
            build = source / "build"

            def run(*command):
                result = subprocess.run(command, text=True, capture_output=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            (source / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(InferredDependencies LANGUAGES CXX)
enable_testing()
foreach(name ai_factory_cuda_tests common_cuda_tests equity_tests fixed_income_tests parameter_generators)
    add_custom_target(${name})
endforeach()
foreach(name runtime parameter_dataset dataset_validation)
    add_library(ai_factory_${name} INTERFACE)
    target_include_directories(ai_factory_${name} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
add_library(ai_factory_equity_probe_dataset STATIC EXCLUDE_FROM_ALL dependency.cpp)
add_library(ai_factory_probe_generation STATIC EXCLUDE_FROM_ALL dependency.cpp)
''' + functions + '''
add_cuda_workbench_test(probe probe.cpp fixture 5)
add_parameter_generator(generator generator.cpp)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/links.txt" CONTENT
    "$<TARGET_PROPERTY:test_probe,LINK_LIBRARIES>\n$<TARGET_PROPERTY:generator,LINK_LIBRARIES>\n")
''')
            (source / "dependency.cpp").write_text("int dependency() { return 0; }\n")
            headers = ["model/equity/markovian/probe/dataset.hpp",
                       "tools/datasets/probe_generation.hpp"]
            for header in headers:
                path = source / header
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("int dependency();\n")
            for file in ("probe.cpp", "generator.cpp"):
                (source / file).write_text("int main() { return 0; }\n")
            run("cmake", "-S", str(source), "-B", str(build))
            for dependency_present in (False, True, False):
                for file, header in zip(("probe.cpp", "generator.cpp"), headers):
                    content = (f'#include "{header}"\nint main() {{ return dependency(); }}\n'
                               if dependency_present else "int main() { return 0; }\n")
                    (source / file).write_text(content)
                # No manual configure: both inference owners must be tracked.
                run("cmake", "--build", str(build), "--target", "test_probe", "generator")
                links = (build / "links.txt").read_text().splitlines()
                self.assertEqual("ai_factory_equity_probe_dataset" in links[0], dependency_present)
                self.assertEqual("ai_factory_probe_generation" in links[1], dependency_present)
                inventory = json.loads(run("ctest", "--test-dir", str(build), "--show-only=json-v1"))
                labels = next(p["value"] for p in inventory["tests"][0]["properties"]
                              if p["name"] == "LABELS")
                self.assertIn("equity" if dependency_present else "common", labels)
                self.assertNotIn("common" if dependency_present else "equity", labels)


if __name__ == "__main__":
    unittest.main()
