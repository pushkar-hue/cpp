from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# Define optimization flags based on the OS
if sys.platform == "win32":
    # Windows (MSVC) flags
    # /O2 = Maximize speed
    # /openmp = Enable multi-threading
    compile_args = ['/O2', '/openmp', '/fp:fast']
    link_args = []
else:
    # Linux/macOS (GCC/Clang) flags
    # -O3 = Aggressive optimization
    # -march=native = Use AVX/AVX2 instructions specific to your CPU
    # -ffast-math = Skip strict IEEE checks for speed
    # -fopenmp = Enable multi-threading
    compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
    link_args = ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "calculator_api",
        ["ptr.cpp"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="calculator_api",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)