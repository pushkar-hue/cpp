from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "calculator_api",
        ["ptr.cpp"],
    )
]

setup(
    name="calculator_api",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)