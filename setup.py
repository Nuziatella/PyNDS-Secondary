"""Setup script for PyNDS package.

The construction crew for your digital time machine! This module handles the
building and installation of PyNDS, including the compilation of C++ extensions
using CMake. It provides custom build classes for handling the complex C++ build
process required for the NooDS emulator integration.

Classes:
    CMakeExtension: Extension class for CMake-based builds
    CMakeBuild: Custom build class for CMake compilation
"""

import os
import platform
import subprocess
import sys
from pprint import pprint

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

c_module_name = "cnds"

cmake_cmd_args = []
for f in sys.argv:
    if f.startswith("-D"):
        cmake_cmd_args.append(f)

for f in cmake_cmd_args:
    sys.argv.remove(f)

print(cmake_cmd_args)


def _get_env_variable(name, default="OFF"):
    if name not in os.environ.keys():
        return default
    return os.environ[name]


class CMakeExtension(Extension):
    """CMake-based extension for building C++ components.

    The foundation layer of your digital time machine! This class extends the
    standard Extension class to handle CMake-based builds, which are required
    for the NooDS emulator integration.
    """

    def __init__(self, name, cmake_lists_dir=".", sources=[], **kwa):
        """Initialize CMake extension.

        Parameters
        ----------
        name : str
            Name of the extension module
        cmake_lists_dir : str, optional
            Directory containing CMakeLists.txt, by default "."
        sources : list, optional
            Source files for the extension, by default []
        **kwa
            Additional keyword arguments passed to Extension
        """
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    """Custom build class for CMake-based extensions.

    The master builder of your digital time machine! This class handles the
    compilation of C++ extensions using CMake, including proper configuration
    and build process management.
    """

    def build_extensions(self):
        """Build all extensions using CMake.

        This method configures and builds all CMake-based extensions,
        handling platform-specific requirements and build configurations.

        Raises
        ------
        RuntimeError
            If CMake is not found or build process fails
        """
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = "Debug" if _get_env_variable("PYNDS_DEBUG") == "ON" else "Release"

            cmake_args = [
                "-DCMAKE_BUILD_TYPE=%s" % cfg,
                # Ask CMake to place the resulting library in the directory
                # containing the extension
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                # Other intermediate static libraries are placed in a
                # temporary build directory instead
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), self.build_temp
                ),
            ]

            if platform.system() == "Windows":
                plat = "x64" if platform.architecture()[0] == "64bit" else "Win32"
                cmake_args += [
                    "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE",
                    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(
                        cfg.upper(), extdir
                    ),
                ]
                if self.compiler.compiler_type == "msvc":
                    cmake_args += [
                        "-DCMAKE_GENERATOR_PLATFORM=%s" % plat,
                    ]
                else:
                    cmake_args += [
                        "-G",
                        "MinGW Makefiles",
                    ]

            cmake_args += cmake_cmd_args

            pprint(cmake_args)

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config and build the extension
            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            # Build with multiple jobs
            num_jobs = os.cpu_count() or 1
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", cfg, "--parallel", str(num_jobs)],
                cwd=self.build_temp,
            )


version = "0.0.4-alpha"

setup(
    name="pynds",
    packages=find_packages(),
    install_requires=["numpy", "pygame"],
    version=version,
    description="Python bindings for NooDS",
    author="unexploredtest",
    author_email="unexploredtest@tutanota.com",
    url="https://github.com/unexploredtest/PyNDS",
    keywords=["noods", "emulator", "nintendo ds", "gameboy advance", "python", "c++"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension(c_module_name)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
)
