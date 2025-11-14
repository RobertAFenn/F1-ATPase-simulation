# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sysconfig
import pybind11
import os
import shutil

# C++ and NVCC compilation flags
CXX_FLAGS = ["-O3", "-std=c++20", "-fPIC", "-Wall"]
NVCC_FLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC", "-std=c++20"]


def find_cuda_libdir():
    for d in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib"]:
        if os.path.isdir(d):
            return d
    return "/usr/local/cuda/lib64"


class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()


class BuildExtension(build_ext):
    def build_extensions(self):
        py_inc = sysconfig.get_paths()["include"]
        for ext in self.extensions:
            ext.include_dirs += [py_inc, str(get_pybind_include())]

            # Compile CUDA sources first
            self._compile_cuda_sources(ext)

            # Add C++ flags
            ext.extra_compile_args = list(CXX_FLAGS)

            # CUDA linkage
            cuda_libdir = find_cuda_libdir()
            ext.library_dirs = ext.library_dirs or []
            if cuda_libdir not in ext.library_dirs:
                ext.library_dirs.append(cuda_libdir)
            ext.libraries = ext.libraries or []
            if "cudart" not in ext.libraries:
                ext.libraries.append("cudart")
            ext.extra_link_args = ext.extra_link_args or []
            ext.extra_link_args += [f"-L{cuda_libdir}", "-lcudart"]

        super().build_extensions()

        # Optional: move final .so into a bin/ folder for clarity
        self._move_so_to_bin()

    def _compile_cuda_sources(self, ext):
        sources = list(ext.sources)
        new_sources = []
        extra_objects = list(getattr(ext, "extra_objects", []))
        for src in sources:
            if src.endswith(".cu"):
                obj_file = os.path.splitext(src)[0] + ".o"
                cmd = ["nvcc", "-c", src, "-o", obj_file] + NVCC_FLAGS
                for inc in ext.include_dirs:
                    cmd += ["-I", inc]
                print("Compiling CUDA:", " ".join(cmd))
                subprocess.check_call(cmd)
                extra_objects.append(obj_file)
            else:
                new_sources.append(src)
        ext.sources = new_sources
        ext.extra_objects = extra_objects

    def _move_so_to_bin(self):
        build_lib = self.build_lib
        for ext in self.extensions:
            so_name = self.get_ext_filename(ext.name)
            src_path = os.path.join(build_lib, so_name)
            bin_dir = os.path.join(os.getcwd(), "bin")
            os.makedirs(bin_dir, exist_ok=True)
            dst_path = os.path.join(bin_dir, os.path.basename(so_name))
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Moved {src_path} -> {dst_path}")


ext_modules = [
    Extension(
        "f1sim",
        sources=[
            "src/core/binding.cpp",
            "src/core/cpp/LangevinGillespie.cpp",
            "src/core/cuda/LangevinGillespie.cu",
        ],
        include_dirs=[
            str(get_pybind_include()),
            "src/core/include",
        ],
        language="c++",
        extra_objects=[],  # .o files from will be added automatically
    )
]

setup(
    name="f1sim",
    version="0.1",
    author="Robert",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
