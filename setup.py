import os
import sys
import shutil
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

FORCE_CPU = False
if "--cpu-only" in sys.argv:
    FORCE_CPU = True
    sys.argv.remove("--cpu-only")


def has_nvidia_gpu():
    if FORCE_CPU:
        print("[-] --cpu-only flag detected. Forcing CPU-only compilation.")
        return False
    nvcc_available = shutil.which("nvcc") is not None
    nvidia_smi_available = False
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        nvidia_smi_available = True
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        pass
    return nvcc_available and nvidia_smi_available


def get_cuda_lib_dir():
    if sys.platform.startswith("win"):
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            return os.path.join(cuda_path, "lib", "x64")
    else:
        for path in ["/usr/local/cuda/lib64", "/usr/lib/cuda/lib64"]:
            if os.path.exists(path):
                return path
    return None


class CustomBuildExt(build_ext):
    def build_extensions(self):
        try:
            import pybind11
        except ImportError:
            raise RuntimeError("pybind11 is required to build this extension.")

        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            is_windows = self.compiler.compiler_type == "msvc"

            # Using C++20 for std::numbers support
            if is_windows:
                ext.extra_compile_args.append("/std:c++20")
                ext.extra_compile_args.append("/O2")
            else:
                ext.extra_compile_args.append("-std=c++20")
                ext.extra_compile_args.append("-O3")

            cu_sources = [s for s in ext.sources if s.endswith(".cu")]
            cpp_sources = [s for s in ext.sources if not s.endswith(".cu")]

            if cu_sources:
                nvcc = shutil.which("nvcc")
                python_includes = [
                    sysconfig.get_path("include"),
                    sysconfig.get_path("platinclude"),
                ]

                for cu_file in cu_sources:
                    obj_ext = ".obj" if is_windows else ".o"
                    obj_file = os.path.join(
                        self.build_temp, os.path.basename(cu_file) + obj_ext
                    )
                    os.makedirs(self.build_temp, exist_ok=True)

                    nvcc_cmd = [
                        nvcc,
                        "-c",
                        cu_file,
                        "-o",
                        obj_file,
                        "-O3",
                        "-std=c++20",
                    ]
                    for inc in ext.include_dirs:
                        nvcc_cmd.append(f"-I{inc}")
                    for inc in python_includes:
                        if inc and os.path.exists(inc):
                            nvcc_cmd.append(f"-I{inc}")
                    for macro, value in ext.define_macros:
                        nvcc_cmd.append(
                            f"-D{macro}" if value is None else f"-D{macro}={value}"
                        )

                    if not is_windows:
                        nvcc_cmd.extend(["-Xcompiler", "-fPIC"])
                    subprocess.check_call(nvcc_cmd)
                    ext.extra_objects.append(obj_file)

                ext.sources = cpp_sources
                ext.libraries.append("cudart")
                cuda_lib_dir = get_cuda_lib_dir()
                if cuda_lib_dir:
                    ext.library_dirs.append(cuda_lib_dir)

        super().build_extensions()

        bin_dir = "bin"
        os.makedirs(bin_dir, exist_ok=True)
        
        if os.path.exists(self.build_lib):
            for root, dirs, files in os.walk(self.build_lib):
                for file in files:
                    if file.endswith(".so") or file.endswith(".pyd"):
                        shutil.copy2(
                            os.path.join(root, file), os.path.join(bin_dir, file)
                        )
                        print(f"[+] Copied {file} to {bin_dir}/")


source_files = ["src/core/binding.cpp", "src/core/cpp/LangevinGillespie.cpp"]
macros = []
if has_nvidia_gpu():
    source_files.append("src/core/cuda/LangevinGillespie.cu")
    macros.append(("HAS_CUDA", "1"))
else:
    macros.append(("CPU_ONLY", "1"))

setup(
    name="f1sim",
    version="1.0.0",
    ext_modules=[
        Extension(
            "f1sim",
            sources=source_files,
            include_dirs=["src/core/include"],
            define_macros=macros,
            language="c++",
        )
    ],
    cmdclass={"build_ext": CustomBuildExt},
)
