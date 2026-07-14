# Windows 11 and NVIDIA CUDA track

This is an additional, local-only track; the existing macOS/Swift/Metal course
remains unchanged.

## Install

Install **Visual Studio 2022 Build Tools** with Desktop development with C++ and
the current Windows SDK. Install **CMake 3.25+**, **Ninja**, and **CUDA Toolkit
12.x** (12.6 is tested). Run commands from an x64 Native Tools Command Prompt so
MSVC is in `PATH`. Confirm `cmake --version`, `ninja --version`, `cl`, `nvcc
--version`, and `nvidia-smi`.

The CUDA preset defaults to compute capability 7.5. Override it when configuring
with `-DCMAKE_CUDA_ARCHITECTURES=86` (or the value for your GPU).

## Build and learn

CPU-only:

```powershell
cmake --preset windows-cpu-debug
cmake --build --preset windows-cpu-debug
ctest --preset windows-cpu-debug
build\windows-cpu-debug\leetllm.exe list
build\windows-cpu-debug\leetllm.exe check 009 --cpu
```

NVIDIA CUDA:

```powershell
cmake --preset windows-cuda-debug
cmake --build --preset windows-cuda-debug
ctest --preset windows-cuda-debug
build\windows-cuda-debug\leetllm.exe check 009 --cuda
build\windows-cuda-debug\leetllm.exe benchmark 009 --cuda --iterations 20
scripts\validate-windows.ps1 -Cuda -Output build\reports\windows-nvidia-validation.json
```

Every `cpp/problems/pNNN` directory contains a CPU oracle, editable CUDA starter,
canonical CUDA solution, and CMake target declarations. CUDA calls use
`CUDA_CHECK`; canonical results are checked against CPU invariants.
Lesson 000 is orientation reading and therefore has no executable target. The
machine-readable Windows catalog is `cpp/lessons/windows-lessons.json`; validate
its paths and 000–047 coverage with `python scripts/check_windows_metadata.py`.

`check --cuda` launches the built canonical CUDA executable. It never reports a
CUDA pass from the CPU oracle alone. `benchmark` performs one warm-up and reports
end-to-end process time, including process startup, transfers, and synchronization;
use Nsight or CUDA events when kernel-only timing is required.

## Profiling

Build Release-like binaries by setting `CMAKE_BUILD_TYPE=Release`, then launch a
canonical lesson under **Nsight Systems** for timeline/transfer analysis or
**Nsight Compute** for kernel metrics:

```powershell
nsys profile build\windows-cuda-debug\p009_cuda_solution.exe
ncu build\windows-cuda-debug\p047_cuda_solution.exe
```

Warm up before measuring and keep transfers, synchronization, duration,
bandwidth, and throughput boundaries explicit.

## Troubleshooting

- `cl` not found: reopen an x64 Native Tools Command Prompt.
- Ninja chooses another compiler: remove only the affected `build\windows-*`
  directory and configure again from the MSVC prompt.
- `nvcc` not found: verify `CUDA_PATH` and Toolkit integration with VS 2022.
- `no kernel image`: set `CMAKE_CUDA_ARCHITECTURES` for the installed GPU.
- driver/runtime mismatch: update the NVIDIA driver and inspect `nvidia-smi`.
- CUDA checks unavailable: use the CPU preset; CPU oracles require no GPU.

All builds, checks, reports, and profiling run on the local machine. No source or
measurement data is transmitted.

CUDA build and runtime results must be collected on actual Windows NVIDIA
hardware. CPU-only CI validates configuration, metadata, and CPU oracles but is
not evidence of CUDA compilation, execution, performance, or numerical parity.
