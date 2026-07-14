Feature: Windows and NVIDIA CUDA LeetLLM curriculum

  Scenario: Configure and build the CPU track on Windows
    Given a Windows 11 checkout with Visual Studio 2022 Build Tools, CMake, and Ninja installed
    When I run "cmake --preset windows-cpu-debug"
    And I run "cmake --build --preset windows-cpu-debug"
    Then the build exits with code 0
    And the Windows LeetLLM CLI executable exists in the configured build directory

  Scenario: Build all CUDA curriculum targets
    Given Windows 11 has CUDA Toolkit 12.6 and an NVIDIA GPU with compute capability 7.5
    When I run "cmake --preset windows-cuda-debug"
    And I run "cmake --build --preset windows-cuda-debug"
    Then the build exits with code 0
    And starter and canonical CUDA targets exist for every problem from 001 through 047

  Scenario: Discover Windows lessons through the CLI
    Given the Windows LeetLLM CLI is built
    When I run "leetllm.exe list"
    Then the output contains 48 lessons numbered 000 through 047
    When I run "leetllm.exe learn 004"
    Then the output includes Windows CPU instructions
    And the output includes an NVIDIA CUDA stage
    And the output includes the CUDA source file to edit

  Scenario: Check CPU and CUDA numerical behavior
    Given problem 009 canonical CPU and CUDA solutions are built
    When I run "leetllm.exe check 009 --cpu"
    Then the command exits with code 0
    And the output reports all stable-softmax cases passed
    When I run "leetllm.exe check 009 --cuda"
    Then the command exits with code 0
    And the output reports comparison against the CPU oracle
    And the output reports the applied absolute and relative tolerances

  Scenario: Generate an NVIDIA benchmark report
    Given the canonical CUDA targets are built on an NVIDIA GPU
    When I run "leetllm.exe report roofline --output build/reports/roofline.json"
    Then the command exits with code 0
    And "build/reports/roofline.json" contains the GPU model
    And it contains the CUDA driver and runtime versions
    And it contains warm-up count, measured duration, bandwidth, and throughput

  Scenario: Run the capstone CPU engine and CUDA verification slice
    Given the canonical problem 047 targets are built
    When I run "leetllm.exe capstone --backend cpu --prompt hello --max-tokens 4 --seed 42"
    Then the command exits with code 0
    And exactly 4 generated token identifiers are reported
    When I run "leetllm.exe check 047 --cuda"
    Then the command exits with code 0
    And the fused QKV plus RoPE CUDA output matches the CPU oracle within the documented tolerance

  Scenario: Validate local-only Windows documentation
    Given I open the Windows setup documentation
    Then it documents Visual Studio 2022, CMake, Ninja, and CUDA Toolkit 12.x installation
    And it documents CPU-only checks, CUDA checks, Nsight profiling, and troubleshooting
    And the documented workflow contains no spec2cloud, cloud execution, telemetry, or source upload step

  Scenario: Preserve the existing macOS track
    Given the Windows and CUDA files have been added
    When I inspect "Package.swift" and the existing problem directories
    Then the Swift CPU and Metal lesson targets remain available
    And the documented macOS commands still include "swift run leetllm list"
    And Windows guidance is presented as an additional platform track

  Scenario: Produce a complete NVIDIA validation summary
    Given all CUDA targets are built on the Windows NVIDIA machine
    When I run "scripts\\validate-windows.ps1 -Cuda -Output build\\reports\\windows-nvidia-validation.json"
    Then the script checks every problem from 001 through 047
    And the JSON summary records each problem as passed, failed, skipped, or unsupported
    And the summary includes the GPU model, compute capability, CUDA versions, and failure diagnostics
