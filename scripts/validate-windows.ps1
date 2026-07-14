[CmdletBinding()]
param(
  [switch]$Cuda,
  [string]$Output = "build\reports\windows-nvidia-validation.json"
)

$ErrorActionPreference = "Stop"
$configuration = if ($Cuda) { "windows-cuda-debug" } else { "windows-cpu-debug" }
$binary = Join-Path "build\$configuration" "leetllm.exe"
if (-not (Test-Path $binary)) {
  throw "LeetLLM CLI not found at $binary. Configure and build preset $configuration first."
}

$gpu = [ordered]@{
  model = $null
  compute_capability = $null
  cuda_driver_version = $null
  cuda_runtime_version = $null
}
$diagnostics = @()
if ($Cuda) {
  try {
    $query = & nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -ne 0) { throw ($query -join "`n") }
    $fields = ($query | Select-Object -First 1).Split(",").Trim()
    $gpu.model = $fields[0]
    $gpu.compute_capability = $fields[1]
    $gpu.cuda_driver_version = $fields[2]
  } catch {
    $diagnostics += "nvidia-smi: $($_.Exception.Message)"
  }
  try {
    $nvcc = & nvcc --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw ($nvcc -join "`n") }
    $release = ($nvcc | Select-String -Pattern "release\s+([0-9.]+)").Matches.Groups[1].Value
    $gpu.cuda_runtime_version = $release
  } catch {
    $diagnostics += "nvcc: $($_.Exception.Message)"
  }
}

$results = @()
1..47 | ForEach-Object {
  $id = $_.ToString("000")
  $backend = if ($Cuda) { "--cuda" } else { "--cpu" }
  $status = if ($Cuda -and -not $gpu.model) { "unsupported" } else { "failed" }
  $message = ""
  if ($status -ne "unsupported") {
    try {
      $text = & $binary check $id $backend 2>&1
      $message = $text -join "`n"
      $status = if ($LASTEXITCODE -eq 0) { "passed" } else { "failed" }
    } catch {
      $message = $_.Exception.Message
      $status = "failed"
    }
  } else {
    $message = "NVIDIA GPU metadata unavailable"
  }
  $results += [ordered]@{ problem = $id; status = $status; diagnostics = $message }
  Write-Host "p$id $status"
}

$summary = [ordered]@{
  generated_at_utc = [DateTime]::UtcNow.ToString("o")
  preset = $configuration
  gpu = $gpu
  diagnostics = $diagnostics
  problems = $results
}
$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $Output
Write-Host "Validation summary written to $Output"
if ($results.status -contains "failed") { exit 1 }

