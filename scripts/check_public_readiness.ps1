param(
  [string]$BaseUrl = "http://127.0.0.1:8765"
)

$ErrorActionPreference = "Stop"

$health = Invoke-RestMethod -Uri "$BaseUrl/api/health"
$missing = @()
foreach ($prop in $health.runtime.checkpoints.PSObject.Properties) {
  if (-not $prop.Value) {
    $missing += $prop.Name
  }
}

[pscustomobject]@{
  ok = $health.ok
  torch = $health.runtime.torch
  cudaAvailable = $health.runtime.cudaAvailable
  cudaVersion = $health.runtime.cudaVersion
  gpuName = $health.runtime.gpuName
  allCheckpointsPresent = ($missing.Count -eq 0)
  missingCheckpoints = $missing
} | ConvertTo-Json -Depth 5
