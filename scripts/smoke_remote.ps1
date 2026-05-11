param(
  [Parameter(Mandatory = $true)]
  [string]$BaseUrl
)

$ErrorActionPreference = "Stop"

$base = $BaseUrl.TrimEnd("/")
$health = Invoke-RestMethod -Uri "$base/api/health"

[pscustomobject]@{
  ok = $health.ok
  torch = $health.runtime.torch
  cuda = $health.runtime.cudaAvailable
  gpu = $health.runtime.gpuName
  checkpoints = $health.runtime.checkpoints
} | ConvertTo-Json -Depth 6
