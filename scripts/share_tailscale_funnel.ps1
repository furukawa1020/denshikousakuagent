param(
  [int]$Port = 8765,
  [switch]$StartBackend,
  [switch]$Background
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-Backend {
  try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
    return [bool]$health.ok
  }
  catch {
    return $false
  }
}

if (-not (Test-Backend)) {
  if (-not $StartBackend) {
    Write-Host "GPU backend is not running."
    Write-Host "Run this first:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\start_public_gpu_backend.ps1"
    exit 1
  }
  Start-Process `
    -FilePath "powershell" `
    -ArgumentList "-ExecutionPolicy Bypass -File `"$root\scripts\start_public_gpu_backend.ps1`" -Port $Port" `
    -WorkingDirectory $root `
    -WindowStyle Hidden
  Start-Sleep -Seconds 4
}

if (-not (Get-Command tailscale -ErrorAction SilentlyContinue)) {
  Write-Host "tailscale is not installed."
  Write-Host "Install it from https://tailscale.com/download and log in."
  exit 1
}

if ($Background) {
  tailscale funnel --bg $Port
}
else {
  tailscale funnel $Port
}
