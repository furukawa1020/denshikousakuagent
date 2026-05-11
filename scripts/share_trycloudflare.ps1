param(
  [int]$Port = 8765,
  [switch]$StartBackend
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

$backendProcess = $null
if (-not (Test-Backend)) {
  if (-not $StartBackend) {
    Write-Host "GPU backend is not running."
    Write-Host "Run this first:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\start_public_gpu_backend.ps1"
    Write-Host ""
    Write-Host "Or run this script with -StartBackend."
    exit 1
  }

  $backendProcess = Start-Process `
    -FilePath "powershell" `
    -ArgumentList "-ExecutionPolicy Bypass -File `"$root\scripts\start_public_gpu_backend.ps1`" -Port $Port" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -PassThru
  Start-Sleep -Seconds 4
}

if (-not (Test-Backend)) {
  throw "GPU backend did not become healthy on http://127.0.0.1:$Port"
}

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
  Write-Host "cloudflared is not installed."
  Write-Host "Install it with:"
  Write-Host "  winget install Cloudflare.cloudflared"
  if ($backendProcess) {
    Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
  }
  exit 1
}

try {
  Write-Host "Opening a public HTTPS tunnel to http://127.0.0.1:$Port"
  Write-Host "Copy the generated https://*.trycloudflare.com URL."
  cloudflared tunnel --url "http://127.0.0.1:$Port"
}
finally {
  if ($backendProcess) {
    Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
  }
}
