param(
  [string]$HostName = "0.0.0.0",
  [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HOST = $HostName
$env:PORT = [string]$Port
$env:PYTHONUNBUFFERED = "1"
& ".\.venv\Scripts\python.exe" "backend\server.py"
