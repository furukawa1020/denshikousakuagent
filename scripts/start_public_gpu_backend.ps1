param(
  [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HOST = "127.0.0.1"
$env:PORT = [string]$Port
$env:PYTHONUNBUFFERED = "1"
& ".\.venv\Scripts\python.exe" "backend\server.py"
