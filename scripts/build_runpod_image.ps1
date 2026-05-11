param(
  [Parameter(Mandatory = $true)]
  [string]$Image
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path "runs\tutorial_agent\best.pt")) {
  throw "runs/ checkpoints are missing. Train or copy runs/ before building the RunPod image."
}

docker build -f Dockerfile.runpod -t $Image .
docker push $Image
