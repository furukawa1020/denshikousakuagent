param(
  [switch]$Build
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$args = @("compose", "-f", "compose.gpu.yml", "up")
if ($Build) {
  $args += "--build"
}
docker @args
