param(
  [string]$Output = "dist\hf-space"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path "runs\tutorial_agent\best.pt")) {
  throw "runs/ checkpoints are missing. Train or copy runs/ before packaging the Space."
}

$target = [System.IO.Path]::GetFullPath((Join-Path $root $Output))
$rootFull = [System.IO.Path]::GetFullPath($root)
if (-not $target.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Output must stay inside the repository: $target"
}

if (Test-Path $target) {
  Remove-Item -LiteralPath $target -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $target | Out-Null

Copy-Item -Recurse -Path "ai_models" -Destination (Join-Path $target "ai_models")
Copy-Item -Recurse -Path "backend" -Destination (Join-Path $target "backend")
Copy-Item -Recurse -Path "frontend" -Destination (Join-Path $target "frontend")
Copy-Item -Recurse -Path "data" -Destination (Join-Path $target "data")
Copy-Item -Recurse -Path "runs" -Destination (Join-Path $target "runs")
Copy-Item -Path "requirements-cpu.txt" -Destination (Join-Path $target "requirements-cpu.txt")
Copy-Item -Path "deploy\huggingface\Dockerfile" -Destination (Join-Path $target "Dockerfile")
Copy-Item -Path "deploy\huggingface\README.md" -Destination (Join-Path $target "README.md")
Copy-Item -Path "deploy\huggingface\gitattributes" -Destination (Join-Path $target ".gitattributes")

Write-Host "Prepared Hugging Face Space at $target"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd $target"
Write-Host "  git init"
Write-Host "  git lfs install"
Write-Host "  git lfs track `"*.pt`""
Write-Host "  git add ."
Write-Host "  git commit -m `"Deploy L Chika No Tsuzuki`""
Write-Host "  git remote add origin https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE"
Write-Host "  git push -u origin main"
