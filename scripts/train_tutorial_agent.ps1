param(
  [int]$Samples = 18000,
  [int]$Epochs = 12,
  [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& ".\.venv\Scripts\python.exe" -m ai_models.tutorial_agent.train_tutorial_agent --device cuda --amp --samples $Samples --epochs $Epochs --batch-size $BatchSize --d-model 192 --n-heads 6 --n-layers 4
