param(
  [string]$Logs = "runtime\tutorial_response_events.jsonl",
  [string]$FeedbackLogs = "runtime\tutorial_feedback_events.jsonl",
  [string]$Export = "runtime\tutorial_response_real_records.jsonl",
  [string]$PreferenceExport = "runtime\tutorial_response_preferences.jsonl",
  [string]$BaseData = "data\tutorial_response_training_robust_120k.jsonl",
  [string]$Output = "runs\tutorial_response_robust_real",
  [int]$Epochs = 3,
  [int]$BatchSize = 64,
  [int]$ExtraWeight = 2,
  [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
if (!(Test-Path $python)) {
  $python = "python"
}

Write-Host "Exporting real tutorial response records..."
& $python -B scripts\export_tutorial_response_logs.py `
  --logs $Logs `
  --feedback-logs $FeedbackLogs `
  --output $Export `
  --preference-output $PreferenceExport `
  --drop-negative

$device = if ($Cpu) { "cpu" } else { "cuda" }
$ampArgs = @()
if (!$Cpu) {
  $ampArgs = @("--amp")
}

Write-Host "Training tutorial response model on $device..."
& $python -m ai_models.tutorial_response.train_tutorial_response `
  --data $BaseData `
  --extra-data $Export `
  --preference-data $PreferenceExport `
  --extra-weight $ExtraWeight `
  --output $Output `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --device $device `
  @ampArgs

Write-Host "Done. New checkpoint: $Output\best.pt"
