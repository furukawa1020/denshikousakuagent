param(
  [string]$Logs = "runtime\tutorial_response_events.jsonl",
  [string]$FeedbackLogs = "runtime\tutorial_feedback_events.jsonl",
  [string]$Export = "runtime\tutorial_response_real_records.jsonl",
  [string]$PreferenceExport = "runtime\tutorial_response_preferences.jsonl",
  [string]$ClassifierExport = "runtime\tutorial_answer_classifier_real_records.jsonl",
  [string]$BaseData = "data\tutorial_response_training_robust_120k.jsonl",
  [string]$ClassifierBaseData = "data\tutorial_answer_classifier_training.jsonl",
  [string]$Output = "runs\tutorial_response_robust_real",
  [string]$ClassifierOutput = "runs\tutorial_answer_classifier_real",
  [int]$Epochs = 3,
  [int]$ClassifierEpochs = 3,
  [int]$BatchSize = 64,
  [int]$ClassifierBatchSize = 192,
  [int]$ClassifierSamples = 70000,
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

Write-Host "Exporting real answer-classifier records..."
& $python -B scripts\export_tutorial_answer_classifier_logs.py `
  --logs $Logs `
  --feedback-logs $FeedbackLogs `
  --output $ClassifierExport `
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

Write-Host "Training tutorial answer classifier on $device..."
& $python -m ai_models.tutorial_response.train_answer_classifier `
  --data $ClassifierBaseData `
  --extra-data $ClassifierExport `
  --extra-weight $ExtraWeight `
  --samples $ClassifierSamples `
  --output $ClassifierOutput `
  --epochs $ClassifierEpochs `
  --batch-size $ClassifierBatchSize `
  --device $device `
  @ampArgs

Write-Host "Done. New checkpoint: $Output\best.pt"
Write-Host "Done. New classifier checkpoint: $ClassifierOutput\best.pt"
