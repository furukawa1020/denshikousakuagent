param(
  [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$server = Start-Process `
  -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList "$root\backend\server.py" `
  -WorkingDirectory $root `
  -WindowStyle Hidden `
  -PassThru

try {
  Start-Sleep -Seconds 3
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health"
  $body = @{
    text = "LED is not lighting. I have ESP32, LED, and resistor. What should I check?"
    projectId = "light_charm"
    currentStage = "debug_triage"
    symptom = "led_not_lighting"
    inventory = "ESP32 LED resistor breadboard jumper wires USB"
    budget = 5000
    board = "esp32"
    device = "cuda"
  } | ConvertTo-Json
  $tutorial = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:$Port/api/tutorial/next" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

  [pscustomobject]@{
    health = $health.ok
    model = $tutorial.model
    device = $tutorial.device
    stage = $tutorial.stage.label
    question = $tutorial.question.text
  } | ConvertTo-Json -Depth 4
}
finally {
  Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}
