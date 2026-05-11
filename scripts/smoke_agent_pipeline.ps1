param(
  [string]$BaseUrl = "http://127.0.0.1:8765"
)

$ErrorActionPreference = "Stop"

$health = Invoke-RestMethod -Uri "$BaseUrl/api/health"
$body = @{
  text = "I do not know what to make yet. Budget is 5000 yen. I want something cute. I have ESP32, LED, resistor, and breadboard."
  inventory = "ESP32 LED resistor breadboard jumper wires USB"
  budget = 5000
  board = "esp32"
  currentStage = "orient"
  symptom = "none"
  device = "cuda"
} | ConvertTo-Json

$agent = Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/agent/run" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body

[pscustomobject]@{
  health = $health.ok
  torch = $health.runtime.torch
  cuda = $health.runtime.cudaAvailable
  gpu = $health.runtime.gpuName
  pipeline = $agent.pipeline
  selectedProject = $agent.selectedProject.title
  tutorialStage = $agent.tutorial.stage.label
  firmwareModel = $agent.firmware.generationMode
} | ConvertTo-Json -Depth 5
