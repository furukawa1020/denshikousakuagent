from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ai_models.makergraph.tokenizer import MakerTokenizer

from .model import load_checkpoint
from .taxonomy import DEBUG_CAUSES, DEBUG_LABELS, FIRMWARE_CLASSES, PROJECT_LABELS, RISK_CLASSES, SAFETY_LABELS


CODEBOOK = {
    "light_charm": """// NeuralAgent selected firmware: Dark-reactive LED charm
const int LED_PIN = 5;
const int SENSOR_PIN = A0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  int lightValue = analogRead(SENSOR_PIN);
  bool isDark = lightValue < 450;
  digitalWrite(LED_PIN, isDark ? HIGH : LOW);
  Serial.print("light=");
  Serial.print(lightValue);
  Serial.print(" dark=");
  Serial.println(isDark);
  delay(200);
}
""",
    "desk_pet": """// NeuralAgent selected firmware: Proximity desk pet
const int LED_PIN = 5;
const int BUZZER_PIN = 18;
const int DISTANCE_PIN = 21;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(DISTANCE_PIN, INPUT);
}

void loop() {
  int nearSignal = digitalRead(DISTANCE_PIN);
  digitalWrite(LED_PIN, nearSignal == HIGH ? HIGH : LOW);
  if (nearSignal == HIGH) {
    tone(BUZZER_PIN, 880, 80);
  }
  Serial.print("near=");
  Serial.println(nearSignal);
  delay(150);
}
""",
    "plant_ping": """// NeuralAgent selected firmware: Plant watering notifier
const int LED_PIN = 5;
const int BUZZER_PIN = 18;
const int SOIL_PIN = A0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
}

void loop() {
  int soilValue = analogRead(SOIL_PIN);
  bool dry = soilValue < 420;
  digitalWrite(LED_PIN, dry ? HIGH : LOW);
  if (dry) {
    tone(BUZZER_PIN, 988, 70);
  }
  Serial.print("soil=");
  Serial.print(soilValue);
  Serial.print(" dry=");
  Serial.println(dry);
  delay(500);
}
""",
    "posture_guard": """// NeuralAgent selected firmware: Posture alert device
const int LED_PIN = 5;
const int BUZZER_PIN = 18;
const int DISTANCE_PIN = 21;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(DISTANCE_PIN, INPUT);
}

void loop() {
  int tooClose = digitalRead(DISTANCE_PIN);
  digitalWrite(LED_PIN, tooClose == HIGH ? HIGH : LOW);
  if (tooClose == HIGH) {
    tone(BUZZER_PIN, 1200, 60);
  }
  Serial.print("tooClose=");
  Serial.println(tooClose);
  delay(200);
}
""",
    "temp_face": """// NeuralAgent selected firmware: Temperature face display
#include <Wire.h>

const int TEMP_PIN = A0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int raw = analogRead(TEMP_PIN);
  float normalized = raw / 1023.0;
  Serial.print("tempRaw=");
  Serial.print(raw);
  Serial.print(" face=");
  if (normalized > 0.65) {
    Serial.println("hot");
  } else if (normalized < 0.35) {
    Serial.println("cold");
  } else {
    Serial.println("comfortable");
  }
  delay(500);
}
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuralAgentTransformer inference.")
    parser.add_argument("--model-dir", default="runs/neural_agents")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = NeuralAgentInference(args.model_dir, args.device)
    print(json.dumps(service.predict({"text": args.text}), ensure_ascii=False, indent=2))


class NeuralAgentInference:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.tokenizer = MakerTokenizer.load(self.model_dir / "tokenizer.json")
        self.model, self.config, self.payload = load_checkpoint(str(self.model_dir / "best.pt"), map_location=self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = build_input_text(payload)
        input_ids, mask = self.tokenizer.encode(text, self.config.max_length, "<log>")
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        outputs = self.model(ids, attention)
        safety_probs = outputs["safety_logits"].sigmoid()[0].detach().cpu()
        risk_probs = outputs["risk_logits"][0].softmax(dim=-1).detach().cpu()
        debug_probs = outputs["debug_logits"][0].softmax(dim=-1).detach().cpu()
        firmware_probs = outputs["firmware_logits"][0].softmax(dim=-1).detach().cpu()

        risk_index = calibrated_risk_index(safety_probs, risk_probs)
        firmware_index = int(firmware_probs.argmax())
        firmware_id = FIRMWARE_CLASSES[firmware_index]
        debug_ranked = debug_probs.argsort(descending=True).tolist()
        safety_ranked = safety_probs.argsort(descending=True).tolist()
        active_safety = [
            {"label": SAFETY_LABELS[index], "score": round(float(safety_probs[index]), 4)}
            for index in safety_ranked
            if float(safety_probs[index]) >= 0.42
        ][:5]
        debug_causes = [
            {
                "cause": DEBUG_CAUSES[index],
                "label": DEBUG_LABELS[DEBUG_CAUSES[index]],
                "score": round(float(debug_probs[index]), 4),
            }
            for index in debug_ranked[:5]
        ]
        return {
            "available": True,
            "model": "NeuralAgentTransformer",
            "device": str(self.device),
            "inputPreview": text[:420],
            "safety": {
                "risk": RISK_CLASSES[risk_index],
                "riskDistribution": {RISK_CLASSES[index]: round(float(value), 4) for index, value in enumerate(risk_probs.tolist())},
                "labels": active_safety,
                "confidence": round(float(risk_probs[risk_index]), 4),
            },
            "debug": {
                "causeRanking": debug_causes,
                "nextQuestion": next_question(debug_causes[0]["cause"]),
                "skillUpdateHint": skill_hint(debug_causes[0]["cause"]),
            },
            "firmware": {
                "projectId": firmware_id,
                "title": PROJECT_LABELS[firmware_id],
                "confidence": round(float(firmware_probs[firmware_index]), 4),
                "classDistribution": {FIRMWARE_CLASSES[index]: round(float(value), 4) for index, value in enumerate(firmware_probs.tolist())},
                "language": "Arduino C++",
                "code": CODEBOOK[firmware_id],
                "generationMode": "neural_classification_codebook",
            },
        }


def build_input_text(payload: dict[str, Any]) -> str:
    keys = [
        "text",
        "symptom",
        "code",
        "compileLog",
        "serialLog",
        "project",
        "projectId",
        "projectGraph",
        "circuitGraph",
        "connectionTable",
        "components",
        "board",
        "budget",
        "inventory",
    ]
    parts = []
    for key in keys:
        value = payload.get(key)
        if value:
            parts.append(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}")
    return " | ".join(parts) or "beginner electronics project led sensor usb breadboard debug"


def calibrated_risk_index(safety_probs: torch.Tensor, risk_probs: torch.Tensor) -> int:
    scores = {SAFETY_LABELS[index]: float(value) for index, value in enumerate(safety_probs.tolist())}
    blocked_score = max(scores.get("ac_mains", 0.0), scores.get("high_voltage", 0.0), scores.get("lipo_charge", 0.0))
    high_score = max(scores.get("motor_direct", 0.0), scores.get("water_power", 0.0), scores.get("thermal_load", 0.0))
    medium_score = max(scores.get("missing_resistor_led", 0.0), scores.get("gnd_not_shared", 0.0), scores.get("voltage_mismatch", 0.0))
    if blocked_score >= 0.30:
        return RISK_CLASSES.index("blocked")
    if high_score >= 0.35:
        return RISK_CLASSES.index("high")
    if medium_score >= 0.40:
        return RISK_CLASSES.index("medium")
    if max(blocked_score, high_score, medium_score) < 0.30 and float(risk_probs.max()) < 0.70:
        return RISK_CLASSES.index("low")
    return int(risk_probs.argmax())


def next_question(cause: str) -> str:
    questions = {
        "usb_port_or_driver": "Does the board appear as a serial/COM port when you reconnect it with a data-capable USB cable?",
        "wrong_pin_mapping": "Which physical pin is the jumper wire connected to, and does it match the pin number in code?",
        "missing_gnd": "Can you confirm that every module GND is connected to the board GND?",
        "led_polarity_or_resistor": "Is the LED long leg on the signal side and is a 220-330 ohm resistor in series?",
        "sensor_power_or_signal": "What value appears in Serial Monitor when you move or touch the sensor?",
        "library_missing": "What is the first 'No such file' or board package line in the compile log?",
        "brownout_power": "Does the board reset exactly when the motor, buzzer, or display turns on?",
        "buzzer_pin_or_polarity": "Is the buzzer positive leg connected to the output pin used by tone()?",
        "unknown": "What is the smallest observable symptom: upload error, no light, no sound, no sensor change, or reset?",
    }
    return questions.get(cause, questions["unknown"])


def skill_hint(cause: str) -> str:
    hints = {
        "missing_gnd": "gnd_common",
        "led_polarity_or_resistor": "resistor_usage",
        "wrong_pin_mapping": "gpio",
        "sensor_power_or_signal": "analog_input",
        "library_missing": "library_install",
        "brownout_power": "power_design",
        "usb_port_or_driver": "upload_environment",
        "buzzer_pin_or_polarity": "digital_output",
    }
    return hints.get(cause, "debugging")


if __name__ == "__main__":
    main()
