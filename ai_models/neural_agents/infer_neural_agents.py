from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ai_models.makergraph.tokenizer import MakerTokenizer

from .model import load_checkpoint
from .taxonomy import (
    BOARD_CLASSES,
    BOARD_LABELS,
    DEBUG_CAUSES,
    DEBUG_LABELS,
    FIRMWARE_CLASSES,
    FIRMWARE_VARIANTS,
    PIN_PROFILES,
    PIN_PROFILE_VALUES,
    PROJECT_LABELS,
    RISK_CLASSES,
    SAFETY_LABELS,
)


PROJECT_VARIANTS = {
    "light_charm": ["light_charm_minimal", "light_charm_debug"],
    "desk_pet": ["desk_pet_led", "desk_pet_buzzer"],
    "plant_ping": ["plant_ping_led", "plant_ping_buzzer"],
    "posture_guard": ["posture_guard_led", "posture_guard_buzzer"],
    "temp_face": ["temp_face_serial", "temp_face_i2c_ready"],
}

BOARD_PIN_PROFILES = {
    "arduino_uno": ["arduino_default"],
    "esp32": ["esp32_default", "esp32_grove_safe"],
    "m5stack": ["m5stack_grove"],
    "pico": ["pico_default"],
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
        firmware_variant_probs = outputs["firmware_variant_logits"][0].softmax(dim=-1).detach().cpu()
        board_probs = outputs["board_logits"][0].softmax(dim=-1).detach().cpu()
        pin_profile_probs = outputs["pin_profile_logits"][0].softmax(dim=-1).detach().cpu()

        risk_index = calibrated_risk_index(safety_probs, risk_probs)
        firmware_id = select_firmware_id(payload, text, firmware_probs)
        firmware_index = FIRMWARE_CLASSES.index(firmware_id)
        firmware_variant_id = select_firmware_variant(payload, text, firmware_id, firmware_variant_probs)
        firmware_variant_index = FIRMWARE_VARIANTS.index(firmware_variant_id)
        board_id = select_board_id(payload, text, board_probs)
        board_index = BOARD_CLASSES.index(board_id)
        pin_profile_id = select_pin_profile(payload, text, board_id, pin_profile_probs)
        pin_profile_index = PIN_PROFILES.index(pin_profile_id)
        debug_scores = calibrated_debug_scores(payload, text, debug_probs)
        debug_ranked = sorted(range(len(DEBUG_CAUSES)), key=lambda index: debug_scores[index], reverse=True)
        primary_debug = DEBUG_CAUSES[debug_ranked[0]]
        code = render_firmware_code(
            firmware_id,
            firmware_variant_id,
            board_id,
            pin_profile_id,
            primary_debug=primary_debug,
            stage=str(payload.get("nextStage") or payload.get("currentStage") or ""),
        )
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
                "score": round(float(debug_scores[index]), 4),
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
                "variant": {
                    "id": firmware_variant_id,
                    "confidence": round(float(firmware_variant_probs[firmware_variant_index]), 4),
                    "distribution": {FIRMWARE_VARIANTS[index]: round(float(value), 4) for index, value in enumerate(firmware_variant_probs.tolist())},
                },
                "board": {
                    "id": board_id,
                    "label": BOARD_LABELS[board_id],
                    "confidence": round(float(board_probs[board_index]), 4),
                    "distribution": {BOARD_CLASSES[index]: round(float(value), 4) for index, value in enumerate(board_probs.tolist())},
                },
                "pinProfile": {
                    "id": pin_profile_id,
                    "pins": PIN_PROFILE_VALUES[pin_profile_id],
                    "confidence": round(float(pin_profile_probs[pin_profile_index]), 4),
                    "distribution": {PIN_PROFILES[index]: round(float(value), 4) for index, value in enumerate(pin_profile_probs.tolist())},
                },
                "language": "Arduino C++",
                "code": code,
                "generationMode": "neural_multiclass_contextual_decode",
                "staticChecks": static_checks(code),
            },
            "decisionTrace": {
                "firmwareDecoder": "project-conditioned neural class",
                "variantDecoder": "project-compatible neural variant",
                "boardDecoder": "selected-board conditioned neural class",
                "pinDecoder": "board-compatible neural pin profile",
                "debugDecoder": "symptom-conditioned neural ranking",
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


def select_firmware_id(payload: dict[str, Any], text: str, firmware_probs: torch.Tensor) -> str:
    explicit = str(payload.get("projectId") or payload.get("project_id") or "").strip()
    if explicit in FIRMWARE_CLASSES:
        return explicit
    lowered = text.lower()
    for project in FIRMWARE_CLASSES:
        if f"projectid {project}" in lowered or f"project: {project}" in lowered:
            return project
    return FIRMWARE_CLASSES[int(firmware_probs.argmax())]


def select_firmware_variant(payload: dict[str, Any], text: str, project_id: str, variant_probs: torch.Tensor) -> str:
    allowed = PROJECT_VARIANTS.get(project_id, FIRMWARE_VARIANTS)
    lowered = text.lower()
    explicit = str(payload.get("firmwareVariant") or payload.get("firmware_variant") or "").strip()
    if explicit in allowed:
        return explicit
    for variant in allowed:
        if variant.lower() in lowered:
            return variant
    allowed_indexes = [FIRMWARE_VARIANTS.index(variant) for variant in allowed]
    best_index = max(allowed_indexes, key=lambda index: float(variant_probs[index]))
    return FIRMWARE_VARIANTS[best_index]


def select_board_id(payload: dict[str, Any], text: str, board_probs: torch.Tensor) -> str:
    explicit = str(payload.get("board") or payload.get("boardClass") or payload.get("board_class") or "").strip()
    if explicit in BOARD_CLASSES:
        return explicit
    lowered = text.lower()
    aliases = {
        "arduino_uno": ["arduino_uno", "arduino uno"],
        "esp32": ["esp32"],
        "m5stack": ["m5stack", "m5"],
        "pico": ["pico", "raspberry pi pico"],
    }
    for board, names in aliases.items():
        if any(name in lowered for name in names):
            return board
    return BOARD_CLASSES[int(board_probs.argmax())]


def select_pin_profile(payload: dict[str, Any], text: str, board_id: str, pin_probs: torch.Tensor) -> str:
    allowed = BOARD_PIN_PROFILES.get(board_id, PIN_PROFILES)
    explicit = str(payload.get("pinProfile") or payload.get("pin_profile") or "").strip()
    if explicit in allowed:
        return explicit
    lowered = text.lower()
    for profile in allowed:
        if profile.lower() in lowered:
            return profile
    allowed_indexes = [PIN_PROFILES.index(profile) for profile in allowed]
    best_index = max(allowed_indexes, key=lambda index: float(pin_probs[index]))
    return PIN_PROFILES[best_index]


def calibrated_debug_scores(payload: dict[str, Any], text: str, debug_probs: torch.Tensor) -> list[float]:
    scores = [float(value) for value in debug_probs.tolist()]
    question = str(payload.get("question") or payload.get("lastQuestion") or "")
    answer = str(payload.get("answer") or payload.get("lastAnswer") or "")
    interpreted = str(payload.get("interpreted") or payload.get("lastInterpreted") or "")
    next_instruction = str(payload.get("nextInstruction") or "")
    evidence = " ".join([
        text,
        str(payload.get("symptom") or ""),
        str(payload.get("compileLog") or ""),
        str(payload.get("serialLog") or ""),
        question,
        answer,
        interpreted,
        next_instruction,
    ]).lower()

    boosts = {
        "usb_port_or_driver": ["upload_failed", "com", "port", "usb", "書き込", "アップロード", "checkpoint", "認識", "ポート"],
        "wrong_pin_mapping": ["wrong pin", "gpio番号", "ピン番号", "番号が違", "配線先", "コードと配線"],
        "missing_gnd": ["gnd", "ground", "グランド", "共通", "マイナス"],
        "led_polarity_or_resistor": ["led_not_lighting", "led", "光らない", "抵抗", "極性", "長い足", "短い足"],
        "sensor_power_or_signal": ["sensor_static", "sensor", "センサー", "値が変わ", "値が出", "analog", "sig"],
        "library_missing": ["library", "no such file", "ライブラリ", "ボードパッケージ", "compile", "コンパイル"],
        "brownout_power": ["brownout", "reset", "電源", "落ちる", "再起動", "モーター", "サーボ"],
        "buzzer_pin_or_polarity": ["buzzer", "tone", "ブザー", "鳴らない", "音"],
    }
    for cause, tokens in boosts.items():
        index = DEBUG_CAUSES.index(cause)
        if any(token in evidence for token in tokens):
            scores[index] += 1.25

    symptom = str(payload.get("symptom") or "").strip()
    direct = {
        "upload_failed": "usb_port_or_driver",
        "app_stuck_after_checkpoint": "usb_port_or_driver",
        "sensor_static": "sensor_power_or_signal",
        "led_not_lighting": "led_polarity_or_resistor",
    }.get(symptom)
    if direct:
        scores[DEBUG_CAUSES.index(direct)] += 1.75

    apply_answer_progress(scores, question, answer, interpreted, next_instruction)

    if "brownout_power" in DEBUG_CAUSES and not any(token in evidence for token in ["brownout", "reset", "電源", "落ちる", "再起動", "モーター", "サーボ"]):
        scores[DEBUG_CAUSES.index("brownout_power")] *= 0.35

    total = sum(max(score, 0.0001) for score in scores)
    return [max(score, 0.0001) / total for score in scores]


def apply_answer_progress(scores: list[float], question: str, answer: str, interpreted: str, next_instruction: str) -> None:
    context = f"{question} {interpreted} {next_instruction}".lower()
    checked_context = f"{question} {interpreted}".lower()
    answer_text = answer.lower()
    ok_answer = any(token in answer_text for token in [
        "yes",
        "ok",
        "done",
        "できた",
        "大丈夫",
        "合って",
        "つながって",
        "入って",
        "見え",
        "光った",
        "鳴った",
        "変わった",
        "出た",
        "はい",
    ])
    bad_answer = any(token in answer_text for token in [
        "no",
        "ng",
        "だめ",
        "違う",
        "ない",
        "できない",
        "光らない",
        "鳴らない",
        "変わらない",
        "出ない",
        "分からない",
        "わからない",
    ])

    topics = [
        ("led_polarity_or_resistor", ["led", "長い足", "短い足", "抵抗", "極性"]),
        ("missing_gnd", ["gnd", "ground", "共通", "グランド"]),
        ("wrong_pin_mapping", ["gpio", "ピン番号", "配線先", "物理ピン", "コード"]),
        ("sensor_power_or_signal", ["sensor", "センサー", "値", "sig", "analog"]),
        ("usb_port_or_driver", ["usb", "com", "ポート", "書き込", "アップロード", "checkpoint"]),
        ("buzzer_pin_or_polarity", ["buzzer", "ブザー", "tone", "音"]),
    ]
    peak_score = max(scores)

    for cause, tokens in topics:
        if any(token in context for token in tokens):
            index = DEBUG_CAUSES.index(cause)
            if ok_answer:
                scores[index] *= 0.08
            elif bad_answer:
                scores[index] += 1.8

    if ok_answer and any(token in context for token in ["led", "長い足", "抵抗", "極性"]):
        scores[DEBUG_CAUSES.index("led_polarity_or_resistor")] *= 0.35
        scores[DEBUG_CAUSES.index("wrong_pin_mapping")] = max(scores[DEBUG_CAUSES.index("wrong_pin_mapping")], peak_score * 1.15)
        scores[DEBUG_CAUSES.index("missing_gnd")] = max(scores[DEBUG_CAUSES.index("missing_gnd")], peak_score * 0.55)
    if ok_answer and any(token in context for token in ["gnd", "ground", "共通"]):
        scores[DEBUG_CAUSES.index("missing_gnd")] *= 0.35
        scores[DEBUG_CAUSES.index("wrong_pin_mapping")] = max(scores[DEBUG_CAUSES.index("wrong_pin_mapping")], peak_score * 1.05)
        scores[DEBUG_CAUSES.index("sensor_power_or_signal")] = max(scores[DEBUG_CAUSES.index("sensor_power_or_signal")], peak_score * 0.5)
    if ok_answer and any(token in checked_context for token in ["ピン番号", "gpio", "配線先"]):
        scores[DEBUG_CAUSES.index("wrong_pin_mapping")] *= 0.5
        scores[DEBUG_CAUSES.index("sensor_power_or_signal")] = max(scores[DEBUG_CAUSES.index("sensor_power_or_signal")], peak_score * 0.85)
        scores[DEBUG_CAUSES.index("library_missing")] = max(scores[DEBUG_CAUSES.index("library_missing")], peak_score * 0.35)
    if "次" in next_instruction and any(token in next_instruction.lower() for token in ["ピン", "gpio", "コード"]):
        scores[DEBUG_CAUSES.index("wrong_pin_mapping")] = max(scores[DEBUG_CAUSES.index("wrong_pin_mapping")], peak_score * 1.25)
    if "次" in next_instruction and any(token in next_instruction.lower() for token in ["値", "シリアル", "センサー"]):
        scores[DEBUG_CAUSES.index("sensor_power_or_signal")] += 0.8


def next_question(cause: str) -> str:
    questions = {
        "usb_port_or_driver": "USBケーブルを差し直した時、ボードはCOM/シリアルポートとして表示されますか？",
        "wrong_pin_mapping": "ジャンパ線が刺さっている物理ピンと、コードのピン番号は同じですか？",
        "missing_gnd": "すべての部品のGNDが、ボードのGNDと同じ基準につながっていますか？",
        "led_polarity_or_resistor": "LEDの長い足は信号側で、220〜330Ωの抵抗が直列に入っていますか？",
        "sensor_power_or_signal": "センサーを動かした時、シリアルモニタの値は変わりますか？",
        "library_missing": "コンパイルログの最初の No such file または board package 行は何ですか？",
        "brownout_power": "ブザー、モーター、画面が動いた瞬間にボードが再起動しますか？",
        "buzzer_pin_or_polarity": "ブザーの＋側は tone() で使っているピンにつながっていますか？",
        "unknown": "一番小さい症状は、書き込みエラー、光らない、鳴らない、値が変わらない、再起動のどれですか？",
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


def render_firmware_code(
    project_id: str,
    variant_id: str,
    board_id: str,
    pin_profile_id: str,
    primary_debug: str = "unknown",
    stage: str = "",
) -> str:
    pins = PIN_PROFILE_VALUES[pin_profile_id]
    led = pins["led"]
    buzzer = pins["buzzer"]
    digital = pins["digital"]
    analog = pins["analog"]
    board_label = BOARD_LABELS[board_id]
    setup_probe = debug_setup_probe(primary_debug, led, buzzer, digital, analog)
    loop_probe = debug_loop_probe(primary_debug, digital, analog)
    if project_id == "light_charm":
        return f"""// Lchika-no-tsuzuki firmware: dark-reactive LED charm
// Debug focus: {primary_debug}
// Board: {board_label}
const int LED_PIN = {led};
const int LIGHT_PIN = {analog};
const int DARK_THRESHOLD = 450;

void setup() {{
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
{setup_probe}
}}

void loop() {{
{loop_probe}
  int lightValue = analogRead(LIGHT_PIN);
  bool isDark = lightValue < DARK_THRESHOLD;
  digitalWrite(LED_PIN, isDark ? HIGH : LOW);
  Serial.print("light=");
  Serial.print(lightValue);
  Serial.print(" dark=");
  Serial.println(isDark ? "yes" : "no");
  delay(200);
}}
"""
    if project_id == "plant_ping":
        buzzer_block = ""
        if variant_id == "plant_ping_buzzer":
            buzzer_block = f"""
  if (isDry) {{
    tone(BUZZER_PIN, 988, 70);
  }}
"""
        return f"""// Lchika-no-tsuzuki firmware: plant watering notifier
// Debug focus: {primary_debug}
// Variant: {variant_id}
// Board: {board_label}
const int LED_PIN = {led};
const int BUZZER_PIN = {buzzer};
const int SOIL_PIN = {analog};
const int DRY_THRESHOLD = 420;

void setup() {{
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
{setup_probe}
}}

void loop() {{
{loop_probe}
  int soilValue = analogRead(SOIL_PIN);
  bool isDry = soilValue < DRY_THRESHOLD;
  digitalWrite(LED_PIN, isDry ? HIGH : LOW);{buzzer_block}
  Serial.print("soil=");
  Serial.print(soilValue);
  Serial.print(" dry=");
  Serial.println(isDry ? "yes" : "no");
  delay(500);
}}
"""
    if project_id == "desk_pet":
        buzzer_block = ""
        if variant_id == "desk_pet_buzzer":
            buzzer_block = f"""
  if (isNear) {{
    tone(BUZZER_PIN, 880, 80);
  }}
"""
        return f"""// Lchika-no-tsuzuki firmware: proximity desk pet
// Debug focus: {primary_debug}
// Variant: {variant_id}
// Board: {board_label}
const int LED_PIN = {led};
const int BUZZER_PIN = {buzzer};
const int DISTANCE_PIN = {digital};

void setup() {{
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(DISTANCE_PIN, INPUT);
{setup_probe}
}}

void loop() {{
{loop_probe}
  int nearSignal = digitalRead(DISTANCE_PIN);
  bool isNear = nearSignal == HIGH;
  digitalWrite(LED_PIN, isNear ? HIGH : LOW);{buzzer_block}
  Serial.print("near=");
  Serial.println(isNear ? "yes" : "no");
  delay(150);
}}
"""
    if project_id == "posture_guard":
        buzzer_block = ""
        if variant_id == "posture_guard_buzzer":
            buzzer_block = f"""
  if (tooClose) {{
    tone(BUZZER_PIN, 1200, 60);
  }}
"""
        return f"""// Lchika-no-tsuzuki firmware: posture alert device
// Debug focus: {primary_debug}
// Variant: {variant_id}
// Board: {board_label}
const int LED_PIN = {led};
const int BUZZER_PIN = {buzzer};
const int DISTANCE_PIN = {digital};

void setup() {{
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(DISTANCE_PIN, INPUT);
{setup_probe}
}}

void loop() {{
{loop_probe}
  int tooCloseSignal = digitalRead(DISTANCE_PIN);
  bool tooClose = tooCloseSignal == HIGH;
  digitalWrite(LED_PIN, tooClose ? HIGH : LOW);{buzzer_block}
  Serial.print("tooClose=");
  Serial.println(tooClose ? "yes" : "no");
  delay(200);
}}
"""
    wire_include = "#include <Wire.h>\n\n" if variant_id == "temp_face_i2c_ready" else ""
    return f"""// Lchika-no-tsuzuki firmware: temperature face display
// Debug focus: {primary_debug}
// Variant: {variant_id}
// Board: {board_label}
{wire_include}const int TEMP_PIN = {analog};
const int HOT_THRESHOLD = 640;

void setup() {{
  Serial.begin(115200);
{setup_probe}
}}

void loop() {{
{loop_probe}
  int tempRaw = analogRead(TEMP_PIN);
  Serial.print("tempRaw=");
  Serial.print(tempRaw);
  Serial.print(" face=");
  if (tempRaw > HOT_THRESHOLD) {{
    Serial.println("hot");
  }} else if (tempRaw < HOT_THRESHOLD - 180) {{
    Serial.println("cold");
  }} else {{
    Serial.println("comfortable");
  }}
    delay(500);
}}
"""


def debug_setup_probe(primary_debug: str, led: int, buzzer: int, digital: int, analog: Any) -> str:
    lines = [
        '  Serial.println("Lchika-no-tsuzuki start");',
        f'  Serial.println("pins led={led} buzzer={buzzer} digital={digital} analog={analog}");',
    ]
    if primary_debug in {"wrong_pin_mapping", "led_polarity_or_resistor"}:
        lines.extend([
            "  digitalWrite(LED_PIN, HIGH);",
            "  delay(250);",
            "  digitalWrite(LED_PIN, LOW);",
        ])
    if primary_debug == "buzzer_pin_or_polarity":
        lines.append("  tone(BUZZER_PIN, 660, 120);")
    return "\n".join(lines)


def debug_loop_probe(primary_debug: str, digital: int, analog: Any) -> str:
    if primary_debug == "wrong_pin_mapping":
        return f'  Serial.println("debug: confirm jumper pin numbers digital={digital}");'
    if primary_debug == "sensor_power_or_signal":
        return f'  Serial.println("debug: move sensor and watch values analog={analog} digital={digital}");'
    if primary_debug == "missing_gnd":
        return '  Serial.println("debug: all parts must share board GND");'
    if primary_debug == "usb_port_or_driver":
        return '  Serial.println("debug: if this line appears, upload and serial are working");'
    return ""


def static_checks(code: str) -> dict[str, Any]:
    return {
        "hasSetup": "void setup()" in code,
        "hasLoop": "void loop()" in code,
        "hasSerialDebug": "Serial." in code or "Serial.begin" in code,
        "mentionsDangerousMains": "AC100V" in code or "relay" in code.lower(),
        "length": len(code),
    }


if __name__ == "__main__":
    main()
