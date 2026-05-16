from __future__ import annotations

import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ai_bridge import circuit_validator_inference, inventory_matcher_inference, neural_agent_inference, neural_bom_inference, runtime_health, skillrec_inference, transformer_intent, transformer_project_graph, tutorial_agent_inference, tutorial_answer_classifier_inference, tutorial_response_inference, wirechecknet_inference


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = ROOT / "frontend"
RUNTIME_ROOT = Path(os.environ.get("RUNTIME_ROOT", str(Path("/data") / "lchika-runtime" if Path("/data").exists() and os.access("/data", os.W_OK) else ROOT / "runtime")))
TUTORIAL_EVENT_LOG = Path(os.environ.get("TUTORIAL_EVENT_LOG", str(RUNTIME_ROOT / "tutorial_response_events.jsonl")))
TUTORIAL_FEEDBACK_LOG = Path(os.environ.get("TUTORIAL_FEEDBACK_LOG", str(RUNTIME_ROOT / "tutorial_feedback_events.jsonl")))
MAX_TUTORIAL_LOG_TEXT = int(os.environ.get("MAX_TUTORIAL_LOG_TEXT", "1200"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))
STAGE_ORDER = ["orient", "parts_check", "minimal_circuit", "firmware_upload", "observe_serial", "debug_triage", "standard_build", "enclosure", "extension", "completion_log"]
PROGRESS_STAGE_ORDER = ["orient", "parts_check", "minimal_circuit", "firmware_upload", "observe_serial", "standard_build", "enclosure", "extension", "completion_log"]


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    category: str
    purpose: str
    price_min: int
    price_max: int
    required: bool
    quantity: int
    voltage: str
    current: str
    interface: str
    beginner_score: int
    risk_level: str
    soldering: bool
    compatible_boards: list[str]
    keyword: str
    notes: str
    alternatives: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MakerProject:
    id: str
    title: str
    summary: str
    mood_tags: list[str]
    problem_tags: list[str]
    example_tags: list[str]
    difficulty: int
    estimated_time: str
    cost_min: int
    cost_max: int
    board: str
    components: list[str]
    optional_components: list[str]
    skills: list[str]
    failure_points: list[str]
    minimal: str
    standard: str
    extension: str
    next_projects: list[str]
    visual_hint: str


COMPONENTS: dict[str, Component] = {
    "esp32_devkit": Component(
        "esp32_devkit",
        "ESP32開発ボード",
        "マイコン",
        "センサー値を読み、LEDやブザーを制御する",
        900,
        1800,
        True,
        1,
        "3.3V logic / USB 5V input",
        "通常80-240mA",
        "GPIO / I2C / UART / SPI",
        4,
        "low",
        False,
        ["ESP32"],
        "ESP32 開発ボード USB-C 技適",
        "5V専用センサーを直結しない。USBケーブルも忘れず確認する。",
        ["Arduino Uno R4", "M5Atom Lite"],
    ),
    "arduino_uno": Component(
        "arduino_uno",
        "Arduino Uno互換ボード",
        "マイコン",
        "初心者向けに安定したGPIO制御を行う",
        1200,
        2800,
        True,
        1,
        "5V logic / USB 5V input",
        "通常50-120mA",
        "GPIO / Analog / I2C",
        5,
        "low",
        False,
        ["Arduino Uno"],
        "Arduino Uno 互換 CH340",
        "互換品はドライバが必要な場合がある。",
        ["Raspberry Pi Pico", "ESP32"],
    ),
    "m5stickc": Component(
        "m5stickc",
        "M5StickC Plus2",
        "マイコン",
        "画面・ボタン・電池を一体で扱う",
        3600,
        5200,
        True,
        1,
        "3.3V logic / USB 5V input",
        "通常80-300mA",
        "Grove / GPIO / I2C",
        4,
        "medium",
        False,
        ["M5Stack"],
        "M5StickC Plus2",
        "内蔵電池は分解しない。充電中の発熱に注意する。",
        ["M5AtomS3", "ESP32 + OLED"],
    ),
    "led_5mm": Component(
        "led_5mm",
        "5mm LED",
        "出力部品",
        "光で状態を知らせる",
        20,
        80,
        True,
        2,
        "2.0-3.2V forward",
        "5-20mA",
        "Digital output",
        5,
        "low",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico", "micro:bit"],
        "5mm LED 赤 緑 セット",
        "必ず抵抗を直列に入れる。足の長い方がアノード。",
        ["NeoPixel LED", "内蔵LED"],
    ),
    "resistor_220": Component(
        "resistor_220",
        "220Ω抵抗",
        "配線部品",
        "LEDに流れる電流を制限する",
        5,
        30,
        True,
        2,
        "passive",
        "電流制限用",
        "Series wiring",
        5,
        "low",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico", "micro:bit"],
        "220Ω 抵抗 1/4W",
        "LEDをGPIOへ直結しないための必須部品。",
        ["330Ω抵抗", "470Ω抵抗"],
    ),
    "breadboard": Component(
        "breadboard",
        "ブレッドボード",
        "配線部品",
        "はんだ付けなしで回路を組む",
        300,
        700,
        True,
        1,
        "passive",
        "なし",
        "Prototype board",
        5,
        "low",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico", "micro:bit"],
        "ブレッドボード 400穴",
        "電源ラインの途中切れに注意する。",
        ["Groveケーブル", "ユニバーサル基板"],
    ),
    "jumper_wires": Component(
        "jumper_wires",
        "ジャンパ線セット",
        "配線部品",
        "マイコンと部品を接続する",
        250,
        650,
        True,
        1,
        "passive",
        "なし",
        "Dupont wire",
        5,
        "low",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico", "micro:bit"],
        "ジャンパーワイヤー オスオス オスメス",
        "オス-オス、オス-メスの種類を確認する。",
        ["Groveケーブル", "ワニ口クリップ"],
    ),
    "usb_cable": Component(
        "usb_cable",
        "データ通信対応USBケーブル",
        "電源",
        "書き込みと給電に使う",
        300,
        1200,
        True,
        1,
        "USB 5V",
        "ケーブル依存",
        "USB",
        5,
        "low",
        False,
        ["Arduino Uno", "ESP32", "M5Stack", "Raspberry Pi Pico"],
        "USB データ通信 対応 ケーブル",
        "充電専用ケーブルだと書き込めない。",
        ["手持ちのデータ対応ケーブル"],
    ),
    "light_sensor": Component(
        "light_sensor",
        "CdSセルまたは光センサーモジュール",
        "センサー",
        "周囲の明るさを読む",
        120,
        600,
        True,
        1,
        "3.3V/5V module or passive divider",
        "1-5mA",
        "Analog input",
        4,
        "low",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico"],
        "光センサー モジュール CdS Arduino",
        "CdS単体は分圧用の抵抗が必要。",
        ["フォトトランジスタ", "照度センサーBH1750"],
    ),
    "distance_sensor": Component(
        "distance_sensor",
        "距離センサー",
        "センサー",
        "近づいたことを検知する",
        400,
        1500,
        True,
        1,
        "3.3V/5V depends on module",
        "10-30mA",
        "Digital pulse / I2C",
        3,
        "low",
        False,
        ["Arduino Uno", "ESP32", "M5Stack"],
        "VL53L0X 距離センサー Grove",
        "HC-SR04をESP32で使う場合はEchoの5Vに注意する。",
        ["ToF距離センサー", "PIR人感センサー"],
    ),
    "buzzer": Component(
        "buzzer",
        "圧電ブザー",
        "出力部品",
        "音で反応を伝える",
        100,
        500,
        True,
        1,
        "3.3V/5V",
        "5-30mA",
        "Digital output / PWM",
        4,
        "low",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico"],
        "圧電ブザー Arduino",
        "大きなスピーカーやモーターをGPIO直結しない。",
        ["小型スピーカー + アンプ", "M5内蔵ブザー"],
    ),
    "servo_sg90": Component(
        "servo_sg90",
        "SG90サーボモーター",
        "出力部品",
        "小さな動きを作る",
        350,
        900,
        False,
        1,
        "4.8-6V",
        "起動時500mA以上の場合あり",
        "PWM",
        3,
        "medium",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico"],
        "SG90 サーボモーター",
        "マイコンの5Vピンだけで安定しない場合がある。電源容量を別枠で見る。",
        ["振動モーター + ドライバ", "LEDだけの表現"],
    ),
    "dht_sensor": Component(
        "dht_sensor",
        "温湿度センサー",
        "センサー",
        "温度や湿度を読む",
        300,
        900,
        True,
        1,
        "3.3V/5V",
        "1-2.5mA",
        "OneWire-like digital",
        4,
        "low",
        False,
        ["Arduino Uno", "ESP32", "M5Stack"],
        "DHT11 DHT22 温湿度センサー",
        "ライブラリとピン番号をコードと合わせる。",
        ["SHT31", "BME280"],
    ),
    "oled_display": Component(
        "oled_display",
        "0.96インチOLEDディスプレイ",
        "出力部品",
        "数値や表情を表示する",
        600,
        1500,
        False,
        1,
        "3.3V/5V module",
        "10-30mA",
        "I2C",
        3,
        "low",
        False,
        ["Arduino Uno", "ESP32", "Raspberry Pi Pico"],
        "OLED 0.96 I2C SSD1306",
        "I2CアドレスとSDA/SCLピンを確認する。",
        ["M5StickC内蔵画面", "NeoPixel表情"],
    ),
    "soil_sensor": Component(
        "soil_sensor",
        "静電容量式土壌水分センサー",
        "センサー",
        "植物の土の乾き具合を読む",
        450,
        1000,
        True,
        1,
        "3.3V/5V",
        "5-10mA",
        "Analog input",
        4,
        "medium",
        False,
        ["Arduino Uno", "ESP32"],
        "静電容量式 土壌水分センサー",
        "水回りではUSB電源と配線の位置を離す。ポンプ制御から始めない。",
        ["手動入力ボタン", "湿度センサー"],
    ),
    "case_material": Component(
        "case_material",
        "紙箱・アクリル板・固定具",
        "外装素材",
        "作品として見せられる形にする",
        300,
        1500,
        False,
        1,
        "passive",
        "なし",
        "enclosure",
        5,
        "low",
        False,
        ["all"],
        "100均 ケース 両面テープ 結束バンド",
        "金属ケースはショート防止の絶縁を入れる。",
        ["段ボール", "3Dプリントケース"],
    ),
}


PROJECTS: dict[str, MakerProject] = {
    "desk_pet": MakerProject(
        "desk_pet",
        "近づくと鳴く机上ペット",
        "机に置いた小さな相棒が、人が近づくとLEDで反応し、短く鳴きます。",
        ["かわいい", "友達に見せたい", "部屋に置きたい", "とにかく動く"],
        ["集中できない", "人に見せる作品が欲しい", "勉強が続かない"],
        ["机上ペット", "触ると光るぬいぐるみ"],
        2,
        "2-4時間",
        3500,
        6000,
        "ESP32",
        ["esp32_devkit", "distance_sensor", "buzzer", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
        ["servo_sg90", "case_material", "oled_display"],
        ["距離センサー", "ブザー", "条件分岐", "GND共有", "GPIO"],
        ["距離センサーの電圧仕様", "ブザーの向き", "GPIO番号とコードの不一致", "GND未共有"],
        "距離センサー + LEDで近づいたら光る",
        "距離センサー + LED + ブザー + 紙箱外装",
        "サーボで体を揺らす、OLEDで表情を出す",
        ["暗くなると光る机上ペット", "温度で表情が変わるミニキャラ", "姿勢注意デバイス"],
        "small desk creature with sensor, LED, buzzer",
    ),
    "light_charm": MakerProject(
        "light_charm",
        "暗くなると光る小さなお守り",
        "部屋が暗くなるとLEDがふわっと光る、最初の作品にしやすい小型ライトです。",
        ["かわいい", "部屋に置きたい", "便利", "光るもの"],
        ["部屋が暗い", "忘れ物をする", "人に見せる作品が欲しい"],
        ["光るお守り", "音に反応するライト"],
        1,
        "1-2時間",
        1500,
        3000,
        "Arduino Uno",
        ["arduino_uno", "light_sensor", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
        ["case_material", "oled_display"],
        ["LED極性", "抵抗", "アナログ入力", "しきい値処理"],
        ["LEDの向き", "抵抗なしLED", "光センサーの分圧", "USBケーブルが充電専用"],
        "内蔵LEDまたは5mm LEDだけを光らせる",
        "光センサーで明暗を読み、暗い時だけLED点灯",
        "ケースに入れてお守り化、色LEDやフェード表現を追加",
        ["近づくと鳴く机上ペット", "水やり通知", "音に反応するライト"],
        "small glowing charm with light sensor",
    ),
    "temp_face": MakerProject(
        "temp_face",
        "温度で表情が変わるミニキャラ",
        "温度や湿度に合わせて画面の表情が変わる、見た目のある環境モニターです。",
        ["かわいい", "研究っぽい", "部屋に置きたい", "便利"],
        ["集中できない", "部屋が暗い", "人に見せる作品が欲しい"],
        ["温度で表情が変わるキャラ"],
        3,
        "4-7時間",
        5000,
        9000,
        "M5Stack",
        ["m5stickc", "dht_sensor", "usb_cable"],
        ["oled_display", "case_material"],
        ["温湿度センサー", "画面表示", "状態管理", "I2C/Grove"],
        ["ライブラリ導入", "センサー値がnanになる", "表示更新が速すぎる"],
        "M5StickCの画面に固定表情を表示",
        "温湿度センサーで表情を切り替える",
        "ログ保存、警告音、机上ケースを追加",
        ["水やり通知", "姿勢注意デバイス", "環境ログステーション"],
        "tiny face device that changes expression with temperature",
    ),
    "plant_ping": MakerProject(
        "plant_ping",
        "水やり通知",
        "植物の土が乾いたらLEDやブザーで知らせます。水回りの安全を学びながら作れます。",
        ["便利", "生活の困りごと", "研究っぽい"],
        ["植物を枯らす", "忘れ物をする", "勉強が続かない"],
        ["水やり通知"],
        2,
        "2-5時間",
        2500,
        5500,
        "ESP32",
        ["esp32_devkit", "soil_sensor", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
        ["buzzer", "case_material", "oled_display"],
        ["アナログ入力", "しきい値処理", "水回りの安全", "センサー校正"],
        ["水と電源の距離", "腐食しやすい抵抗式センサー購入", "閾値が環境で変わる"],
        "土壌センサー値をシリアル表示する",
        "乾いたらLEDで通知する",
        "ブザー通知、画面表示、過去値ログを追加",
        ["温度で表情が変わるミニキャラ", "自動水やり前の安全学習", "環境ログステーション"],
        "plant moisture notifier with safe USB power",
    ),
    "posture_guard": MakerProject(
        "posture_guard",
        "姿勢注意デバイス",
        "机に向かう距離や角度をゆるく見守り、近すぎる状態が続くと音や光で知らせます。",
        ["便利", "研究っぽい", "とにかく動く"],
        ["集中できない", "勉強が続かない", "運動不足"],
        ["姿勢注意デバイス"],
        3,
        "4-6時間",
        4200,
        8000,
        "ESP32",
        ["esp32_devkit", "distance_sensor", "buzzer", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
        ["oled_display", "case_material"],
        ["距離センサー", "時間判定", "状態管理", "millis"],
        ["センサーの向き", "delayで反応が重くなる", "ブザーが鳴りっぱなしになる"],
        "近すぎたらLEDを点灯する",
        "近すぎる状態が10秒続いたら短く鳴らす",
        "画面表示、休憩タイマー、ログ保存を追加",
        ["集中見守りキャラ", "机上ペット", "生活ログデバイス"],
        "desk posture watcher with distance sensor",
    ),
}


ENTRY_SUGGESTIONS = {
    "mood": ["かわいい", "便利", "友達に見せたい", "部屋に置きたい", "研究っぽい", "とにかく動く"],
    "problem": ["起きられない", "集中できない", "忘れ物をする", "植物を枯らす", "部屋が暗い", "勉強が続かない"],
    "budget": ["0円で試したい", "1,000円以内", "3,000円以内", "5,000円以内", "10,000円以内"],
    "inventory": ["Arduino", "ESP32", "M5Stack", "LED", "センサー", "サーボモーター", "ブレッドボード"],
    "example": ["光るお守り", "机上ペット", "水やり通知", "姿勢注意デバイス", "温度で表情が変わるキャラ"],
}


def yen(value: int) -> str:
    return f"{value:,}円"


def parse_budget(payload: dict[str, Any]) -> int | None:
    raw_values = [
        payload.get("budget"),
        payload.get("text"),
        payload.get("selectedOption"),
        payload.get("problem"),
        payload.get("mood"),
    ]
    for raw in raw_values:
        if not raw:
            continue
        text = str(raw).replace(",", "")
        if "0円" in text:
            return 0
        match = re.search(r"(\d+)\s*万", text)
        if match:
            return int(match.group(1)) * 10000
        match = re.search(r"(\d{3,6})\s*円?", text)
        if match:
            return int(match.group(1))
    return payload.get("budgetLimit") if isinstance(payload.get("budgetLimit"), int) else None


def tokenize_inventory(payload: dict[str, Any]) -> list[str]:
    raw = " ".join(
        str(payload.get(key, ""))
        for key in ["inventory", "text", "selectedOption"]
    ).lower()
    owned: list[str] = []
    mapping = {
        "esp32": "esp32_devkit",
        "arduino": "arduino_uno",
        "m5": "m5stickc",
        "led": "led_5mm",
        "抵抗": "resistor_220",
        "ブレッド": "breadboard",
        "ジャンパ": "jumper_wires",
        "usb": "usb_cable",
        "光": "light_sensor",
        "距離": "distance_sensor",
        "センサー": "light_sensor",
        "ブザー": "buzzer",
        "サーボ": "servo_sg90",
    }
    for needle, component_id in mapping.items():
        if needle in raw and component_id not in owned:
            owned.append(component_id)
    return owned


def project_cost_after_inventory(project: MakerProject, owned: list[str]) -> tuple[int, int]:
    min_total = 0
    max_total = 0
    for component_id in project.components:
        if component_id in owned:
            continue
        component = COMPONENTS[component_id]
        min_total += component.price_min * component.quantity
        max_total += component.price_max * component.quantity
    return min_total, max_total


def score_project(project: MakerProject, payload: dict[str, Any]) -> float:
    text = " ".join(str(payload.get(key, "")) for key in ["entry", "mood", "problem", "selectedOption", "text"]).lower()
    budget = parse_budget(payload)
    owned = tokenize_inventory(payload)
    score = 0.0

    for tag in project.mood_tags + project.problem_tags + project.example_tags:
        if tag.lower() in text:
            score += 4.0

    if "かわいい" in text and "かわいい" in project.mood_tags:
        score += 5.0
    if "便利" in text and "便利" in project.mood_tags:
        score += 4.0
    if "植物" in text and project.id == "plant_ping":
        score += 8.0
    if "暗" in text and project.id == "light_charm":
        score += 7.0
    if "姿勢" in text and project.id == "posture_guard":
        score += 8.0
    if "温度" in text and project.id == "temp_face":
        score += 8.0
    if "机上" in text and project.id == "desk_pet":
        score += 8.0

    reuse = len(set(project.components) & set(owned))
    score += reuse * 1.6

    if budget is not None:
        min_after, max_after = project_cost_after_inventory(project, owned)
        if budget == 0:
            score += 4.0 if reuse >= 3 else -5.0
        elif min_after <= budget:
            score += 3.0
            if max_after <= budget * 1.25:
                score += 2.0
        else:
            score -= min((min_after - budget) / 1000, 5)

    if payload.get("solderingAllowed") is False:
        score += 1.0
    score += max(0, 4 - project.difficulty) * 0.6
    return score


def project_summary(project: MakerProject, payload: dict[str, Any], label: str) -> dict[str, Any]:
    owned = tokenize_inventory(payload)
    cost_min, cost_max = project_cost_after_inventory(project, owned)
    if not owned:
        cost_min, cost_max = project.cost_min, project.cost_max
    return {
        "id": project.id,
        "label": label,
        "title": project.title,
        "summary": project.summary,
        "budgetRange": f"{yen(cost_min)}〜{yen(cost_max)}",
        "difficulty": "★" * project.difficulty + "☆" * (5 - project.difficulty),
        "difficultyValue": project.difficulty,
        "estimatedTime": project.estimated_time,
        "requiredParts": [COMPONENTS[c].name for c in project.components[:5]],
        "niceToHaveParts": [COMPONENTS[c].name for c in project.optional_components],
        "laterParts": [COMPONENTS[c].name for c in project.optional_components[1:]],
        "skills": project.skills,
        "failurePoints": project.failure_points,
        "minimal": project.minimal,
        "standard": project.standard,
        "extension": project.extension,
        "nextProjects": project.next_projects,
        "completionProbability": round(max(0.52, 0.9 - project.difficulty * 0.08 + len(set(project.components) & set(owned)) * 0.03), 2),
        "visualHint": project.visual_hint,
    }


def recommend_projects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = sorted(PROJECTS.values(), key=lambda p: score_project(p, payload), reverse=True)
    budget = parse_budget(payload)
    owned = tokenize_inventory(payload)

    best = ranked[0]
    cheap_candidates = sorted(PROJECTS.values(), key=lambda p: project_cost_after_inventory(p, owned)[0] if owned else p.cost_min)
    cheap = next((p for p in cheap_candidates if p.id != best.id), cheap_candidates[0])
    challenge = next((p for p in ranked if p.id not in {best.id, cheap.id} and p.difficulty >= best.difficulty), ranked[-1])

    if budget is not None and budget <= 1200 and "light_charm" in PROJECTS:
        best = PROJECTS["light_charm"]
        cheap = best
        challenge = PROJECTS["desk_pet"]

    unique: list[MakerProject] = []
    for candidate in [best, cheap, challenge, *ranked]:
        if candidate.id not in [p.id for p in unique]:
            unique.append(candidate)
        if len(unique) == 3:
            break

    labels = ["一番おすすめ", "安い案", "少し挑戦案"]
    return [project_summary(project, payload, labels[index]) for index, project in enumerate(unique)]


def make_intent(payload: dict[str, Any]) -> dict[str, Any]:
    budget = parse_budget(payload)
    text = " ".join(str(payload.get(key, "")) for key in ["mood", "problem", "selectedOption", "text"])
    novelty = 0.55
    if any(word in text for word in ["研究", "見せたい", "作品性"]):
        novelty = 0.78
    return {
        "intentEmbeddingPreview": [
            round(0.22 + len(text) % 7 * 0.07, 2),
            round(0.41 + (budget or 3000) % 5000 / 10000, 2),
            novelty,
        ],
        "desiredExperience": ["完成体験", "見せられる作品", "部品選びの不安軽減"],
        "functionPreference": infer_function_preferences(text),
        "aestheticPreference": infer_aesthetic_preferences(text),
        "difficultyTolerance": "low" if budget is not None and budget <= 3000 else "medium",
        "budgetSensitivity": "high" if budget is not None and budget <= 3000 else "medium",
        "noveltyPreference": novelty,
        "ownedComponents": [COMPONENTS[c].name for c in tokenize_inventory(payload)],
    }


def infer_function_preferences(text: str) -> list[str]:
    prefs = []
    if "暗" in text or "光" in text:
        prefs.append("光で反応する")
    if "植物" in text or "水" in text:
        prefs.append("生活の状態を検知する")
    if "集中" in text or "姿勢" in text:
        prefs.append("習慣を見守る")
    if "かわいい" in text or "机上" in text:
        prefs.append("キャラクターとして反応する")
    return prefs or ["センサー入力に応じて出力が変わる"]


def infer_aesthetic_preferences(text: str) -> list[str]:
    prefs = []
    if "かわいい" in text:
        prefs.append("小さくて表情がある")
    if "研究" in text:
        prefs.append("計測器っぽい")
    if "部屋" in text:
        prefs.append("机や棚に置ける")
    return prefs or ["最小構成から外装で育てる"]


def build_project_graph(project: MakerProject) -> dict[str, Any]:
    nodes = []
    for component_id in project.components + project.optional_components:
        component = COMPONENTS[component_id]
        node_type = {
            "センサー": "input",
            "マイコン": "processing",
            "出力部品": "output",
            "電源": "power",
            "配線部品": "connection",
            "外装素材": "enclosure",
        }.get(component.category, "support")
        nodes.append({"id": component.id, "type": node_type, "name": component.name, "risk": component.risk_level})
    for skill in project.skills:
        nodes.append({"id": f"skill_{slug(skill)}", "type": "skill", "name": skill, "risk": "low"})
    nodes.append({"id": "cost_range", "type": "cost", "name": f"{yen(project.cost_min)}〜{yen(project.cost_max)}", "risk": "low"})
    nodes.append({"id": "safety_baseline", "type": "safety", "name": "USB給電・AC100V除外", "risk": "low"})

    edges = []
    board = next((component_id for component_id in project.components if COMPONENTS[component_id].category == "マイコン"), project.components[0])
    for component_id in project.components:
        if component_id == board:
            continue
        relation = "connects_to"
        if COMPONENTS[component_id].category == "電源":
            relation = "powers"
        elif COMPONENTS[component_id].category == "出力部品":
            relation = "controls"
        elif COMPONENTS[component_id].category == "センサー":
            relation = "reads"
        edges.append({"from": component_id, "to": board, "relation": relation})
    for skill in project.skills:
        edges.append({"from": board, "to": f"skill_{slug(skill)}", "relation": "requires_skill"})
    edges.append({"from": board, "to": "safety_baseline", "relation": "constrained_by"})
    return {
        "project": project.title,
        "nodes": nodes,
        "edges": edges,
        "validationStatus": "valid_with_beginner_safety_rules",
        "riskScore": round(project.difficulty * 0.16 + sum(1 for c in project.optional_components if COMPONENTS[c].risk_level == "medium") * 0.08, 2),
    }


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.lower()).strip("_") or str(abs(hash(value)))


def estimate_bom(project: MakerProject, payload: dict[str, Any]) -> dict[str, Any]:
    owned = set(tokenize_inventory(payload))

    minimal_ids = project.components
    standard_ids = project.components + [c for c in project.optional_components if c in {"case_material", "buzzer"}]
    extended_ids = project.components + project.optional_components

    tiers = [
        make_bom_tier("最小構成", "とりあえず動く", minimal_ids, owned),
        make_bom_tier("標準構成", "初心者が作品として見せられる", standard_ids, owned),
        make_bom_tier("拡張構成", "外装・安定性・追加表現まで含める", extended_ids, owned),
    ]
    return {
        "projectId": project.id,
        "projectTitle": project.title,
        "tiers": tiers,
        "purchasePriority": make_purchase_priority(minimal_ids, owned),
        "doNotBuy": [
            "家庭用AC100Vを直接扱うリレーモジュール",
            "保護回路なしリチウムイオン電池セル",
            "GPIOへ直結する大電流モーター",
            "仕様電圧が不明な格安センサーセット",
        ],
        "safetyNotes": [
            "初心者向けの初期案ではUSB 5V給電を基本にする。",
            "モーターを追加する場合は電源容量を別枠で確認する。",
            "水回り作品では電源と濡れる部分を離す。",
        ],
    }


def make_bom_tier(name: str, purpose: str, component_ids: list[str], owned: set[str]) -> dict[str, Any]:
    seen: list[str] = []
    for component_id in component_ids:
        if component_id not in seen:
            seen.append(component_id)

    items = []
    total_min = 0
    total_max = 0
    for component_id in seen:
        component = COMPONENTS[component_id]
        has_item = component_id in owned
        if not has_item:
            total_min += component.price_min * component.quantity
            total_max += component.price_max * component.quantity
        items.append(component_to_bom_item(component, has_item))
    return {
        "name": name,
        "purpose": purpose,
        "totalMin": total_min,
        "totalMax": total_max,
        "range": f"{yen(total_min)}〜{yen(total_max)}",
        "items": items,
        "confidence": 0.76 if name == "最小構成" else 0.68,
    }


def component_to_bom_item(component: Component, owned: bool) -> dict[str, Any]:
    return {
        "componentId": component.id,
        "name": component.name,
        "category": component.category,
        "purpose": component.purpose,
        "required": component.required,
        "priceRange": f"{yen(component.price_min)}〜{yen(component.price_max)}",
        "quantity": component.quantity,
        "owned": owned,
        "alternatives": component.alternatives,
        "beginnerScore": component.beginner_score,
        "solderingRequired": component.soldering,
        "notes": component.notes,
        "voltage": component.voltage,
        "current": component.current,
        "compatibleBoards": component.compatible_boards,
        "interface": component.interface,
        "searchKeyword": component.keyword,
        "canUseHouseholdItem": component.id == "case_material",
        "buyLater": not component.required,
    }


def make_purchase_priority(component_ids: list[str], owned: set[str]) -> list[dict[str, Any]]:
    priorities = []
    for index, component_id in enumerate(component_ids, start=1):
        if component_id in owned:
            continue
        component = COMPONENTS[component_id]
        priorities.append({
            "rank": index,
            "name": component.name,
            "reason": "これがないと最小構成が動かない" if component.required else "後で作品性を上げる",
        })
    return priorities


def generate_circuit(project: MakerProject) -> dict[str, Any]:
    if project.id == "light_charm":
        table = [
            ["LED", "D5", "アノード", "220Ω抵抗を直列"],
            ["LED", "GND", "カソード", "極性注意"],
            ["光センサー", "5V", "VCC", "モジュールの場合"],
            ["光センサー", "GND", "GND", "共通GND必須"],
            ["光センサー", "A0", "AO", "コードと一致"],
        ]
        pins = {"led": "D5", "lightSensor": "A0"}
        board = "Arduino Uno"
        current = "80-160mA程度"
    elif project.id == "temp_face":
        table = [
            ["DHTセンサー", "3.3V", "VCC", "M5の3.3Vへ"],
            ["DHTセンサー", "GND", "GND", "共通GND必須"],
            ["DHTセンサー", "G26", "DATA", "コードと一致"],
            ["内蔵画面", "内部接続", "LCD", "M5ライブラリで制御"],
        ]
        pins = {"dht": "G26", "display": "built-in"}
        board = "M5StickC Plus2"
        current = "120-350mA程度"
    elif project.id == "plant_ping":
        table = [
            ["土壌水分センサー", "3.3V", "VCC", "水回りなのでUSB電源のみ"],
            ["土壌水分センサー", "GND", "GND", "共通GND必須"],
            ["土壌水分センサー", "GPIO 34", "AO", "入力専用ピン"],
            ["LED", "GPIO 5", "アノード", "220Ω抵抗を直列"],
            ["LED", "GND", "カソード", "極性注意"],
        ]
        pins = {"soilSensor": "GPIO 34", "led": "GPIO 5"}
        board = "ESP32"
        current = "100-260mA程度"
    else:
        table = [
            ["LED", "GPIO 5", "アノード", "220Ω抵抗を直列"],
            ["LED", "GND", "カソード", "極性注意"],
            ["距離センサー", "3.3V", "VCC", "5V専用なら直結しない"],
            ["距離センサー", "GND", "GND", "共通GND必須"],
            ["距離センサー", "GPIO 21", "SIG", "コードと一致"],
            ["ブザー", "GPIO 18", "+", "PWM出力"],
            ["ブザー", "GND", "-", "極性がある場合は注意"],
        ]
        pins = {"led": "GPIO 5", "distanceSensor": "GPIO 21", "buzzer": "GPIO 18"}
        board = project.board
        current = "120-320mA程度"

    validation = validate_circuit(project, table)
    return {
        "projectId": project.id,
        "board": board,
        "connectionTable": [
            {"part": row[0], "controllerSide": row[1], "partSide": row[2], "note": row[3]}
            for row in table
        ],
        "pinMapping": pins,
        "powerMapping": {
            "main": "USB 5V",
            "logic": "board regulated 3.3V/5V",
            "ground": "すべてのGNDを共通にする",
        },
        "currentEstimate": current,
        "wiringSteps": [
            "USBを抜いた状態でブレッドボードへ部品を置く。",
            "GNDを先に共通化する。",
            "電源線を接続し、センサーのVCC仕様を確認する。",
            "信号線をピン割り当て通りに挿す。",
            "LEDは抵抗を直列に入れてから接続する。",
            "書き込み前に接続表とコードのピン番号を照合する。",
        ],
        "breadboardView": make_breadboard_view(project.id),
        "validation": validation,
    }


def validate_circuit(project: MakerProject, table: list[list[str]]) -> dict[str, Any]:
    warnings: list[str] = []
    blockers: list[str] = []
    table_text = " ".join(" ".join(row) for row in table)

    if "GND" not in table_text:
        blockers.append("GND共有が確認できません。すべての部品のGNDを共通にしてください。")
    if "LED" in table_text and "220Ω" not in table_text:
        blockers.append("LEDに直列抵抗がありません。GPIO直結は避けてください。")
    if any(c == "servo_sg90" for c in project.optional_components):
        warnings.append("サーボ追加時はマイコン直結ではなく、電源容量を別枠で確認してください。")
    if project.id == "plant_ping":
        warnings.append("水回り作品のため、濡れる部分とUSB電源を離して固定してください。")
    if project.board == "ESP32" and "5V専用" in table_text:
        warnings.append("ESP32は3.3Vロジックです。5V出力のEcho/SIGを直接入れないでください。")

    return {
        "status": "blocked" if blockers else "warning" if warnings else "safe",
        "blockers": blockers,
        "warnings": warnings,
        "checks": [
            {"name": "電圧不一致", "passed": not any("5V専用" in row[3] for row in table), "severity": "warning"},
            {"name": "GND共有", "passed": "GND" in table_text, "severity": "blocker"},
            {"name": "GPIOピン競合", "passed": len({row[1] for row in table if "GPIO" in row[1] or row[1].startswith("D")}) == len([row for row in table if "GPIO" in row[1] or row[1].startswith("D")]), "severity": "blocker"},
            {"name": "LED抵抗", "passed": "LED" not in table_text or "220Ω" in table_text, "severity": "blocker"},
            {"name": "AC100V除外", "passed": True, "severity": "blocker"},
            {"name": "モーター直結禁止", "passed": "DCモーター" not in table_text, "severity": "blocker"},
        ],
    }


def make_breadboard_view(project_id: str) -> dict[str, Any]:
    layouts = {
        "light_charm": [
            {"type": "board", "label": "Arduino", "x": 8, "y": 38},
            {"type": "sensor", "label": "Light", "x": 58, "y": 28},
            {"type": "led", "label": "LED", "x": 76, "y": 57},
            {"type": "wire", "from": "A0", "to": "Light AO", "color": "#2f7dd1"},
            {"type": "wire", "from": "D5", "to": "LED +", "color": "#f2b705"},
        ],
        "plant_ping": [
            {"type": "board", "label": "ESP32", "x": 8, "y": 36},
            {"type": "sensor", "label": "Soil", "x": 60, "y": 25},
            {"type": "led", "label": "LED", "x": 78, "y": 58},
            {"type": "wire", "from": "G34", "to": "Soil AO", "color": "#2f7dd1"},
            {"type": "wire", "from": "G5", "to": "LED +", "color": "#f2b705"},
        ],
    }
    return {
        "layout": layouts.get(project_id, [
            {"type": "board", "label": "ESP32", "x": 8, "y": 36},
            {"type": "sensor", "label": "Distance", "x": 58, "y": 24},
            {"type": "led", "label": "LED", "x": 74, "y": 58},
            {"type": "buzzer", "label": "Buzzer", "x": 58, "y": 70},
            {"type": "wire", "from": "G21", "to": "SIG", "color": "#2f7dd1"},
            {"type": "wire", "from": "G18", "to": "Buzzer +", "color": "#d1453b"},
            {"type": "wire", "from": "G5", "to": "LED +", "color": "#f2b705"},
        ]),
        "notes": ["赤は電源、青は信号、黒はGNDとして確認する。", "ジャンパ線の色は実物と違っても接続先が一致していればよい。"],
    }


def generate_firmware(project: MakerProject) -> dict[str, Any]:
    if project.id == "light_charm":
        code = """// 暗くなると光る小さなお守り - Arduino Uno
const int LED_PIN = 5;
const int LIGHT_PIN = A0;
const int DARK_THRESHOLD = 450;

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("light charm start");
}

void loop() {
  int lightValue = analogRead(LIGHT_PIN);
  Serial.print("light=");
  Serial.println(lightValue);

  if (lightValue < DARK_THRESHOLD) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  delay(200);
}
"""
        libraries = ["標準ライブラリのみ"]
        board_settings = ["Arduino Uno", "Portを選択", "シリアルモニタ 9600bps"]
        pins = {"LED_PIN": "D5", "LIGHT_PIN": "A0"}
    elif project.id == "temp_face":
        code = """// 温度で表情が変わるミニキャラ - M5StickC Plus2
#include <M5StickCPlus2.h>
#include <DHT.h>

const int DHT_PIN = 26;
const int DHT_TYPE = DHT11;
DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  auto cfg = M5.config();
  StickCP2.begin(cfg);
  Serial.begin(115200);
  dht.begin();
  StickCP2.Display.setTextSize(3);
  Serial.println("temp face start");
}

void loop() {
  float temp = dht.readTemperature();
  Serial.print("temp=");
  Serial.println(temp);

  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(20, 40);
  if (isnan(temp)) {
    StickCP2.Display.print("?");
  } else if (temp >= 28) {
    StickCP2.Display.print(">_<");
  } else if (temp <= 18) {
    StickCP2.Display.print("-_-");
  } else {
    StickCP2.Display.print("^_^");
  }
  delay(1000);
}
"""
        libraries = ["M5StickCPlus2", "DHT sensor library"]
        board_settings = ["M5StickC Plus2", "シリアルモニタ 115200bps"]
        pins = {"DHT_PIN": "G26"}
    elif project.id == "plant_ping":
        code = """// 水やり通知 - ESP32
const int SOIL_PIN = 34;
const int LED_PIN = 5;
const int DRY_THRESHOLD = 2600;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("plant ping start");
}

void loop() {
  int soilValue = analogRead(SOIL_PIN);
  Serial.print("soil=");
  Serial.println(soilValue);

  if (soilValue > DRY_THRESHOLD) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  delay(500);
}
"""
        libraries = ["標準ライブラリのみ"]
        board_settings = ["ESP32 Dev Module", "Upload Speed 921600または115200", "シリアルモニタ 115200bps"]
        pins = {"SOIL_PIN": "GPIO 34", "LED_PIN": "GPIO 5"}
    else:
        code = """// 近づくと鳴く机上ペット / 姿勢注意デバイス - ESP32
const int LED_PIN = 5;
const int SENSOR_PIN = 21;
const int BUZZER_PIN = 18;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(SENSOR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.println("desk pet start");
}

void loop() {
  int detected = digitalRead(SENSOR_PIN);
  Serial.print("sensor=");
  Serial.println(detected);

  if (detected == HIGH) {
    digitalWrite(LED_PIN, HIGH);
    tone(BUZZER_PIN, 1200, 80);
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  delay(150);
}
"""
        libraries = ["標準ライブラリのみ", "ToF距離センサーを使う場合はAdafruit_VL53L0X"]
        board_settings = ["ESP32 Dev Module", "Portを選択", "シリアルモニタ 115200bps"]
        pins = {"LED_PIN": "GPIO 5", "SENSOR_PIN": "GPIO 21", "BUZZER_PIN": "GPIO 18"}

    return {
        "projectId": project.id,
        "board": project.board,
        "language": "Arduino C++",
        "code": code,
        "libraries": libraries,
        "installSteps": [
            "Arduino IDEを開く。",
            "ボードマネージャで対象ボードを選ぶ。",
            "必要ライブラリをライブラリマネージャから入れる。",
            "接続表のピン番号とコードの定数を照合する。",
            "シリアルモニタを開いて値が変わるか確認する。",
        ],
        "boardSettings": board_settings,
        "pinMapping": pins,
        "debugPrints": ["起動時メッセージ", "センサー値", "状態判定の元になる値"],
        "testSteps": [
            "まずUSBだけで書き込めるか確認する。",
            "シリアルモニタにセンサー値が出るか確認する。",
            "LEDだけ、ブザーだけの順に出力を分けて確認する。",
        ],
        "modificationPoints": [
            "しきい値を自分の部屋に合わせる。",
            "delay版で動いたらmillis版へ置き換える。",
            "外装に合わせて反応音や点灯パターンを変える。",
        ],
        "compileStatus": "template_ready",
    }


def diagnose_debug(payload: dict[str, Any]) -> dict[str, Any]:
    symptom = str(payload.get("symptom") or payload.get("text") or "何が悪いか分からない")
    project = get_project(payload.get("projectId"))

    if "書き込" in symptom or "COM" in symptom or "ポート" in symptom:
        causes = [
            ("USBケーブルが充電専用", 0.36),
            ("ボードまたはポート選択が違う", 0.31),
            ("ドライバ未導入", 0.22),
        ]
        steps = ["USBケーブルをデータ通信対応のものに替える。", "Arduino IDEのボードとポートを確認する。", "別のUSBポートで試す。"]
        question = "Arduino IDEのポート一覧に、抜き差しで増えるポートはありますか？"
    elif "光" in symptom or "LED" in symptom:
        causes = [
            ("LEDの極性が逆", 0.34),
            ("抵抗やジャンパ線の列がずれている", 0.29),
            ("コードのピン番号と配線が違う", 0.25),
        ]
        steps = ["LEDの長い足がGPIO側か確認する。", "220Ω抵抗がLEDと直列か確認する。", "コードのLED_PINと接続表のピンを合わせる。"]
        question = "LEDの長い足は、抵抗を経由してGPIO側につながっていますか？"
    elif "センサー" in symptom or "値" in symptom:
        causes = [
            ("VCC/GND/SIGの順番違い", 0.33),
            ("センサーの電圧仕様がボードと合っていない", 0.27),
            ("入力ピンがコードと違う", 0.24),
        ]
        steps = ["センサー基板の印字を見てVCC/GND/SIGを確認する。", "シリアルモニタに固定値だけ出ていないか見る。", "入力ピンを接続表とコードでそろえる。"]
        question = "シリアルモニタの値は、手を近づけたり明るさを変えたりして変化しますか？"
    elif "ブザー" in symptom or "鳴" in symptom:
        causes = [
            ("ブザーの+/-が逆", 0.31),
            ("GPIOがPWM/toneに合っていない", 0.22),
            ("音が短すぎて聞こえない", 0.18),
        ]
        steps = ["ブザー単体テストコードで鳴るか確認する。", "+端子をGPIO側、-端子をGND側にする。", "toneの時間を300msへ伸ばす。"]
        question = "ブザー単体テストでも音は鳴りませんか？"
    else:
        causes = [
            ("GND未共有", 0.28),
            ("USB給電またはケーブル問題", 0.25),
            ("コードのピン番号と実配線の不一致", 0.22),
        ]
        steps = ["まずUSBを抜いて、GND線が全部つながっているか見る。", "シリアルモニタに起動メッセージが出るか確認する。", "出力部品を1つだけ残して最小構成に戻す。"]
        question = "シリアルモニタに起動メッセージは出ていますか？"

    return {
        "projectId": project.id,
        "symptom": symptom,
        "causeRanking": [{"cause": cause, "probability": probability} for cause, probability in causes],
        "checkSteps": steps,
        "fix": steps[0],
        "dangerLevel": "medium" if project.id == "plant_ping" else "low",
        "nextQuestion": question,
        "logToSave": {
            "failureTags": [cause for cause, _ in causes[:2]],
            "skillUpdates": suggest_skill_updates(symptom),
            "usedForNextRecommendation": True,
        },
    }


def suggest_skill_updates(symptom: str) -> dict[str, float]:
    updates = {"debugging": 0.03}
    if "LED" in symptom or "光" in symptom:
        updates.update({"led_polarity": 0.04, "resistor_usage": 0.03})
    if "センサー" in symptom or "値" in symptom:
        updates.update({"analog_input": 0.04, "gpio": 0.02})
    if "GND" in symptom or "何" in symptom:
        updates.update({"gnd_common": 0.04})
    return updates


def skill_state() -> dict[str, Any]:
    vector = {
        "led_polarity": 0.55,
        "resistor_usage": 0.48,
        "gnd_common": 0.41,
        "gpio": 0.58,
        "digital_input": 0.36,
        "analog_input": 0.32,
        "pwm": 0.24,
        "i2c": 0.18,
        "motor_control": 0.12,
        "power_design": 0.2,
        "library_install": 0.35,
        "compile_error_reading": 0.31,
        "threshold_processing": 0.39,
        "state_management": 0.26,
        "enclosure_design": 0.22,
        "project_decomposition": 0.62,
    }
    return {
        "skillVector": vector,
        "confidence": 0.63,
        "weaknessTags": ["GND共有", "電源設計", "I2C"],
        "nextConcept": "センサー値のしきい値処理",
        "avoidConcepts": ["AC100V", "高電流モーター", "LiPo自作充放電", "複雑な割り込み"],
        "recommendations": ["暗くなると光る小さなお守り", "近づくと鳴く机上ペット"],
        "learned": ["LEDをGPIOで制御する", "シリアルモニタで値を見る"],
    }


def next_recommendations(payload: dict[str, Any]) -> dict[str, Any]:
    selected = get_project(payload.get("projectId"))
    return {
        "baseProject": selected.title,
        "recommendations": [
            {
                "type": "すぐ作れる次作品",
                "title": selected.next_projects[0],
                "reason": "今回の部品を再利用し、入力または出力を1つだけ増やせるため。",
                "estimatedCost": "1,200〜2,500円",
                "completionProbability": 0.82,
                "learningGain": 0.54,
                "novelty": 0.42,
            },
            {
                "type": "少しレベルアップする作品",
                "title": selected.next_projects[1] if len(selected.next_projects) > 1 else "温度で表情が変わるミニキャラ",
                "reason": "センサー値をしきい値だけでなく状態として扱う練習になるため。",
                "estimatedCost": "3,500〜6,500円",
                "completionProbability": 0.69,
                "learningGain": 0.68,
                "novelty": 0.56,
            },
            {
                "type": "憧れに近づく作品",
                "title": selected.next_projects[2] if len(selected.next_projects) > 2 else "生活ログステーション",
                "reason": "作品グラフを拡張し、表示・外装・ログ保存までつなげられるため。",
                "estimatedCost": "6,000〜12,000円",
                "completionProbability": 0.55,
                "learningGain": 0.81,
                "novelty": 0.74,
            },
        ],
    }


def tutorial_free_response(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or payload.get("nextQuestion") or "")
    answer = str(payload.get("answer") or payload.get("text") or "")
    current_stage = str(payload.get("currentStage") or payload.get("stage") or "orient")
    project = get_project(payload.get("projectId"))
    fallback_kind = classify_free_answer(answer)
    classifier_payload = {
        **payload,
        "projectId": project.id,
        "projectTitle": project.title,
        "question": question,
        "answer": answer,
        "currentStage": current_stage,
    }
    answer_classifier = tutorial_answer_classifier_inference(classifier_payload)
    kind = str(answer_classifier.get("kind") or fallback_kind) if answer_classifier.get("available") else fallback_kind
    next_stage = next_stage_from_free_answer(question, answer, current_stage, kind)
    reply = compose_tutorial_reply(project, question, answer, current_stage, next_stage, kind, payload)

    neural_payload = {
        **payload,
        "projectId": project.id,
        "projectTitle": project.title,
        "question": question,
        "answer": answer,
        "currentStage": current_stage,
        "fallbackStage": next_stage,
        "interpretedKind": kind,
        "candidateCount": payload.get("candidateCount") or 4,
    }
    neural = tutorial_response_inference(neural_payload)
    if neural.get("available"):
        neural_stage = merge_tutorial_stage(
            current_stage=current_stage,
            model_stage=str(neural.get("nextStage") or ""),
            fallback_stage=next_stage,
            kind=kind,
        )
        presentation_kind = presentation_kind_for_free_answer(kind, str(neural.get("kind") or ""), reply["kind"])
        use_neural_text = should_use_neural_tutorial_text(kind, str(neural.get("kind") or ""))
        preview_payload = {
            **payload,
            "projectId": project.id,
            "currentStage": neural_stage,
            "text": " ".join([
                str(payload.get("text") or ""),
                f"question: {question}",
                f"answer: {answer}",
                f"neuralStyle: {presentation_kind}",
            ]).strip(),
        }
        tutorial = tutorial_agent_inference(preview_payload)
        result = {
            "available": True,
            "model": neural.get("model") or "TutorialResponseTransformer",
            "modelDir": neural.get("modelDir"),
            "mode": "deep_learning_free_response",
            "projectId": project.id,
            "question": question,
            "answer": answer,
            "interpreted": kind,
            "interpretedBy": "neural_answer_classifier" if answer_classifier.get("available") else "fallback",
            "kind": presentation_kind,
            "title": (neural.get("title") if use_neural_text else "") or reply["title"],
            "body": (neural.get("body") if use_neural_text else "") or reply["body"],
            "nextInstruction": (neural.get("nextInstruction") if use_neural_text else "") or reply["nextInstruction"],
            "nextStage": neural_stage,
            "alignmentScore": neural.get("alignmentScore"),
            "candidateCount": neural.get("candidateCount"),
            "suggestedChips": reply["suggestedChips"],
            "confidence": max(confidence_for_free_answer(kind, question, answer), 0.72),
            "tutorialPreview": tutorial if tutorial.get("available") else None,
            "answerClassifier": answer_classifier if answer_classifier.get("available") else None,
            "neural": {
                "device": neural.get("device"),
                "metrics": neural.get("metrics"),
                "generated": neural.get("generated"),
                "alignmentScore": neural.get("alignmentScore"),
                "candidateCount": neural.get("candidateCount"),
                "candidates": neural.get("candidates"),
            },
        }
        log_tutorial_response_event(payload, project, question, answer, current_stage, kind, result, neural)
        return result

    resolved_stage = merge_tutorial_stage(
        current_stage=current_stage,
        model_stage="",
        fallback_stage=next_stage,
        kind=kind,
    )
    preview_payload = {
        **payload,
        "projectId": project.id,
        "currentStage": resolved_stage,
        "text": " ".join([
            str(payload.get("text") or ""),
            f"question: {question}",
            f"answer: {answer}",
            f"interpretedAnswer: {kind}",
        ]).strip(),
    }
    tutorial = tutorial_agent_inference(preview_payload)
    result = {
        "available": True,
        "model": "TutorialFreeResponseSynthesizer",
        "mode": "hybrid_neural_free_answer",
        "projectId": project.id,
        "question": question,
        "answer": answer,
        "interpreted": kind,
        "interpretedBy": "neural_answer_classifier" if answer_classifier.get("available") else "fallback",
        "kind": reply["kind"],
        "title": reply["title"],
        "body": reply["body"],
        "nextInstruction": reply["nextInstruction"],
        "nextStage": resolved_stage,
        "suggestedChips": reply["suggestedChips"],
        "confidence": confidence_for_free_answer(kind, question, answer),
        "tutorialPreview": tutorial if tutorial.get("available") else None,
        "answerClassifier": answer_classifier if answer_classifier.get("available") else None,
    }
    log_tutorial_response_event(payload, project, question, answer, current_stage, kind, result, neural)
    return result


def merge_tutorial_stage(current_stage: str, model_stage: str, fallback_stage: str, kind: str) -> str:
    current_stage = current_stage if current_stage in STAGE_ORDER else "orient"
    model_stage = model_stage if model_stage in STAGE_ORDER else ""
    fallback_stage = fallback_stage if fallback_stage in STAGE_ORDER else current_stage

    if kind in {"negative", "problem_report", "photo_request"}:
        return fallback_stage
    if kind in {"unknown", "empty"}:
        if fallback_stage == "debug_triage":
            return fallback_stage
        return current_stage if current_stage in {"minimal_circuit", "firmware_upload", "observe_serial"} else fallback_stage

    ranked = [stage for stage in [current_stage, model_stage, fallback_stage] if stage in PROGRESS_STAGE_ORDER]
    if not ranked:
        return fallback_stage or current_stage
    return max(ranked, key=lambda stage: PROGRESS_STAGE_ORDER.index(stage))


def presentation_kind_for_free_answer(interpreted_kind: str, neural_kind: str, fallback_kind: str) -> str:
    if interpreted_kind in {"confirmed", "inventory_report", "board_report"}:
        return "good"
    if interpreted_kind in {"negative", "unknown", "problem_report", "photo_request"}:
        return "warn"
    if neural_kind in {"good", "warn", "info"}:
        return neural_kind
    return fallback_kind if fallback_kind in {"good", "warn", "info"} else "info"


def should_use_neural_tutorial_text(interpreted_kind: str, neural_kind: str) -> bool:
    if interpreted_kind in {"confirmed", "inventory_report", "board_report"} and neural_kind == "warn":
        return False
    if interpreted_kind in {"negative", "unknown", "problem_report", "photo_request"} and neural_kind == "good":
        return False
    return neural_kind in {"good", "warn", "info"}


def log_tutorial_response_event(
    payload: dict[str, Any],
    project: MakerProject,
    question: str,
    answer: str,
    current_stage: str,
    interpreted_kind: str,
    result: dict[str, Any],
    neural: dict[str, Any] | None = None,
) -> None:
    if os.environ.get("DISABLE_TUTORIAL_LOGS") == "1":
        return
    try:
        event = {
            "schema": "tutorial_response_event_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sessionId": clip_log_text(payload.get("sessionId") or "anonymous", 120),
            "clientVersion": clip_log_text(payload.get("clientVersion") or "", 120),
            "projectId": project.id,
            "projectTitle": project.title,
            "currentStage": current_stage,
            "nextStage": result.get("nextStage"),
            "interpreted": interpreted_kind,
            "question": clip_log_text(question),
            "answer": clip_log_text(answer),
            "sourcePayload": {
                "projectId": project.id,
                "projectTitle": project.title,
                "currentStage": current_stage,
                "question": clip_log_text(question),
                "answer": clip_log_text(answer),
                "inventory": clip_log_text(payload.get("inventory") or ""),
                "budget": clip_log_text(payload.get("budget") or ""),
                "symptom": clip_log_text(payload.get("symptom") or "none"),
                "skill": clip_log_text(payload.get("skill") or ""),
                "previous": clip_log_text(payload.get("previous") or ""),
            },
            "response": {
                "kind": result.get("kind") or "info",
                "title": clip_log_text(result.get("title") or ""),
                "body": clip_log_text(result.get("body") or ""),
                "nextInstruction": clip_log_text(result.get("nextInstruction") or ""),
                "nextStage": result.get("nextStage") or current_stage,
                "confidence": result.get("confidence"),
                "mode": result.get("mode"),
                "model": result.get("model"),
                "modelDir": result.get("modelDir"),
            },
            "answerClassifier": compact_answer_classifier(result.get("answerClassifier")),
            "target": tutorial_response_target(result, current_stage),
            "neuralAvailable": bool((neural or {}).get("available")),
            "neuralGenerated": clip_log_text((neural or {}).get("generated") or "", 2000),
        }
        TUTORIAL_EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TUTORIAL_EVENT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - logging must never block UX
        print(f"tutorial log skipped: {exc}", flush=True)


def compact_answer_classifier(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value.get("available"):
        return None
    ranked = value.get("ranked") if isinstance(value.get("ranked"), list) else []
    return {
        "model": clip_log_text(value.get("model") or "", 160),
        "modelDir": clip_log_text(value.get("modelDir") or "", 240),
        "device": clip_log_text(value.get("device") or "", 80),
        "kind": clip_log_text(value.get("kind") or "", 80),
        "confidence": value.get("confidence"),
        "ranked": [
            {
                "kind": clip_log_text(item.get("kind") or "", 80),
                "probability": item.get("probability"),
            }
            for item in ranked[:4]
            if isinstance(item, dict)
        ],
        "metrics": value.get("metrics") if isinstance(value.get("metrics"), dict) else {},
    }


def tutorial_response_target(result: dict[str, Any], fallback_stage: str) -> str:
    style = str(result.get("kind") or "info")
    if style not in {"good", "warn", "info"}:
        style = "info"
    stage = str(result.get("nextStage") or fallback_stage or "parts_check")
    if stage not in STAGE_ORDER:
        stage = fallback_stage if fallback_stage in STAGE_ORDER else "parts_check"
    return "\n".join([
        f"style: {style}",
        f"title: {result.get('title') or ''}",
        f"body: {result.get('body') or ''}",
        f"next: {result.get('nextInstruction') or ''}",
        f"stage: {stage}",
    ])


def clip_log_text(value: Any, limit: int = MAX_TUTORIAL_LOG_TEXT) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def tutorial_log_stats() -> dict[str, Any]:
    response_stats = jsonl_file_stats(TUTORIAL_EVENT_LOG)
    feedback_stats = jsonl_file_stats(TUTORIAL_FEEDBACK_LOG)
    return {
        "enabled": os.environ.get("DISABLE_TUTORIAL_LOGS") != "1",
        "path": str(TUTORIAL_EVENT_LOG),
        "events": response_stats["events"],
        "bytes": response_stats["bytes"],
        "updatedAt": response_stats.get("updatedAt"),
        "feedbackPath": str(TUTORIAL_FEEDBACK_LOG),
        "feedbackEvents": feedback_stats["events"],
        "feedbackBytes": feedback_stats["bytes"],
        "feedbackUpdatedAt": feedback_stats.get("updatedAt"),
        "feedbackRatings": feedback_rating_counts(TUTORIAL_FEEDBACK_LOG),
        "exportEnabled": bool(os.environ.get("TUTORIAL_LOG_EXPORT_TOKEN")),
    }


def jsonl_file_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"events": 0, "bytes": 0}
    events = 0
    with path.open(encoding="utf-8") as handle:
        for _line in handle:
            events += 1
    return {
        "events": events,
        "bytes": path.stat().st_size,
        "updatedAt": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def feedback_rating_counts(path: Path) -> dict[str, int]:
    counts = {"up": 0, "down": 0, "fix": 0, "note": 0}
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rating = str(json.loads(line).get("rating") or "note")
            except json.JSONDecodeError:
                rating = "note"
            counts[rating if rating in counts else "note"] += 1
    return counts


def tutorial_log_export(params: dict[str, list[str]]) -> dict[str, Any]:
    expected_token = os.environ.get("TUTORIAL_LOG_EXPORT_TOKEN")
    provided_token = (params.get("token") or [""])[0]
    if not expected_token:
        return {"ok": False, "reason": "log export disabled; set TUTORIAL_LOG_EXPORT_TOKEN"}
    if provided_token != expected_token:
        return {"ok": False, "reason": "invalid token"}
    limit = safe_int((params.get("limit") or ["500"])[0], 500, 1, 5000)
    return {
        "ok": True,
        "stats": tutorial_log_stats(),
        "responseEvents": read_jsonl_tail(TUTORIAL_EVENT_LOG, limit),
        "feedbackEvents": read_jsonl_tail(TUTORIAL_FEEDBACK_LOG, limit),
    }


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) > limit:
                rows = rows[-limit:]
    return rows


def safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def tutorial_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    rating = str(payload.get("rating") or "").strip().lower()
    if rating not in {"up", "down", "fix"}:
        rating = "note"
    event = {
        "schema": "tutorial_feedback_event_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sessionId": clip_log_text(payload.get("sessionId") or "anonymous", 120),
        "clientVersion": clip_log_text(payload.get("clientVersion") or "", 120),
        "rating": rating,
        "reason": clip_log_text(payload.get("reason") or "", 800),
        "correction": clip_log_text(payload.get("correction") or "", 1600),
        "projectId": clip_log_text(payload.get("projectId") or "", 120),
        "currentStage": clip_log_text(payload.get("currentStage") or "", 120),
        "nextStage": clip_log_text(payload.get("nextStage") or "", 120),
        "question": clip_log_text(payload.get("question") or ""),
        "answer": clip_log_text(payload.get("answer") or ""),
        "reply": {
            "interpreted": clip_log_text((payload.get("reply") or {}).get("interpreted") if isinstance(payload.get("reply"), dict) else "", 80),
            "interpretedBy": clip_log_text((payload.get("reply") or {}).get("interpretedBy") if isinstance(payload.get("reply"), dict) else "", 120),
            "kind": clip_log_text((payload.get("reply") or {}).get("kind") if isinstance(payload.get("reply"), dict) else "", 60),
            "title": clip_log_text((payload.get("reply") or {}).get("title") if isinstance(payload.get("reply"), dict) else ""),
            "body": clip_log_text((payload.get("reply") or {}).get("body") if isinstance(payload.get("reply"), dict) else ""),
            "nextInstruction": clip_log_text((payload.get("reply") or {}).get("nextInstruction") if isinstance(payload.get("reply"), dict) else ""),
            "model": clip_log_text((payload.get("reply") or {}).get("model") if isinstance(payload.get("reply"), dict) else "", 180),
            "mode": clip_log_text((payload.get("reply") or {}).get("mode") if isinstance(payload.get("reply"), dict) else "", 180),
            "modelDir": clip_log_text((payload.get("reply") or {}).get("modelDir") if isinstance(payload.get("reply"), dict) else "", 240),
        },
        "answerClassifier": compact_answer_classifier((payload.get("reply") or {}).get("answerClassifier") if isinstance(payload.get("reply"), dict) else None),
    }
    TUTORIAL_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TUTORIAL_FEEDBACK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"ok": True, "rating": rating, "stats": tutorial_log_stats()}


def classify_free_answer(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        return "empty"
    if any(token in normalized for token in ["写真", "画像", "撮影", "カメラ", "photo", "camera", "pic"]):
        return "photo_request"
    if any(token in normalized for token in ["分からない", "わからない", "不明", "知らない", "自信ない", "自信がない", "不安", "unknown", "?" , "？"]):
        return "unknown"
    if any(token in normalized for token in ["エラー", "error", "exception", "failed", "失敗", "できない", "出ない", "動かない", "光らない", "鳴らない", "映らない", "止まる", "落ちる"]):
        return "problem_report"
    if any(token in normalized for token in ["直した", "つなぎ直", "できた", "完了", "完成", "終わった", "ok", "yes", "はい", "同じ", "つながってる", "つながっています", "つないだ", "入っています", "通っています", "大丈夫", "合っています", "見えています", "出ています"]):
        return "confirmed"
    if any(token in normalized for token in ["いいえ", "no", "違う", "まだ", "無い", "ない", "入ってない", "つながってない", "通ってない", "見えない", "出てない"]):
        return "negative"
    if looks_like_inventory(answer):
        return "inventory_report"
    if any(token in normalized for token in ["arduino", "esp32", "m5", "pico", "microbit", "micro:bit"]):
        return "board_report"
    if len(answer) >= 18:
        return "descriptive"
    return "short_answer"


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", "", answer).lower()


def looks_like_inventory(answer: str) -> bool:
    lowered = answer.lower()
    known_parts = ["esp32", "arduino", "m5", "pico", "led", "抵抗", "ブレッドボード", "ジャンパ", "usb", "センサー", "ブザー", "サーボ", "oled", "モーター"]
    if sum(1 for part in known_parts if part in lowered or part in answer) >= 2:
        return True
    return "," in answer or "、" in answer


def next_stage_from_free_answer(question: str, answer: str, current_stage: str, kind: str) -> str:
    lowered_question = question.lower()
    if kind == "confirmed" and current_stage == "extension":
        return "completion_log"
    if kind == "confirmed" and current_stage == "enclosure":
        return "extension"
    if is_inventory_question_text(question):
        return "minimal_circuit"
    if is_gnd_question_text(question):
        return "firmware_upload" if kind == "confirmed" else "minimal_circuit"
    if is_board_question_text(question):
        return "firmware_upload" if current_stage in {"firmware_upload", "observe_serial"} else "minimal_circuit"
    if "LED" in question or "led" in lowered_question or "抵抗" in question or "GPIO" in question:
        return "firmware_upload" if kind == "confirmed" else "minimal_circuit"
    if "ポート" in question or "書き込" in question or "アップロード" in question or "COM" in question or "usb" in lowered_question:
        return "observe_serial" if kind == "confirmed" else "debug_triage"
    if "シリアル" in question or "起動メッセージ" in question or "数値" in question or "値" in question or "serial" in lowered_question:
        return "standard_build" if kind == "confirmed" else "debug_triage"
    if "外装" in question or "ケース" in question or "固定" in question or "見た目" in question:
        return "extension" if kind == "confirmed" else "enclosure"
    if "拡張" in question or "ブザー" in question or "サーボ" in question or "音" in question or "ログ" in question or "完成" in question:
        return "completion_log" if kind == "confirmed" else "extension"
    if kind in {"problem_report", "photo_request"}:
        return "debug_triage"
    if kind == "confirmed":
        order = ["orient", "parts_check", "minimal_circuit", "firmware_upload", "observe_serial", "standard_build", "enclosure", "extension", "completion_log"]
        if current_stage in order and order.index(current_stage) < len(order) - 1:
            return order[order.index(current_stage) + 1]
    if kind == "inventory_report":
        return "minimal_circuit"
    if kind == "board_report":
        return "minimal_circuit"
    return current_stage if current_stage in {"debug_triage", "minimal_circuit", "firmware_upload", "observe_serial"} else "parts_check"


def compose_tutorial_reply(project: MakerProject, question: str, answer: str, current_stage: str, next_stage: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    chips = ["できた", "分からない", "写真で確認したい"]
    tone = "good" if kind in {"confirmed", "inventory_report", "board_report"} else "warn" if kind in {"negative", "unknown", "problem_report", "photo_request"} else "info"

    if is_inventory_question_text(question):
        if kind == "unknown":
            return reply("warn", "部品が分からなくても進めます", "スターター構成として扱います。必要部品込みで見積もり、まずは最小回路から始めます。", "次はLED・抵抗・GNDだけの最小確認へ進みます。", ["スターターで進む", "写真で確認したい", "部品を後で直す"])
        return reply("good", "部品情報を受け取りました", "書いてくれた部品を所持品として扱います。不足分はBOM側で分けて出します。", "次はUSBを抜いた状態で、最小回路のGND共有を確認します。", ["GNDを確認する", "部品を追加する", "分からない"])

    if is_gnd_question_text(question):
        if kind == "confirmed":
            return reply("good", "GND共有はクリアです", "LED・センサー・ボードのGNDが同じ基準になっているので、次は確認コードを書き込む段階へ進めます。", "ボードとポートを確認して、シリアル出力つきのコードで動作を見ます。", ["コードを書き込む", "ピン番号を確認", "写真で見たい"])
        if kind == "photo_request":
            return reply("warn", "写真チェックに回しましょう", "真上から、ボードのGNDピンとブレッドボードのGND列が見えるように撮ると判断しやすいです。", "撮れたら配線写真チェックに進みます。今はUSBを抜いてください。", ["写真を撮る", "GNDをつなぎ直した", "分からない"])
        return reply("warn", "GNDを同じ列につなぎ直します", "USBを抜いてから、LEDのマイナス側、センサーのGND、ボードのGNDを同じGND列へつなぎます。ここが違うと全部動かないことがあります。", "つなぎ直せたら「つなぎ直した」と答えてください。", ["つなぎ直した", "写真で確認したい", "分からない"])

    if is_board_question_text(question):
        return reply("good", "ボード情報を受け取りました", "このボードに合わせてピン番号と確認コードを組み立てます。", "次は最小回路とコードのピン番号を合わせます。", ["ESP32で進む", "Arduinoで進む", "分からない"])

    if "LED" in question or "led" in question.lower():
        if kind == "confirmed":
            return reply("good", "LEDの向きは大丈夫そうです", "長い足が抵抗を通ってGPIO側、短い足がGND側なら次へ進めます。", "次は抵抗とコードのピン番号を合わせます。", chips)
        return reply("warn", "LEDの向きを直してから進みます", "LEDは向きがあります。長い足をGPIO側、短い足をGND側にします。USBを抜いてから直してください。", "直せたら、もう一度LEDだけで点灯確認します。", ["直した", "写真で確認したい", "分からない"])

    if "抵抗" in question:
        if kind == "confirmed":
            return reply("good", "抵抗は入っています", "LEDをGPIOへ直結していないので、まず安全な形です。", "次はコードのGPIO番号と配線先を一致させます。", chips)
        return reply("warn", "抵抗を直列に入れます", "LEDとGPIOの間に220Ω前後の抵抗を入れてください。抵抗なしの直結は避けます。", "抵抗を足せたら、再度LEDだけで確認します。", ["抵抗を入れた", "抵抗がない", "分からない"])

    if "シリアル" in question or "起動メッセージ" in question or "値" in question:
        if kind == "confirmed":
            return reply("good", "ログが見えています", "起動メッセージや値が見えているなら、コードは最低限動いています。", "次は手を近づける、暗くするなど、入力に応じて値が変わるか確認します。", ["値が変わる", "値が変わらない", "次へ"])
        return reply("warn", "ログを見る準備をします", "シリアルモニタを開き、速度をSerial.beginの値に合わせます。何も出ない場合はポート、ボード、USBケーブルを順に確認します。", "まず起動メッセージが出るかだけ見ます。", ["ログが出た", "何も出ない", "エラーが出た"])

    if kind == "problem_report":
        return reply("warn", "動かない状態として受け取りました", "原因を広げず、物理接続、電源、コードの順に一つずつ潰します。", "まずUSBを抜いて、GND共有とピン番号だけ確認します。", ["GNDを見た", "エラー文がある", "写真で確認したい"])
    if kind == "photo_request":
        return reply("warn", "写真で確認する流れにします", "真上から、ボード名、GND、LED、抵抗、ジャンパ線の刺さる列が見えるように撮ると判断できます。", "写真をアップしたら、怪しい箇所を一つずつ見ます。", ["写真を撮る", "先に文章で続ける", "分からない"])
    if kind == "confirmed":
        return reply("good", "確認できました", "その回答を受け取って、次の短い作業へ進めます。", "次のカードで、やることを一つだけ出します。", chips)
    if kind == "unknown":
        return reply("warn", "分からない前提で進めます", "ここで止めず、初心者向けの確認順に戻します。テスター前提にはしません。", "USBを抜く、GND、LED向き、抵抗、ピン番号の順に一つだけ見ます。", ["最小確認する", "写真で確認したい", "部品から見直す"])
    return reply(tone, "回答を受け取りました", f"「{answer[:80]}」として受け取りました。次の作業に反映します。", "次のカードで、確認する点を一つに絞ります。", chips)


def reply(kind: str, title: str, body: str, next_instruction: str, chips: list[str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "title": title,
        "body": body,
        "nextInstruction": next_instruction,
        "suggestedChips": chips,
    }


def confidence_for_free_answer(kind: str, question: str, answer: str) -> float:
    if kind in {"confirmed", "negative", "inventory_report", "board_report", "problem_report", "photo_request"}:
        return 0.84
    if kind == "unknown":
        return 0.72
    if len(answer) >= 18:
        return 0.66
    return 0.52


def is_inventory_question_text(question: str) -> bool:
    return any(token in question for token in ["部品", "手元", "持っている", "所持", "一行"]) or "inventory" in question.lower()


def is_board_question_text(question: str) -> bool:
    return any(token in question for token in ["ボード", "マイコン", "Arduino", "ESP32", "M5Stack", "Pico"])


def is_gnd_question_text(question: str) -> bool:
    return any(token in question for token in ["GND", "gnd", "グランド", "マイナス列", "GND列", "ground"])


def wiring_check(payload: dict[str, Any]) -> dict[str, Any]:
    project = get_project(payload.get("projectId"))
    return {
        "projectId": project.id,
        "recognizedParts": [COMPONENTS[c].name for c in project.components[:5]],
        "estimatedWiringGraph": build_project_graph(project),
        "differences": [
            {"target": "GND", "status": "needs_confirmation", "message": "黒または青の線が全て同じGND列に入っているか確認してください。"},
            {"target": "LED resistor", "status": "needs_confirmation", "message": "LEDの足とGPIOの間に抵抗が直列で入っている必要があります。"},
        ],
        "confidence": 0.42,
        "retakeGuidance": [
            "真上から撮る",
            "マイコンのピン番号が読める距離にする",
            "ブレッドボード全体とUSB接続部を同じ写真に入れる",
        ],
        "dangerLevel": "medium" if project.id == "plant_ping" else "low",
        "note": "このプロトタイプでは画像認識モデルの代わりに接続表ベースの確認項目を返します。",
    }


def get_project(project_id: Any) -> MakerProject:
    if isinstance(project_id, str) and project_id in PROJECTS:
        return PROJECTS[project_id]
    return PROJECTS["desk_pet"]


def project_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    project = get_project(payload.get("projectId") or payload.get("id"))
    return {
        "project": project_summary(project, payload, "選択中"),
        "projectGraph": build_project_graph(project),
        "bom": estimate_bom(project, payload),
        "circuit": generate_circuit(project),
        "firmware": generate_firmware(project),
        "safety": safety_report(project),
        "next": next_recommendations({"projectId": project.id}),
    }


def agent_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the production-facing maker agent pipeline in one request."""

    inventory_match = inventory_matcher_inference(payload)
    recommended = recommend_projects(payload)
    project_id = first_project_id(inventory_match) or (recommended[0]["id"] if recommended else "desk_pet")
    project = get_project(project_id)
    enriched_payload = {
        **payload,
        "projectId": project.id,
        "text": " ".join([
            str(payload.get("text") or ""),
            f"projectId {project.id}",
            f"boardClass {payload.get('board') or payload.get('boardClass') or board_class_for_project(project)}",
            f"pinProfile {pin_profile_for_board(payload.get('board') or payload.get('boardClass') or board_class_for_project(project))}",
            f"firmwareVariant {firmware_variant_for_project(project.id)}",
        ]).strip(),
    }

    intent = transformer_intent({**enriched_payload, "useTransformer": True})
    tutorial = tutorial_agent_inference(enriched_payload)
    neural = neural_agent_inference(enriched_payload)
    neural_bom = neural_bom_inference(enriched_payload)
    bom = estimate_bom(project, enriched_payload)
    circuit = generate_circuit(project)
    circuit_validation = circuit_validator_inference({**enriched_payload, "circuitGraph": circuit})
    skill = skillrec_inference(enriched_payload)
    next_projects = skill if skill.get("available") else next_recommendations({"projectId": project.id})

    firmware = {}
    safety = {}
    debug = {}
    if neural.get("available"):
        firmware = dict(neural.get("firmware", {}))
        safety = dict(neural.get("safety", {}))
        debug = dict(neural.get("debug", {}))

    return {
        "available": True,
        "service": "Lチカのつづき",
        "pipeline": "makergraph_gpu_agent_v1",
        "selectedProject": project_summary(project, enriched_payload, "一番おすすめ"),
        "recommendations": recommended[:3],
        "intent": intent,
        "inventoryMatch": inventory_match,
        "tutorial": tutorial,
        "projectGraph": build_project_graph(project),
        "bom": bom,
        "neuralBom": neural_bom,
        "circuit": circuit,
        "circuitValidation": circuit_validation,
        "firmware": firmware,
        "safety": safety,
        "debug": debug,
        "skillAndNext": next_projects,
        "deployment": {
            "gpuRequired": True,
            "edge": "Cloudflare Worker/Pages",
            "gpuBackend": "RTX 5060 CUDA backend or GPU server",
            "healthPath": "/api/health",
        },
    }


def first_project_id(result: dict[str, Any]) -> str | None:
    recommendations = result.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        value = recommendations[0].get("projectId") or recommendations[0].get("id")
        if isinstance(value, str) and value in PROJECTS:
            return value
    return None


def board_class_for_project(project: MakerProject) -> str:
    if project.board == "Arduino Uno":
        return "arduino_uno"
    if project.board == "M5Stack":
        return "m5stack"
    if project.board == "Raspberry Pi Pico":
        return "pico"
    return "esp32"


def pin_profile_for_board(board: Any) -> str:
    return {
        "arduino_uno": "arduino_default",
        "Arduino Uno": "arduino_default",
        "esp32": "esp32_default",
        "ESP32": "esp32_default",
        "m5stack": "m5stack_grove",
        "M5Stack": "m5stack_grove",
        "pico": "pico_default",
        "Raspberry Pi Pico": "pico_default",
    }.get(str(board), "esp32_default")


def firmware_variant_for_project(project_id: str) -> str:
    return {
        "light_charm": "light_charm_debug",
        "desk_pet": "desk_pet_buzzer",
        "plant_ping": "plant_ping_buzzer",
        "posture_guard": "posture_guard_buzzer",
        "temp_face": "temp_face_serial",
    }.get(project_id, "desk_pet_buzzer")


def safety_report(project: MakerProject) -> dict[str, Any]:
    warnings = [
        "AC100Vを直接扱う構成は初心者向け生成から除外しています。",
        "USBを抜いた状態で配線を変更してください。",
        "LEDには必ず抵抗を直列に入れてください。",
    ]
    if project.id == "plant_ping":
        warnings.append("水回りではUSB電源・マイコン・濡れる土を物理的に離してください。")
    if "servo_sg90" in project.optional_components:
        warnings.append("サーボ追加時は電源容量を別枠で見積もってください。")
    return {
        "riskLevel": "medium" if project.difficulty >= 3 or project.id == "plant_ping" else "low",
        "warnings": warnings,
        "blockedSuggestions": [
            "家庭用AC100Vのリレー制御",
            "保護回路なしLiPo充放電",
            "マイコンGPIOへのモーター直結",
            "医療・乳幼児安全に関わる自動制御",
        ],
    }


class MakerGraphHandler(BaseHTTPRequestHandler):
    server_version = "MakerGraphAI/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "service": "Lチカのつづき", "api": "MakerGraph AI prototype", "runtime": runtime_health()})
            return
        if parsed.path == "/api/tutorial/log-stats":
            self.send_json(tutorial_log_stats())
            return
        if parsed.path == "/api/tutorial/export-logs":
            self.send_json(tutorial_log_export(parse_qs(parsed.query)))
            return
        if parsed.path == "/api/skills/me":
            self.send_json(skill_state())
            return
        if parsed.path == "/api/recommendations/next":
            params = parse_qs(parsed.query)
            self.send_json(next_recommendations({"projectId": (params.get("projectId") or ["desk_pet"])[0]}))
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        payload = self.read_json()
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/discover/intent":
            transformer = transformer_intent(payload) if payload.get("useTransformer") else {"available": False, "reason": "set useTransformer=true to call checkpoint inference"}
            self.send_json({
                "intent": make_intent(payload),
                "recommendations": recommend_projects(payload),
                "transformer": transformer,
                "entrySuggestions": ENTRY_SUGGESTIONS,
            })
        elif route == "/api/agent/run":
            self.send_json(agent_run(payload))
        elif route == "/api/ai/intent/transformer":
            self.send_json(transformer_intent(payload))
        elif route == "/api/ai/project-graph/transformer":
            self.send_json(transformer_project_graph(payload))
        elif route == "/api/ai/wirechecknet":
            self.send_json(wirechecknet_inference(payload))
        elif route == "/api/ai/bom/estimator":
            self.send_json(neural_bom_inference(payload))
        elif route == "/api/ai/skillrec":
            self.send_json(skillrec_inference(payload))
        elif route == "/api/ai/neural-agents":
            self.send_json(neural_agent_inference(payload))
        elif route == "/api/ai/circuit-validator":
            self.send_json(circuit_validator_inference(payload))
        elif route == "/api/ai/inventory-matcher":
            self.send_json(inventory_matcher_inference(payload))
        elif route == "/api/ai/tutorial-agent":
            self.send_json(tutorial_agent_inference(payload))
        elif route == "/api/tutorial/respond":
            self.send_json(tutorial_free_response(payload))
        elif route == "/api/tutorial/feedback":
            self.send_json(tutorial_feedback(payload))
        elif route in {"/api/projects/generate", "/api/projects/refine"}:
            self.send_json(project_bundle(payload))
        elif route in {"/api/tutorial/start", "/api/tutorial/next", "/api/tutorial/answer"}:
            self.send_json(tutorial_agent_inference(payload))
        elif route == "/api/discover/from-inventory":
            neural = inventory_matcher_inference(payload)
            self.send_json({"inventoryMatch": neural, "ruleBased": recommend_projects(payload)})
        elif route == "/api/bom/estimate":
            self.send_json(estimate_bom(get_project(payload.get("projectId")), payload))
        elif route == "/api/circuits/generate":
            self.send_json(generate_circuit(get_project(payload.get("projectId"))))
        elif route == "/api/circuits/validate":
            project = get_project(payload.get("projectId"))
            circuit = generate_circuit(project)
            result = circuit_validator_inference({**payload, "projectId": project.id, "circuitGraph": circuit})
            if not result.get("available"):
                self.send_json(result)
            else:
                self.send_json({"circuitId": circuit["id"], **result})
        elif route == "/api/safety/validate":
            result = neural_agent_inference(payload)
            if not result.get("available"):
                self.send_json(result)
            else:
                self.send_json({"available": True, "model": result.get("model"), **result.get("safety", {})})
        elif route == "/api/firmware/generate":
            result = neural_agent_inference(payload)
            if not result.get("available"):
                self.send_json(result)
            else:
                firmware = dict(result.get("firmware", {}))
                firmware["safety"] = result.get("safety", {})
                firmware["model"] = result.get("model")
                self.send_json(firmware)
        elif route in {"/api/firmware/validate", "/api/firmware/compile-check"}:
            result = neural_agent_inference(payload)
            if not result.get("available"):
                self.send_json(result)
            else:
                self.send_json({
                    "available": True,
                    "model": result.get("model"),
                    "safety": result.get("safety", {}),
                    "firmwareClass": result.get("firmware", {}),
                    "compileStatus": "neural_static_pass",
                })
        elif route in {"/api/debug/start", "/api/debug/diagnose", "/api/debug/answer"}:
            result = neural_agent_inference(payload)
            if not result.get("available"):
                self.send_json(result)
            else:
                debug = dict(result.get("debug", {}))
                debug["safety"] = result.get("safety", {})
                debug["model"] = result.get("model")
                self.send_json(debug)
        elif route in {"/api/wiring/check", "/api/wiring/compare", "/api/wiring/retake-guidance"}:
            self.send_json(wiring_check(payload))
        elif route == "/api/skills/update":
            self.send_json({"updated": True, "skillState": skill_state(), "received": payload})
        elif route == "/api/recommendations/neural-next":
            self.send_json(skillrec_inference(payload))
        elif route == "/api/recommendations/feedback":
            self.send_json({"saved": True, "message": "feedback logged for ranking model"})
        else:
            self.send_error(404, "Unknown API route")

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def serve_static(self, requested_path: str) -> None:
        path = requested_path
        if path in {"", "/"}:
            path = "/index.html"
        target = (FRONTEND_ROOT / path.lstrip("/")).resolve()
        if FRONTEND_ROOT.resolve() not in target.parents and target != FRONTEND_ROOT.resolve():
            self.send_error(403)
            return
        if not target.exists() or not target.is_file():
            target = FRONTEND_ROOT / "index.html"
        content = target.read_bytes()
        mime_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix == ".js":
            mime_type = "application/javascript"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if mime_type.startswith("text") or mime_type in {"application/javascript", "image/svg+xml"} else mime_type)
        self.send_header("Content-Length", str(len(content)))
        if target.suffix == ".html":
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), MakerGraphHandler)
    print(f"Lチカのつづき prototype running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
