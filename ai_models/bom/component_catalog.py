from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BOMComponent:
    id: str
    name: str
    category: str
    price_min: int
    price_median: int
    price_max: int
    risk_level: str
    aliases: tuple[str, ...]


COMPONENTS: list[BOMComponent] = [
    BOMComponent("esp32_devkit", "ESP32開発ボード", "マイコン", 900, 1300, 1800, "low", ("ESP32", "esp32")),
    BOMComponent("arduino_uno", "Arduino Uno互換ボード", "マイコン", 1200, 1800, 2800, "low", ("Arduino Uno", "Arduino")),
    BOMComponent("m5stickc", "M5StickC Plus2", "マイコン", 3600, 4300, 5200, "medium", ("M5StickC", "M5Stack")),
    BOMComponent("led_5mm", "5mm LED", "出力部品", 20, 50, 80, "low", ("LED",)),
    BOMComponent("resistor_220", "220Ω抵抗", "配線部品", 5, 15, 30, "low", ("220Ω抵抗", "抵抗")),
    BOMComponent("breadboard", "ブレッドボード", "配線部品", 300, 500, 700, "low", ("ブレッドボード",)),
    BOMComponent("jumper_wires", "ジャンパ線セット", "配線部品", 250, 450, 650, "low", ("ジャンパ線", "ジャンパーワイヤー")),
    BOMComponent("usb_cable", "データ通信対応USBケーブル", "電源", 300, 700, 1200, "low", ("USBケーブル", "USB")),
    BOMComponent("light_sensor", "光センサー", "センサー", 120, 350, 600, "low", ("光センサー", "light_sensor")),
    BOMComponent("distance_sensor", "距離センサー", "センサー", 400, 900, 1500, "low", ("距離センサー", "distance_sensor")),
    BOMComponent("buzzer", "圧電ブザー", "出力部品", 100, 250, 500, "low", ("ブザー", "buzzer")),
    BOMComponent("servo_sg90", "SG90サーボモーター", "出力部品", 350, 650, 900, "medium", ("サーボモーター", "servo")),
    BOMComponent("dht_sensor", "温湿度センサー", "センサー", 300, 600, 900, "low", ("温湿度センサー", "DHT")),
    BOMComponent("oled_display", "0.96インチOLED", "出力部品", 600, 1000, 1500, "low", ("OLED", "画面")),
    BOMComponent("soil_sensor", "静電容量式土壌水分センサー", "センサー", 450, 750, 1000, "medium", ("土壌水分センサー", "soil_sensor")),
    BOMComponent("case_material", "紙箱・ケース・固定具", "外装素材", 300, 900, 1500, "low", ("ケース", "外装", "固定具")),
]


COMPONENT_BY_ID = {component.id: component for component in COMPONENTS}
COMPONENT_IDS = [component.id for component in COMPONENTS]


PROJECT_COMPONENTS = {
    "desk_pet": ["esp32_devkit", "distance_sensor", "buzzer", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
    "light_charm": ["arduino_uno", "light_sensor", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
    "temp_face": ["m5stickc", "dht_sensor", "usb_cable"],
    "plant_ping": ["esp32_devkit", "soil_sensor", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
    "posture_guard": ["esp32_devkit", "distance_sensor", "buzzer", "led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"],
}


PROJECT_OPTIONALS = {
    "desk_pet": ["servo_sg90", "case_material", "oled_display"],
    "light_charm": ["case_material", "oled_display"],
    "temp_face": ["case_material", "oled_display", "buzzer"],
    "plant_ping": ["buzzer", "case_material", "oled_display"],
    "posture_guard": ["oled_display", "case_material"],
}


RISK_CLASSES = ["low", "medium", "water_caution", "power_caution"]


def component_cost_quantiles(component_ids: list[str]) -> tuple[int, int, int]:
    q10 = sum(COMPONENT_BY_ID[component_id].price_min for component_id in component_ids)
    q50 = sum(COMPONENT_BY_ID[component_id].price_median for component_id in component_ids)
    q90 = sum(COMPONENT_BY_ID[component_id].price_max for component_id in component_ids)
    return q10, q50, q90


def project_risk(project_id: str, component_ids: list[str]) -> str:
    if project_id == "plant_ping":
        return "water_caution"
    if any(component_id == "servo_sg90" for component_id in component_ids):
        return "power_caution"
    if any(COMPONENT_BY_ID[component_id].risk_level == "medium" for component_id in component_ids):
        return "medium"
    return "low"
