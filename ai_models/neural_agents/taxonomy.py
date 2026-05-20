from __future__ import annotations

PROJECTS = [
    "light_charm",
    "desk_pet",
    "plant_ping",
    "posture_guard",
    "temp_face",
]

PROJECT_LABELS = {
    "light_charm": "Dark-reactive LED charm",
    "desk_pet": "Proximity desk pet",
    "plant_ping": "Plant watering notifier",
    "posture_guard": "Posture alert device",
    "temp_face": "Temperature face display",
}

DEBUG_CAUSES = [
    "usb_port_or_driver",
    "wrong_pin_mapping",
    "missing_gnd",
    "led_polarity_or_resistor",
    "sensor_power_or_signal",
    "library_missing",
    "brownout_power",
    "buzzer_pin_or_polarity",
    "unknown",
]

DEBUG_LABELS = {
    "usb_port_or_driver": "USBケーブル、ボード選択、ポート認識まわりが怪しいです。",
    "wrong_pin_mapping": "コードのピン番号と実際の配線先がずれている可能性があります。",
    "missing_gnd": "GNDが共通になっていない可能性があります。",
    "led_polarity_or_resistor": "LEDの向き、または直列抵抗まわりが怪しいです。",
    "sensor_power_or_signal": "センサーのVCC/GND/SIG、またはしきい値が怪しいです。",
    "library_missing": "必要なライブラリかボードパッケージが入っていない可能性があります。",
    "brownout_power": "電源容量不足、または瞬間的な電圧低下が起きている可能性があります。",
    "buzzer_pin_or_polarity": "ブザーの極性、またはtone()に使うピンが怪しいです。",
    "unknown": "情報が足りないので、まず一つ観察を増やします。",
}

SAFETY_LABELS = [
    "ac_mains",
    "high_voltage",
    "lipo_charge",
    "motor_direct",
    "missing_resistor_led",
    "gnd_not_shared",
    "voltage_mismatch",
    "water_power",
    "thermal_load",
    "ok_low_voltage",
]

RISK_CLASSES = ["low", "medium", "high", "blocked"]

FIRMWARE_CLASSES = PROJECTS

FIRMWARE_VARIANTS = [
    "light_charm_minimal",
    "light_charm_debug",
    "desk_pet_led",
    "desk_pet_buzzer",
    "plant_ping_led",
    "plant_ping_buzzer",
    "posture_guard_led",
    "posture_guard_buzzer",
    "temp_face_serial",
    "temp_face_i2c_ready",
]

BOARD_CLASSES = ["arduino_uno", "esp32", "m5stack", "pico"]

BOARD_LABELS = {
    "arduino_uno": "Arduino Uno",
    "esp32": "ESP32",
    "m5stack": "M5Stack",
    "pico": "Raspberry Pi Pico",
}

PIN_PROFILES = [
    "arduino_default",
    "esp32_default",
    "esp32_grove_safe",
    "m5stack_grove",
    "pico_default",
]

PIN_PROFILE_VALUES = {
    "arduino_default": {"led": 5, "buzzer": 6, "digital": 7, "analog": "A0"},
    "esp32_default": {"led": 5, "buzzer": 18, "digital": 21, "analog": "34"},
    "esp32_grove_safe": {"led": 25, "buzzer": 26, "digital": 32, "analog": "34"},
    "m5stack_grove": {"led": 2, "buzzer": 26, "digital": 36, "analog": "35"},
    "pico_default": {"led": 15, "buzzer": 14, "digital": 16, "analog": "26"},
}


def index_or_default(values: list[str], value: str, default: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return values.index(default)
