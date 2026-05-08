from __future__ import annotations


SKILLS = [
    "led_polarity",
    "resistor_usage",
    "gnd_common",
    "gpio",
    "digital_input",
    "analog_input",
    "pwm",
    "i2c",
    "uart",
    "spi",
    "motor_control",
    "power_design",
    "library_install",
    "compile_error_reading",
    "threshold_processing",
    "state_management",
    "enclosure_design",
    "project_decomposition",
    "debugging",
]


SKILL_LABELS = {
    "led_polarity": "LEDの極性",
    "resistor_usage": "抵抗の使い方",
    "gnd_common": "GND共有",
    "gpio": "GPIO",
    "digital_input": "デジタル入力",
    "analog_input": "アナログ入力",
    "pwm": "PWM",
    "i2c": "I2C",
    "uart": "UART",
    "spi": "SPI",
    "motor_control": "モーター制御",
    "power_design": "電源設計",
    "library_install": "ライブラリ導入",
    "compile_error_reading": "コンパイルエラー読解",
    "threshold_processing": "しきい値処理",
    "state_management": "状態管理",
    "enclosure_design": "外装設計",
    "project_decomposition": "作品分解",
    "debugging": "デバッグ",
}


EVENT_TYPES = [
    "project_started",
    "question",
    "hint_viewed",
    "compile_error",
    "wiring_error",
    "wiring_check",
    "debug_success",
    "code_edit",
    "completed",
    "gave_up",
]


PROJECTS = [
    "light_charm",
    "desk_pet",
    "plant_ping",
    "temp_face",
    "posture_guard",
]


PROJECT_LABELS = {
    "light_charm": "暗くなると光る小さなお守り",
    "desk_pet": "近づくと鳴く机上ペット",
    "plant_ping": "水やり通知",
    "temp_face": "温度で表情が変わるミニキャラ",
    "posture_guard": "姿勢注意デバイス",
}


PROJECT_SKILLS = {
    "light_charm": ["led_polarity", "resistor_usage", "gpio", "analog_input", "threshold_processing"],
    "desk_pet": ["gpio", "digital_input", "gnd_common", "debugging", "threshold_processing"],
    "plant_ping": ["analog_input", "threshold_processing", "power_design", "debugging"],
    "temp_face": ["library_install", "i2c", "state_management", "compile_error_reading"],
    "posture_guard": ["digital_input", "state_management", "threshold_processing", "debugging"],
}


PROJECT_DIFFICULTY = {
    "light_charm": 1,
    "desk_pet": 2,
    "plant_ping": 2,
    "temp_face": 3,
    "posture_guard": 3,
}


PROJECT_COST = {
    "light_charm": 2500,
    "desk_pet": 5000,
    "plant_ping": 4500,
    "temp_face": 8000,
    "posture_guard": 6500,
}


PROJECT_COMPONENTS = {
    "light_charm": {"arduino_uno", "light_sensor", "led_5mm", "resistor_220", "breadboard", "jumper_wires"},
    "desk_pet": {"esp32_devkit", "distance_sensor", "buzzer", "led_5mm", "resistor_220", "breadboard", "jumper_wires"},
    "plant_ping": {"esp32_devkit", "soil_sensor", "led_5mm", "resistor_220", "breadboard", "jumper_wires"},
    "temp_face": {"m5stickc", "dht_sensor", "oled_display"},
    "posture_guard": {"esp32_devkit", "distance_sensor", "buzzer", "led_5mm", "resistor_220", "breadboard"},
}


RECOMMENDATION_TYPES = ["すぐ作れる次作品", "少しレベルアップする作品", "憧れに近づく作品"]


def skill_index(skill: str) -> int:
    return SKILLS.index(skill)


def project_index(project: str) -> int:
    return PROJECTS.index(project)
