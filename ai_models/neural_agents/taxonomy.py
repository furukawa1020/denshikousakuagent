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
    "usb_port_or_driver": "USB cable, board selection, or driver is likely wrong.",
    "wrong_pin_mapping": "The code pin and wiring pin likely do not match.",
    "missing_gnd": "GND is likely not shared across the board and parts.",
    "led_polarity_or_resistor": "LED direction or current-limiting resistor is likely wrong.",
    "sensor_power_or_signal": "Sensor VCC, GND, signal pin, or threshold is likely wrong.",
    "library_missing": "A required library or board package is missing.",
    "brownout_power": "Power capacity is likely insufficient or unstable.",
    "buzzer_pin_or_polarity": "Buzzer polarity or output pin is likely wrong.",
    "unknown": "The evidence is not enough; ask for one more observation.",
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
