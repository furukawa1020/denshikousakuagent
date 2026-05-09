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


def index_or_default(values: list[str], value: str, default: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return values.index(default)

