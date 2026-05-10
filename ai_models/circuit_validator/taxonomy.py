from __future__ import annotations

CIRCUIT_ISSUES = [
    "ok",
    "gnd_not_shared",
    "led_missing_resistor",
    "voltage_mismatch",
    "pin_conflict",
    "motor_direct_gpio",
    "over_current",
    "floating_input",
    "reversed_polarity",
]

RISK_CLASSES = ["low", "medium", "high", "blocked"]

REPAIR_ACTIONS = [
    "none",
    "connect_common_gnd",
    "add_led_resistor",
    "use_level_shifter_or_3v3_part",
    "move_to_free_gpio",
    "add_motor_driver",
    "separate_power_supply",
    "add_pullup_or_pulldown",
    "flip_polarity",
]

BOARD_CLASSES = ["arduino_uno", "esp32", "m5stack", "pico"]

