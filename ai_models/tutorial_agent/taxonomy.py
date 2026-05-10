from __future__ import annotations

PROJECT_IDS = [
    "light_charm",
    "desk_pet",
    "plant_ping",
    "posture_guard",
    "temp_face",
]

PROJECT_TITLES = {
    "light_charm": "暗くなると光る小さなお守り",
    "desk_pet": "近づくと鳴く机上ペット",
    "plant_ping": "水やり通知",
    "posture_guard": "姿勢注意デバイス",
    "temp_face": "温度で表情が変わるミニキャラ",
}

TUTORIAL_STAGES = [
    "orient",
    "parts_check",
    "minimal_circuit",
    "firmware_upload",
    "observe_serial",
    "debug_triage",
    "standard_build",
    "enclosure",
    "extension",
    "completion_log",
]

STAGE_LABELS = {
    "orient": "作るものを一つに絞る",
    "parts_check": "部品を机に並べて確認する",
    "minimal_circuit": "最小回路を組む",
    "firmware_upload": "動作確認コードを書き込む",
    "observe_serial": "シリアル出力で状態を見る",
    "debug_triage": "動かない原因を一つに絞る",
    "standard_build": "作品として見せられる構成にする",
    "enclosure": "固定と外装を整える",
    "extension": "拡張機能を一つ足す",
    "completion_log": "完成ログを保存して次作品へ進む",
}

ACTIONS = [
    "show_three_choices",
    "ask_single_question",
    "check_inventory",
    "guide_wiring",
    "generate_test_firmware",
    "request_serial_log",
    "request_photo",
    "diagnose_one_cause",
    "save_build_log",
    "unlock_next_project",
]

ACTION_LABELS = {
    "show_three_choices": "3案だけ出す",
    "ask_single_question": "質問を一つだけ出す",
    "check_inventory": "所持部品と不足部品を確認する",
    "guide_wiring": "配線を一手順ずつ進める",
    "generate_test_firmware": "確認用コードを出す",
    "request_serial_log": "シリアルログを見る",
    "request_photo": "配線写真を求める",
    "diagnose_one_cause": "最有力原因を一つ確認する",
    "save_build_log": "制作ログへ残す",
    "unlock_next_project": "次作品を解放する",
}

QUESTIONS = [
    "choose_mood",
    "confirm_budget",
    "confirm_inventory",
    "confirm_board",
    "check_led_polarity",
    "check_resistor",
    "check_gnd",
    "check_usb_port",
    "check_serial_value",
    "choose_next_extension",
]

QUESTION_TEXT = {
    "choose_mood": "今日は、かわいい・便利・人に見せたい、どれに一番近いですか？",
    "confirm_budget": "今回の上限予算は何円くらいにしますか？",
    "confirm_inventory": "手元にある部品を、分かる範囲で一行で書けますか？",
    "confirm_board": "使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？",
    "check_led_polarity": "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
    "check_resistor": "LEDとGPIOの間に220Ω前後の抵抗は入っていますか？",
    "check_gnd": "センサー、LED、ボードのGNDは同じGND列につながっていますか？",
    "check_usb_port": "Arduino IDEやエディタで、ボード名とポートは選べていますか？",
    "check_serial_value": "シリアルモニタに数値や起動メッセージは出ていますか？",
    "choose_next_extension": "次は見た目、音、センサー追加のどれを足したいですか？",
}

CHECKPOINTS = [
    "not_started",
    "idea_selected",
    "parts_ready",
    "wired_minimal",
    "code_uploaded",
    "observed_signal",
    "needs_debug",
    "standard_done",
    "completed",
]

CHECKPOINT_LABELS = {
    "not_started": "まだ開始前",
    "idea_selected": "作品候補を選択済み",
    "parts_ready": "部品確認済み",
    "wired_minimal": "最小回路まで完了",
    "code_uploaded": "コード書き込み済み",
    "observed_signal": "動作ログ確認済み",
    "needs_debug": "デバッグ中",
    "standard_done": "標準構成まで完了",
    "completed": "完成ログ保存済み",
}

ROUTES = ["minimal", "standard", "extension", "debug_recovery"]

ROUTE_LABELS = {
    "minimal": "最小版から進める",
    "standard": "標準版まで進める",
    "extension": "拡張版まで進める",
    "debug_recovery": "復旧優先で進める",
}

CONCEPTS = [
    "project_decomposition",
    "bom_check",
    "breadboard_rows",
    "led_polarity",
    "resistor_usage",
    "gnd_common",
    "gpio_pin_match",
    "firmware_upload",
    "serial_monitor",
    "analog_threshold",
    "sensor_power",
    "safe_power",
    "enclosure_fixing",
    "next_project_link",
]

CONCEPT_LABELS = {
    "project_decomposition": "作品を入力・処理・出力に分ける",
    "bom_check": "必要部品と不足部品を確認する",
    "breadboard_rows": "ブレッドボードの列のつながり",
    "led_polarity": "LEDの向き",
    "resistor_usage": "抵抗を直列に入れる理由",
    "gnd_common": "GND共有",
    "gpio_pin_match": "コードと配線のピン一致",
    "firmware_upload": "ボード設定と書き込み",
    "serial_monitor": "シリアルモニタで観察する",
    "analog_threshold": "センサー値としきい値",
    "sensor_power": "センサーのVCC/GND/SIG",
    "safe_power": "安全なUSB給電",
    "enclosure_fixing": "固定と外装",
    "next_project_link": "次作品への発展",
}


def index_or_default(values: list[str], value: str, default: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return values.index(default)
