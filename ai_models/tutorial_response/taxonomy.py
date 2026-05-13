from __future__ import annotations

PROJECTS = {
    "desk_pet": "近づくと鳴く机上ペット",
    "light_charm": "暗くなると光る小さなお守り",
    "plant_ping": "水やり通知",
    "posture_guard": "姿勢注意デバイス",
    "temp_face": "温度で表情が変わるミニキャラ",
}

STAGES = [
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

QUESTIONS = [
    "今日は、かわいい・便利・人に見せたい、どれに一番近いですか？",
    "今回の上限予算は何円くらいにしますか？",
    "手元にある部品を、分かる範囲で一行で書けますか？",
    "使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？",
    "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
    "LEDとGPIOの間に220Ω前後の抵抗は入っていますか？",
    "センサー、LED、ボードのGNDは同じGND列につながっていますか？",
    "Arduino IDEやエディタで、ボード名とポートは選べていますか？",
    "シリアルモニタに数値や起動メッセージは出ていますか？",
    "次は見た目、音、センサー追加のどれを足したいですか？",
]

ANSWER_PATTERNS = {
    "confirmed": [
        "はい",
        "つながっています",
        "同じGND列に入れました",
        "できました",
        "直しました",
        "起動メッセージが出ました",
        "LEDは点きました",
        "抵抗は入っています",
        "ESP32で進めます",
    ],
    "negative": [
        "いいえ",
        "まだできていません",
        "たぶん違う列です",
        "抵抗が入っていません",
        "LEDが光りません",
        "ログが出ません",
        "ポートが見えません",
    ],
    "unknown": [
        "分からない",
        "自信がないです",
        "どこがGNDか分かりません",
        "部品名が分かりません",
        "たぶん合っているけど不安です",
    ],
    "photo_request": [
        "写真で確認したいです",
        "写真を送れば分かりますか",
        "配線画像を見てほしいです",
    ],
    "problem_report": [
        "コンパイルエラーが出ました",
        "checkpoint loaded で止まります",
        "USBを挿すと落ちます",
        "値がずっと0です",
        "ブザーが鳴りません",
    ],
    "inventory_report": [
        "ESP32, LED, 220Ω抵抗, ブレッドボード, ジャンパ線, USBケーブル",
        "Arduino Uno と LED と抵抗があります",
        "M5Stack と Grove ケーブルがあります",
        "何を持っているか分からないのでスターター構成で進みたいです",
    ],
}

