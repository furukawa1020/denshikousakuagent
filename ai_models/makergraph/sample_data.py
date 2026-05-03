from __future__ import annotations

from typing import Any


PROJECT_CATALOG: list[dict[str, Any]] = [
    {
        "id": "desk_pet",
        "title": "近づくと鳴く机上ペット",
        "description": "距離センサーで人が近づいたことを検知し、LEDとブザーで小さな机上ペットの反応を作る。",
        "components": ["ESP32", "距離センサー", "LED", "220Ω抵抗", "ブザー", "ブレッドボード", "ジャンパ線", "USBケーブル"],
        "graph": ["input:distance_sensor", "processing:esp32", "output:led", "output:buzzer", "power:usb_5v", "safety:gnd_common"],
        "skills": ["GPIO", "GND共有", "条件分岐", "デジタル入力", "ブザー"],
        "budget": "3500-6000",
        "difficulty": 2,
        "failure_logs": ["GND未共有", "GPIO番号不一致", "距離センサー電圧仕様の確認漏れ"],
    },
    {
        "id": "light_charm",
        "title": "暗くなると光る小さなお守り",
        "description": "光センサーで明るさを読み、暗くなったらLEDを光らせる。Lチカの次に最小構成で作りやすい。",
        "components": ["Arduino Uno", "光センサー", "LED", "220Ω抵抗", "ブレッドボード", "ジャンパ線", "USBケーブル"],
        "graph": ["input:light_sensor", "processing:arduino", "output:led", "power:usb_5v", "safety:resistor_required"],
        "skills": ["LED極性", "抵抗", "アナログ入力", "しきい値処理"],
        "budget": "1500-3000",
        "difficulty": 1,
        "failure_logs": ["LED極性逆", "抵抗なしLED", "充電専用USBケーブル"],
    },
    {
        "id": "temp_face",
        "title": "温度で表情が変わるミニキャラ",
        "description": "温湿度センサーとM5StickCの画面で、温度に応じて表情が変わるキャラクターを作る。",
        "components": ["M5StickC", "温湿度センサー", "USBケーブル", "Groveケーブル"],
        "graph": ["input:temperature_sensor", "processing:m5stickc", "output:lcd_face", "power:usb_5v", "skill:state_management"],
        "skills": ["温湿度センサー", "画面表示", "状態管理", "ライブラリ導入"],
        "budget": "5000-9000",
        "difficulty": 3,
        "failure_logs": ["DHTライブラリ未導入", "センサー値nan", "表示更新が速すぎる"],
    },
    {
        "id": "plant_ping",
        "title": "水やり通知",
        "description": "土壌水分センサーで植物の土の乾き具合を読み、乾いたらLEDやブザーで知らせる。",
        "components": ["ESP32", "静電容量式土壌水分センサー", "LED", "220Ω抵抗", "ブレッドボード", "ジャンパ線", "USBケーブル"],
        "graph": ["input:soil_sensor", "processing:esp32", "output:led", "power:usb_5v", "safety:water_distance"],
        "skills": ["アナログ入力", "しきい値処理", "水回り安全", "センサー校正"],
        "budget": "2500-5500",
        "difficulty": 2,
        "failure_logs": ["抵抗式センサーの腐食", "水と電源が近い", "しきい値が環境で変わる"],
    },
    {
        "id": "posture_guard",
        "title": "姿勢注意デバイス",
        "description": "距離センサーで机との距離を測り、近すぎる状態が続くとLEDやブザーで知らせる。",
        "components": ["ESP32", "距離センサー", "LED", "220Ω抵抗", "ブザー", "ブレッドボード", "ジャンパ線"],
        "graph": ["input:distance_sensor", "processing:esp32", "output:led", "output:buzzer", "skill:millis", "skill:state_management"],
        "skills": ["距離センサー", "時間判定", "状態管理", "millis"],
        "budget": "4200-8000",
        "difficulty": 3,
        "failure_logs": ["delayで反応が重い", "ブザーが鳴りっぱなし", "センサーの向きが違う"],
    },
]


TRAINING_EXAMPLES: list[dict[str, Any]] = [
    {
        "query": "作りたいものは分からない。予算は5000円。かわいいものがいい。",
        "positive_project_id": "desk_pet",
        "targets": {"difficulty_tolerance": 0.42, "budget_sensitivity": 0.72, "novelty_preference": 0.61},
    },
    {
        "query": "Lチカの次に進みたい。安くて失敗しにくい光る作品がいい。",
        "positive_project_id": "light_charm",
        "targets": {"difficulty_tolerance": 0.25, "budget_sensitivity": 0.9, "novelty_preference": 0.35},
    },
    {
        "query": "植物をよく枯らす。自動水やりは怖いので通知だけしたい。",
        "positive_project_id": "plant_ping",
        "targets": {"difficulty_tolerance": 0.45, "budget_sensitivity": 0.68, "novelty_preference": 0.52},
    },
    {
        "query": "部屋に置ける研究っぽい環境モニターを作りたい。M5Stackを使いたい。",
        "positive_project_id": "temp_face",
        "targets": {"difficulty_tolerance": 0.62, "budget_sensitivity": 0.44, "novelty_preference": 0.76},
    },
    {
        "query": "集中できないので姿勢や机との距離を注意してくれるものがほしい。",
        "positive_project_id": "posture_guard",
        "targets": {"difficulty_tolerance": 0.58, "budget_sensitivity": 0.55, "novelty_preference": 0.64},
    },
    {
        "query": "友達に見せたい。近づいたら反応して鳴く小さいキャラがいい。",
        "positive_project_id": "desk_pet",
        "targets": {"difficulty_tolerance": 0.5, "budget_sensitivity": 0.58, "novelty_preference": 0.8},
    },
    {
        "query": "1000円から3000円くらいで、暗い部屋で光るものを作りたい。",
        "positive_project_id": "light_charm",
        "targets": {"difficulty_tolerance": 0.3, "budget_sensitivity": 0.95, "novelty_preference": 0.41},
    },
    {
        "query": "センサー値をシリアルモニタで見て、乾いたら知らせる植物用の作品を作りたい。",
        "positive_project_id": "plant_ping",
        "targets": {"difficulty_tolerance": 0.5, "budget_sensitivity": 0.62, "novelty_preference": 0.48},
    },
    {
        "query": "温度で表情が変わるかわいいミニキャラを作りたい。画面表示もやりたい。",
        "positive_project_id": "temp_face",
        "targets": {"difficulty_tolerance": 0.64, "budget_sensitivity": 0.5, "novelty_preference": 0.84},
    },
    {
        "query": "勉強が続かない。近すぎる姿勢が続いたら短く鳴るものがいい。",
        "positive_project_id": "posture_guard",
        "targets": {"difficulty_tolerance": 0.6, "budget_sensitivity": 0.57, "novelty_preference": 0.62},
    },
]


def project_by_id(project_id: str) -> dict[str, Any]:
    for project in PROJECT_CATALOG:
        if project["id"] == project_id:
            return project
    raise KeyError(project_id)


def project_to_document(project: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"title: {project['title']}",
            f"description: {project['description']}",
            "components: " + ", ".join(project["components"]),
            "graph: " + ", ".join(project["graph"]),
            "skills: " + ", ".join(project["skills"]),
            f"budget: {project['budget']}",
            f"difficulty: {project['difficulty']}",
            "failure_logs: " + ", ".join(project["failure_logs"]),
        ]
    )
