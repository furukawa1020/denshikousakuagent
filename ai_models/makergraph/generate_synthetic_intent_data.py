from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .sample_data import PROJECT_CATALOG, project_to_document


MOOD_TEMPLATES = {
    "desk_pet": [
        "かわいい机上の相棒を作りたい",
        "近づくと反応する小さいキャラがほしい",
        "友達に見せられる動く作品がいい",
        "ESP32で鳴いたり光ったりする作品を作りたい",
    ],
    "light_charm": [
        "Lチカの次に安く作れる光るものがいい",
        "暗くなると光る小さなお守りを作りたい",
        "部屋に置ける失敗しにくいライトがいい",
        "LEDと光センサーでまず完成体験がほしい",
    ],
    "temp_face": [
        "温度で表情が変わるかわいいキャラを作りたい",
        "M5Stackで部屋に置ける環境モニターがほしい",
        "研究っぽく温湿度を見られる作品にしたい",
        "画面表示とセンサーを組み合わせたい",
    ],
    "plant_ping": [
        "植物を枯らすので水やり通知がほしい",
        "自動水やりは怖いから通知だけしたい",
        "土の乾き具合をセンサーで見たい",
        "水回りでも安全なUSB給電の作品にしたい",
    ],
    "posture_guard": [
        "集中できないので姿勢を注意してくれるものがほしい",
        "机との距離が近すぎたら知らせる装置がいい",
        "勉強中に短く鳴って休憩を促す作品を作りたい",
        "距離センサーと時間判定を学びたい",
    ],
}


BUDGET_PHRASES = [
    "予算は1000円以内",
    "予算は3000円くらい",
    "5000円以内で作りたい",
    "1万円までは出せる",
    "部品を持っているので追加費用を抑えたい",
]


INVENTORY_PHRASES = [
    "Arduinoを持っている",
    "ESP32を持っている",
    "M5StickCがある",
    "LEDと抵抗はある",
    "ブレッドボードとジャンパ線はある",
    "センサーはまだ持っていない",
]


CONSTRAINT_PHRASES = [
    "はんだ付けなしで作りたい",
    "AC100Vは扱いたくない",
    "リチウムイオン電池は避けたい",
    "まず最小構成で動かしたい",
    "失敗したらデバッグしやすいものがいい",
]


PROFILE_BY_PROJECT = {
    "desk_pet": {"difficulty_tolerance": 0.48, "budget_sensitivity": 0.62, "novelty_preference": 0.74},
    "light_charm": {"difficulty_tolerance": 0.24, "budget_sensitivity": 0.9, "novelty_preference": 0.38},
    "temp_face": {"difficulty_tolerance": 0.64, "budget_sensitivity": 0.48, "novelty_preference": 0.78},
    "plant_ping": {"difficulty_tolerance": 0.45, "budget_sensitivity": 0.68, "novelty_preference": 0.52},
    "posture_guard": {"difficulty_tolerance": 0.58, "budget_sensitivity": 0.56, "novelty_preference": 0.62},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic intent/project contrastive training data.")
    parser.add_argument("--output", default="data/intent_training_synthetic.jsonl")
    parser.add_argument("--records-per-project", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = generate_records(args.records_per_project, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "records": len(records)}, ensure_ascii=False, indent=2))


def generate_records(records_per_project: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    projects = {project["id"]: project for project in PROJECT_CATALOG}
    records: list[dict[str, Any]] = []
    for project_id, templates in MOOD_TEMPLATES.items():
        project = projects[project_id]
        for _ in range(records_per_project):
            query = build_query(project_id, templates, rng)
            records.append({
                "query": query,
                "positive_project": project,
                "positive_project_text": project_to_document(project),
                "targets": jitter_profile(PROFILE_BY_PROJECT[project_id], rng),
                "hard_negative_project_ids": choose_hard_negatives(project_id, rng),
            })
    rng.shuffle(records)
    return records


def build_query(project_id: str, templates: list[str], rng: random.Random) -> str:
    parts = [rng.choice(templates)]
    if rng.random() < 0.82:
        parts.append(rng.choice(BUDGET_PHRASES))
    if rng.random() < 0.74:
        parts.append(inventory_phrase_for(project_id, rng))
    if rng.random() < 0.7:
        parts.append(rng.choice(CONSTRAINT_PHRASES))
    if rng.random() < 0.36:
        parts.append(f"避けたいことは{avoidance_for(project_id, rng)}")
    rng.shuffle(parts)
    return "。".join(parts) + "。"


def inventory_phrase_for(project_id: str, rng: random.Random) -> str:
    preferred = {
        "desk_pet": ["ESP32を持っている", "LEDと抵抗はある", "ブレッドボードとジャンパ線はある"],
        "light_charm": ["Arduinoを持っている", "LEDと抵抗はある", "ブレッドボードとジャンパ線はある"],
        "temp_face": ["M5StickCがある", "センサーはまだ持っていない"],
        "plant_ping": ["ESP32を持っている", "センサーはまだ持っていない"],
        "posture_guard": ["ESP32を持っている", "ブレッドボードとジャンパ線はある"],
    }
    if rng.random() < 0.72:
        return rng.choice(preferred[project_id])
    return rng.choice(INVENTORY_PHRASES)


def avoidance_for(project_id: str, rng: random.Random) -> str:
    common = ["AC100V", "リチウムイオン電池", "難しいはんだ付け", "高電流モーター"]
    project_specific = {
        "plant_ping": ["水と電源が近い配線", "自動ポンプ制御"],
        "posture_guard": ["鳴りっぱなしのブザー", "複雑な割り込み"],
        "temp_face": ["ライブラリ地獄", "画面だけでセンサーがない作品"],
    }
    return rng.choice(project_specific.get(project_id, common) + common)


def jitter_profile(profile: dict[str, float], rng: random.Random) -> dict[str, float]:
    return {
        key: round(min(1.0, max(0.0, value + rng.uniform(-0.08, 0.08))), 3)
        for key, value in profile.items()
    }


def choose_hard_negatives(project_id: str, rng: random.Random) -> list[str]:
    confusing = {
        "desk_pet": ["posture_guard", "temp_face"],
        "light_charm": ["desk_pet", "plant_ping"],
        "temp_face": ["desk_pet", "posture_guard"],
        "plant_ping": ["light_charm", "temp_face"],
        "posture_guard": ["desk_pet", "plant_ping"],
    }
    negatives = confusing[project_id][:]
    others = [project["id"] for project in PROJECT_CATALOG if project["id"] != project_id and project["id"] not in negatives]
    rng.shuffle(others)
    return negatives + others[:2]


if __name__ == "__main__":
    main()
