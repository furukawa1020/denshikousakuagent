from __future__ import annotations

from ai_models.bom.component_catalog import COMPONENT_IDS, PROJECT_COMPONENTS


PROJECT_IDS = ["desk_pet", "light_charm", "temp_face", "plant_ping", "posture_guard"]

PROJECT_LABELS = {
    "desk_pet": "近づくと鳴く机上ペット",
    "light_charm": "暗くなると光る小さなお守り",
    "temp_face": "温度で表情が変わるミニキャラ",
    "plant_ping": "水やり通知",
    "posture_guard": "姿勢注意デバイス",
}

MATCH_TIERS = ["ready_now", "buy_small", "budget_stretch", "too_hard"]

MATCH_TIER_LABELS = {
    "ready_now": "すぐ作れる",
    "buy_small": "少し買えば作れる",
    "budget_stretch": "予算を少し超えそう",
    "too_hard": "今は難しめ",
}


def project_index(project_id: str) -> int:
    return PROJECT_IDS.index(project_id) if project_id in PROJECT_IDS else 0


def tier_index(tier: str) -> int:
    return MATCH_TIERS.index(tier) if tier in MATCH_TIERS else 0

