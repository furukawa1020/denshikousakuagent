from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_models.tutorial_response.data import build_source, write_jsonl
from ai_models.tutorial_response.taxonomy import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate contextual neural tutorial records with memory and progress state.")
    parser.add_argument("--response-output", default="runtime/tutorial_response_contextual_v1.jsonl")
    parser.add_argument("--classifier-output", default="runtime/tutorial_answer_classifier_contextual_v1.jsonl")
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=51701)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    response_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    for index in range(args.samples):
        response_row, classifier_row = make_pair(rng, index)
        response_rows.append(response_row)
        classifier_rows.append(classifier_row)
    write_jsonl(args.response_output, response_rows)
    write_jsonl(args.classifier_output, classifier_rows)
    counts: dict[str, int] = {}
    topics: dict[str, int] = {}
    for row in classifier_rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
        topics[row["topic"]] = topics.get(row["topic"], 0) + 1
    print({
        "event": "tutorial_contextual_neural_data_generated",
        "responseOutput": args.response_output,
        "classifierOutput": args.classifier_output,
        "records": args.samples,
        "kinds": counts,
        "topics": topics,
    })


def make_pair(rng: random.Random, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = rng.choices(SCENARIOS, weights=[item.get("weight", 1) for item in SCENARIOS], k=1)[0]
    project_id = rng.choice(list(PROJECTS))
    payload = {
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": scenario["stage"],
        "tutorialState": maybe_mutate_state(dict(scenario["state"]), rng),
        "question": rng.choice(scenario["questions"]),
        "answer": mutate_answer(rng.choice(scenario["answers"]), rng),
        "inventory": rng.choice(INVENTORIES),
        "budget": rng.choice(["0", "1000", "3000", "5000", "10000"]),
        "symptom": scenario.get("symptom", "none"),
        "skill": rng.choice(SKILLS),
        "previous": rng.choice(PREVIOUS_LOGS),
        "lastQuestion": rng.choice(scenario.get("lastQuestions") or [""]),
        "lastAnswer": rng.choice(scenario.get("lastAnswers") or [""]),
        "lastInterpreted": scenario.get("lastInterpreted", ""),
        "interpretedKind": scenario["kind"],
        "fallbackStage": scenario["nextStage"],
        "tutorialTopic": scenario["topic"],
    }
    source = build_source(payload)
    title = rng.choice(scenario["titles"])
    body = rng.choice(scenario["bodies"])
    next_instruction = rng.choice(scenario["next"])
    target = "\n".join([
        f"style:{scenario['style']}",
        f"title:{title}",
        f"body:{body}",
        f"next:{next_instruction}",
        f"stage:{scenario['nextStage']}",
    ])
    response_row = {
        "source": source,
        "target": target,
        "feedback": "fix",
        "generator": "tutorial_contextual_neural_v1",
        "topic": scenario["topic"],
    }
    classifier_row = {
        "source": source,
        "kind": scenario["kind"],
        "target": scenario["kind"],
        "feedback": "fix",
        "generator": "tutorial_contextual_neural_v1",
        "topic": scenario["topic"],
    }
    return response_row, classifier_row


def maybe_mutate_state(state: dict[str, bool], rng: random.Random) -> dict[str, bool]:
    if rng.random() < 0.08:
        return {key: bool(value) for key, value in state.items() if rng.random() > 0.2}
    return state


def mutate_answer(text: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "", "いま見た感じ、", "たぶん、", "確認したら、", "写真なしですが、", "I think "])
    suffix = rng.choice(["", "", "です", "だと思います", "、次は？", "。これで合ってますか？", ". what next?"])
    replacements = [
        ("GPIO", rng.choice(["GPIO", "gpio", "ピン"])),
        ("GND", rng.choice(["GND", "gnd", "マイナス列", "グランド"])),
        ("LED", rng.choice(["LED", "led", "発光ダイオード"])),
        ("220Ω", rng.choice(["220Ω", "220オーム", "抵抗"])),
    ]
    mutated = text
    for source, target in replacements:
        if rng.random() < 0.18:
            mutated = mutated.replace(source, target)
    return f"{prefix}{mutated}{suffix}"


INVENTORIES = [
    "ESP32、LED、220Ω抵抗、ブレッドボード、ジャンパ線、USBケーブル",
    "Arduino Uno、LED、330Ω抵抗、ブレッドボード",
    "M5Stack、Groveケーブル、距離センサー、LED",
    "Raspberry Pi Pico、LED、抵抗、ジャンパ線",
    "LEDと線はある。抵抗はあるか分からない",
    "部品名は曖昧。スターターキットっぽい箱がある",
]

SKILLS = [
    "",
    "Lチカだけやった",
    "GND共有がまだ不安",
    "抵抗の向きは分からない",
    "コンパイルエラーが苦手",
    "配線写真を見ながらなら進められる",
]

PREVIOUS_LOGS = [
    "",
    "前回は部品名で止まった",
    "GNDの質問で止まりやすい",
    "LEDの向き確認までは進んだ",
    "checkpoint loadedで止まったことがある",
    "動いたら次作品も作りたい",
]


SCENARIOS = [
    {
        "topic": "inventory_report",
        "kind": "inventory_report",
        "style": "good",
        "stage": "parts_check",
        "nextStage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 5,
        "questions": ["手元にある部品を、分かる範囲で一行で書けますか？", "持っている部品を教えてください。"],
        "answers": ["ESP32、LED、抵抗、ブレッドボードがあります", "ArduinoとLEDとジャンパ線があります", "M5Stackとセンサーがあります"],
        "titles": ["部品情報を受け取りました", "その部品で最小構成から進めます", "所持品を制作ルートに反映します"],
        "bodies": ["書いてくれた部品を優先して使い、不足しそうなものは部品表で分けます。ここでは止まらず、まず動く最小回路へ進めます。", "手元の部品で試せる形に寄せます。最初は完成形ではなく、LED・抵抗・GNDだけの小さい確認から始めます。"],
        "next": ["USBを抜いた状態で、GND共有の確認へ進みます。", "次はボード、LED、抵抗、GNDだけを見ます。"],
    },
    {
        "topic": "inventory_unknown",
        "kind": "unknown",
        "style": "warn",
        "stage": "parts_check",
        "nextStage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 3,
        "questions": ["手元にある部品を、分かる範囲で一行で書けますか？"],
        "answers": ["よく分からないです", "箱に入ってるけど名前が読めません", "部品名が分からないのでスターター構成で進めたい"],
        "titles": ["部品名が曖昧でも進めます", "分かる部品だけで大丈夫です", "スターター構成として扱います"],
        "bodies": ["部品名が完璧でなくても止めません。いったんLED・抵抗・ブレッドボード・USBケーブルがある前提で、足りないものは後で分けます。", "分からない部品は写真チェックに回せます。いまは最小構成を先に作る流れにします。"],
        "next": ["次はUSBを抜いて、GNDとLEDだけの確認へ進みます。", "分かるものだけ残して、最小回路から始めます。"],
    },
    {
        "topic": "gnd_confirmed",
        "kind": "confirmed",
        "style": "good",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "lastQuestions": ["手元にある部品を、分かる範囲で一行で書けますか？"],
        "lastAnswers": ["ESP32、LED、抵抗、ブレッドボード"],
        "lastInterpreted": "inventory_report",
        "weight": 6,
        "questions": ["センサー、LED、ボードのGNDは同じGND列につながっていますか？", "GNDは同じ列につながっていますか？"],
        "answers": ["同じGND列につながっています", "ボードのGNDとLEDの短い足が同じマイナス列です", "全部同じグランドに入っています"],
        "titles": ["GND共有は進められます", "GNDの基準はそろっています", "電源の基準は大丈夫そうです"],
        "bodies": ["GNDが同じ列なら、最小回路の土台はかなり安定します。まだ完成扱いにはせず、次はLEDの向きだけを確認します。", "同じGNDになっているので、コード以前の大きな詰まりは一つ減りました。次はLEDの長い足と短い足を見ます。"],
        "next": ["LEDの長い足が抵抗を通ってGPIO側、短い足がGND側か確認します。", "次はLEDの向きを一つだけ見ます。"],
    },
    {
        "topic": "gnd_negative",
        "kind": "negative",
        "style": "warn",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 4,
        "questions": ["センサー、LED、ボードのGNDは同じGND列につながっていますか？", "GNDは同じ列につながっていますか？"],
        "answers": ["別の列かもしれない", "つながっていません", "GND線をまだ出していないです", "同じか自信ないです"],
        "titles": ["まずGNDをそろえます", "ここはGNDで止まって正解です", "GND共有を先に直します"],
        "bodies": ["GNDが分かれていると、コードが正しくても反応しません。USBを抜いて、ボードのGNDとLEDの短い足側を同じマイナス列にまとめます。", "いまは先へ進まず、GNDだけ直すのが一番早いです。線の色より、刺さっている列を見ます。"],
        "next": ["同じGND列に移せたら「つなぎ直した」と答えてください。", "USBを抜いてから、GND線だけを同じ列へ移します。"],
    },
    {
        "topic": "led_confirmed",
        "kind": "confirmed",
        "style": "good",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "state": {"gnd": True, "led": False, "resistor": False, "firmware": False},
        "lastQuestions": ["センサー、LED、ボードのGNDは同じGND列につながっていますか？"],
        "lastAnswers": ["同じGND列です"],
        "lastInterpreted": "confirmed",
        "weight": 7,
        "questions": ["LEDの長い足は、抵抗を通ってGPIO側につながっていますか？", "LEDの向きは合っていますか？"],
        "answers": ["長い足が抵抗を通ってGPIO側につながっています", "LEDの長い足はGPIO側、短い足はGND側です", "向きは合っていると思います"],
        "titles": ["LEDの向きは進められます", "LEDの極性は大丈夫そうです", "LEDの向き確認はクリアです"],
        "bodies": ["長い足がGPIO側、短い足がGND側ならLEDの向きは進められます。次は直結になっていないか、抵抗だけを分けて確認します。", "LEDの極性は合っていそうです。ここで完成に飛ばず、次は保護抵抗が一本入っているかを見ます。"],
        "next": ["LEDとGPIOの間に220Ω前後の抵抗が一本入っているか確認します。", "次は抵抗確認に進み、GPIO直結になっていないかだけ見ます。"],
    },
    {
        "topic": "led_unknown",
        "kind": "unknown",
        "style": "warn",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "state": {"gnd": True, "led": False, "resistor": False, "firmware": False},
        "weight": 3,
        "questions": ["LEDの長い足は、抵抗を通ってGPIO側につながっていますか？", "LEDの向きは合っていますか？"],
        "answers": ["足の長さが分からない", "向きに自信がありません", "どっちが長い足か見えません", "写真で見てほしい"],
        "titles": ["LEDの向きだけ確認しましょう", "ここはLEDの極性を見ます", "分からなくても一つずつ見ます"],
        "bodies": ["LEDは向きがあります。USBを抜いて、足の長い側をGPIO側、短い側をGND側にします。足が切られている場合は平らな面も手がかりです。", "自信がないなら進めずにLEDだけ見ます。ここをそろえると後のデバッグがかなり楽になります。"],
        "next": ["向きを確認できたら「長い足がGPIO側」と答えてください。", "写真で確認したい場合は、LEDの足元が見えるように真上から撮ります。"],
    },
    {
        "topic": "resistor_confirmed",
        "kind": "confirmed",
        "style": "good",
        "stage": "minimal_circuit",
        "nextStage": "firmware_upload",
        "state": {"gnd": True, "led": True, "resistor": False, "firmware": False},
        "lastQuestions": ["LEDの長い足は、抵抗を通ってGPIO側につながっていますか？"],
        "lastAnswers": ["長い足が抵抗経由でGPIO側です"],
        "lastInterpreted": "confirmed",
        "weight": 7,
        "questions": ["LEDとGPIOの間に220Ω前後の抵抗は入っていますか？", "LEDとGPIOの間に抵抗が入っていますか？"],
        "answers": ["220Ωの抵抗を一本はさんでいます", "抵抗は直列に入っています", "LEDとGPIOの間に抵抗があります", "直結ではなく抵抗経由です"],
        "titles": ["抵抗は入っています", "LED保護はできています", "最小回路はコード確認へ進めます"],
        "bodies": ["抵抗が直列に入っているなら、LEDをGPIOへ直結していないので安全側です。これで最小回路の確認はコード側へ進めます。", "GND、LEDの向き、抵抗がそろったので、次はコードのGPIO番号と配線先を一致させます。"],
        "next": ["ボードとポートを選び、シリアル出力つきの確認コードを書き込みます。", "次はコードのGPIO番号と配線先が同じか確認します。"],
    },
    {
        "topic": "resistor_missing",
        "kind": "negative",
        "style": "warn",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "state": {"gnd": True, "led": True, "resistor": False, "firmware": False},
        "weight": 4,
        "questions": ["LEDとGPIOの間に220Ω前後の抵抗は入っていますか？", "LEDとGPIOの間に抵抗が入っていますか？"],
        "answers": ["抵抗なしでつないでいます", "ジャンパ線だけです", "抵抗が見当たりません", "直結かもしれません"],
        "titles": ["抵抗を一本入れます", "LED直結はここで止めます", "保護抵抗を先に足します"],
        "bodies": ["LEDとGPIOを直結しないように、220Ω前後の抵抗を一本直列に入れます。抵抗に向きはありません。", "ここは先へ進まず、抵抗を足すのが正解です。USBを抜いてから作業します。"],
        "next": ["抵抗を入れたら「抵抗を入れた」と答えてください。", "USBを抜いて、LEDとGPIOの間に抵抗を一本はさみます。"],
    },
    {
        "topic": "board_report",
        "kind": "board_report",
        "style": "good",
        "stage": "parts_check",
        "nextStage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 3,
        "questions": ["使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？", "マイコンの名前は分かりますか？"],
        "answers": ["ESP32です", "Arduino Unoを使います", "M5Stackです", "Picoだと思います"],
        "titles": ["ボード情報を反映しました", "このマイコンで進めます", "ボードに合わせて確認します"],
        "bodies": ["ボードに合わせてピン番号と書き込み手順を変えます。まずはボード固有の機能に寄せず、LEDだけの最小確認から入ります。", "ボード名が分かったので、配線とコードのGPIO番号を一致させやすくなりました。"],
        "next": ["次はGND共有とLEDの向きを確認します。", "最小回路のGND確認へ進みます。"],
    },
    {
        "topic": "upload_confirmed",
        "kind": "confirmed",
        "style": "good",
        "stage": "firmware_upload",
        "nextStage": "observe_serial",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 4,
        "questions": ["ボードとポートを選んで書き込めましたか？", "確認コードを書き込めましたか？"],
        "answers": ["書き込みできました", "COMポートを選んでアップロードできました", "エラーなしで入りました"],
        "titles": ["書き込みは通りました", "コードはボードに入りました", "次はシリアル確認です"],
        "bodies": ["書き込みが通ったので、次は回路が反応しているかをシリアルモニタで見ます。ここで値を見ると、配線ミスとコードミスを分けられます。", "アップロードできたなら、PCとボードの接続は一段クリアです。次はログを見ます。"],
        "next": ["シリアルモニタを開いて、起動メッセージや値が出るか確認します。", "Serial Monitorの速度をコード側と合わせます。"],
    },
    {
        "topic": "upload_problem",
        "kind": "problem_report",
        "style": "warn",
        "stage": "firmware_upload",
        "nextStage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 5,
        "questions": ["ボードとポートを選んで書き込めましたか？", "確認コードを書き込めましたか？"],
        "answers": ["COMポートが出ません", "Failed to connect と出ます", "checkpoint loadedで止まります", "書き込みでエラーが出ます"],
        "titles": ["書き込み周りを切り分けます", "PC接続の確認に戻ります", "アップロードエラーとして扱います"],
        "bodies": ["配線を大きく触る前に、USBケーブル、ポート、ボード設定を一つずつ見ます。充電専用ケーブルだとここで止まります。", "書き込みエラーは回路よりPC接続側の可能性があります。まず表示されているエラー文とポートを確認します。"],
        "next": ["表示されているエラー文を一行だけ貼ってください。", "別のUSBケーブルかポートで、ボードが認識されるか確認します。"],
    },
    {
        "topic": "serial_confirmed",
        "kind": "confirmed",
        "style": "good",
        "stage": "observe_serial",
        "nextStage": "standard_build",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 4,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？", "Serial Monitorに値が出ていますか？"],
        "answers": ["start と値が出ています", "手を近づけると値が変わります", "ログが見えています"],
        "titles": ["ログが見えています", "センサー値まで確認できました", "最小動作は見えています"],
        "bodies": ["起動メッセージや値が出ているなら、コードは最低限動いています。次は作品らしく見える標準構成へ広げられます。", "値が変わるところまで見えたので、入力から出力までの一本線ができました。"],
        "next": ["次は鳴らす、光らせ方を変える、外装を足す、のどれかを選びます。", "標準構成として、反応条件と見た目を整えます。"],
    },
    {
        "topic": "serial_problem",
        "kind": "problem_report",
        "style": "warn",
        "stage": "observe_serial",
        "nextStage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 4,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？", "Serial Monitorに値が出ていますか？"],
        "answers": ["何も出ません", "文字化けしています", "ずっと0です", "値が変わりません"],
        "titles": ["ログの見え方を切り分けます", "シリアル確認で止めます", "値が出ない原因を一つずつ見ます"],
        "bodies": ["ここで一気に配線を全部変えず、速度設定、ポート、起動メッセージ、センサー線の順に分けます。", "値が出ない場合は、コードが動いていないのか、センサー入力が来ていないのかを分けます。"],
        "next": ["シリアルモニタの速度と、表示されている文字を一行だけ教えてください。", "まず start だけでも出るか確認します。"],
    },
    {
        "topic": "photo_request",
        "kind": "photo_request",
        "style": "warn",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 3,
        "questions": ["配線が合っているか不安ですか？", "LEDの向きは合っていますか？", "GNDは同じ列ですか？"],
        "answers": ["写真で見てほしいです", "配線写真を送ります", "画像チェックしたいです"],
        "titles": ["写真チェックに切り替えます", "配線写真で確認します", "画像で怪しい箇所を見ます"],
        "bodies": ["写真で見る場合も、USBは抜いた状態にします。ボード名、GND、LED、抵抗、ジャンパ線の刺さる列が一枚に入ると判断しやすいです。", "真上から明るく撮ると、同じ列かどうかを見やすくなります。"],
        "next": ["USBを抜いて、真上から配線全体を撮ってください。", "撮れたら配線写真チェックへ進みます。"],
    },
    {
        "topic": "stuck_report",
        "kind": "problem_report",
        "style": "warn",
        "stage": "minimal_circuit",
        "nextStage": "debug_triage",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 3,
        "questions": ["いまの状態をそのまま書いてください。", "どこで止まっていますか？"],
        "answers": ["進まないです", "返答が来ません", "何を答えればいいか分かりません", "LEDが光りません"],
        "titles": ["止まっている状態を受け取りました", "ここから一問ずつ戻します", "詰まりをデバッグとして扱います"],
        "bodies": ["いまは完成まで飛ばず、止まった箇所をログにします。まず物理、次に電源、最後にコードの順で一つずつ見ます。", "分からない回答でも大丈夫です。次の確認を一つに絞ります。"],
        "next": ["まずUSBを抜いて、GNDが同じ列かだけ見ます。", "一番近い症状を一言で書いてください。例: 光らない、書き込めない、値が出ない。"],
    },
]


if __name__ == "__main__":
    main()
