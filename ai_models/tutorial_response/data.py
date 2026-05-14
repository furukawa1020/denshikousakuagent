from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_models.makergraph.tokenizer import MakerTokenizer

from .taxonomy import PROJECTS, STAGES


QUESTION_BANK = {
    "mood": [
        "今日は、かわいい・便利・人に見せたい、どれに一番近いですか？",
        "作るなら、かわいい系・実用系・見せる系のどれに寄せたいですか？",
        "今の気分に近いのは、光るもの・動くもの・生活の困りごとのどれですか？",
    ],
    "budget": [
        "今回の上限予算は何円くらいにしますか？",
        "追加で買える部品代は、だいたい何円までにしますか？",
        "まずは0円で試すか、少し買い足して進めるか、どちらにしますか？",
    ],
    "inventory": [
        "手元にある部品を、分かる範囲で一行で書けますか？",
        "今机の上にある部品名を、分かるものだけ並べてください。",
        "持っているボード、LED、抵抗、センサー、ケーブルを分かる範囲で教えてください。",
        "部品名があいまいでもいいので、箱や袋に書いてある名前を一行で書けますか？",
    ],
    "board": [
        "使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？",
        "コードを書き込む先のマイコンは何ですか？",
        "Arduino IDEで選ぶ予定のボード名は分かりますか？",
    ],
    "gnd": [
        "センサー、LED、ボードのGNDは同じGND列につながっていますか？",
        "ボードのGNDと、LEDやセンサーのマイナス側は同じ列でつながっていますか？",
        "黒や青のジャンパ線は、全部同じGNDラインに集まっていますか？",
        "ブレッドボードのマイナス列とマイコンのGNDはつながっていますか？",
    ],
    "led": [
        "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
        "LEDの短い足はGND側、長い足は抵抗側になっていますか？",
        "LEDの向きは、足の長さを見て確認できていますか？",
    ],
    "resistor": [
        "LEDとGPIOの間に220Ω前後の抵抗は入っていますか？",
        "LEDをGPIOへ直結せず、抵抗を直列に入れていますか？",
        "抵抗はLEDの足とGPIOの間に一本入っていますか？",
    ],
    "usb_port": [
        "Arduino IDEやエディタで、ボード名とポートは選べていますか？",
        "PC側でCOMポート、またはシリアルポートは表示されていますか？",
        "USBケーブルはデータ通信対応で、書き込み先ポートが見えていますか？",
    ],
    "serial": [
        "シリアルモニタに数値や起動メッセージは出ていますか？",
        "Serial Monitorに start やセンサー値は表示されていますか？",
        "手を近づけたり暗くしたりした時、シリアルの値は変わりますか？",
    ],
    "extension": [
        "次は見た目、音、センサー追加のどれを足したいですか？",
        "完成版に近づけるなら、光・音・動き・外装のどれを足したいですか？",
        "次の一歩は、かわいさ強化、実用性強化、ログ保存のどれにしますか？",
    ],
}


INVENTORY_CONTEXTS = [
    "ESP32, LED, 220Ω抵抗, ブレッドボード, ジャンパ線, USBケーブル",
    "Arduino Uno, LED, 330Ω抵抗, ブレッドボード, USBケーブル",
    "M5Stack, Groveケーブル, LED, 距離センサー",
    "Raspberry Pi Pico, LED, 抵抗, ジャンパ線",
    "不明。スターター構成で進みたい",
    "ESP32と何本かのジャンパ線だけ分かる。抵抗はあるか不明",
    "LEDセット、抵抗セット、ブレッドボード、古いUSBケーブル",
    "M5StickC, Groveセンサー, 紙箱, 両面テープ",
]


ANSWER_BANK = {
    "inventory": {
        "inventory_report": [
            "ESP32、LED、220Ω抵抗、ブレッドボード、ジャンパ線、USBケーブルがあります",
            "Arduino UnoとLEDと抵抗セットがあります。ブレッドボードもあります",
            "M5Stack、Groveケーブル、距離センサーっぽいものがあります",
            "LEDと抵抗はありますが、センサー名が分かりません",
            "部品箱にESP32、LED、ブザー、サーボ、ジャンパ線が入っています",
        ],
        "unknown": [
            "部品名がよく分からないです",
            "箱に色々入っているけど、どれが何か自信がありません",
            "スターターキットっぽいものはありますが名前が読めません",
            "写真なら分かるかもしれませんが、文章では不安です",
        ],
        "photo_request": [
            "写真で見てもらいたいです",
            "部品を机に並べて撮れば判断できますか",
            "袋の文字が読めないので画像で確認したいです",
        ],
    },
    "gnd": {
        "confirmed": [
            "はい、黒い線は全部同じマイナス列に入れました",
            "ボードのGNDからブレッドボードの青い列につながっています",
            "LEDの短い足側もセンサーのGNDも同じ列にしました",
            "つなぎ直したら全部同じGNDラインになりました",
        ],
        "negative": [
            "たぶん違う列に刺さっています",
            "LEDのGNDだけ別のところに行っている気がします",
            "青い列が途中で切れているかもしれません",
            "ボードのGNDからブレッドボードに線を出していませんでした",
        ],
        "unknown": [
            "同じ列か分かりません",
            "ブレッドボードの列のつながりがまだ分かっていません",
            "黒い線はありますが、同じGNDか自信ないです",
        ],
        "photo_request": [
            "写真でGNDが合っているか見てほしいです",
            "配線がごちゃっとしているので画像で確認したいです",
        ],
    },
    "led": {
        "confirmed": [
            "LEDの長い足は抵抗側につながっています",
            "短い足をGND側にしました",
            "LEDの向きは足の長さで確認しました",
        ],
        "negative": [
            "LEDの足の長さを見ずに挿していました",
            "長い足がGND側かもしれません",
            "LEDの向きは逆だった気がします",
        ],
        "unknown": [
            "足を切ってしまって長さが分かりません",
            "LEDの向きが分からないです",
        ],
    },
    "resistor": {
        "confirmed": [
            "220Ωの抵抗をLEDとGPIOの間に入れています",
            "抵抗は直列に入っています",
            "色の帯は分からないけど抵抗っぽい部品は間に入れました",
        ],
        "negative": [
            "抵抗を入れずにLEDを直接つないでいました",
            "抵抗が見つからないです",
            "LEDとGPIOの間にはジャンパ線しかありません",
        ],
        "unknown": [
            "どれが抵抗か分からないです",
            "抵抗値が合っているか不安です",
        ],
    },
    "board": {
        "board_report": [
            "ESP32で進めたいです",
            "Arduino Unoを使っています",
            "M5Stackです。Groveケーブルがあります",
            "Picoを持っています",
        ],
        "unknown": [
            "ボード名が分からないです",
            "ESP32っぽいですが自信がないです",
        ],
        "problem_report": [
            "ポートが表示されません",
            "書き込みでエラーが出ています",
            "USBを挿しても認識されません",
        ],
    },
    "usb_port": {
        "confirmed": [
            "COMポートが見えています",
            "ボード名とポートは選べました",
            "書き込みは成功しました",
        ],
        "negative": [
            "ポートが出てきません",
            "ボード名が合っているか分かりません",
            "書き込みに失敗します",
        ],
        "problem_report": [
            "A fatal error occurred と出ます",
            "Failed to connect と表示されています",
            "checkpoint loadedで止まっています",
        ],
    },
    "serial": {
        "confirmed": [
            "start と表示されました",
            "数値がシリアルモニタに出ています",
            "手を近づけると値が変わりました",
        ],
        "negative": [
            "何も表示されません",
            "値がずっと0です",
            "文字化けしています",
        ],
        "problem_report": [
            "シリアルモニタを開くとエラーになります",
            "リセットを押すと一瞬だけ出て消えます",
        ],
    },
    "extension": {
        "confirmed": [
            "音を足したいです",
            "外装をかわいくしたいです",
            "センサーを一つ増やしたいです",
            "ログ保存をやってみたいです",
        ],
        "unknown": [
            "次に何を足せばいいか分かりません",
            "完成に近づけたいけど迷っています",
        ],
        "problem_report": [
            "拡張したら動かなくなりました",
            "ブザーを足したらLEDも消えました",
        ],
    },
}


ROBUST_CONTEXTS = {
    "inventories": [
        "ESP32 DevKit, LED 5個, 220Ω抵抗, ブレッドボード, ジャンパ線, USBケーブル",
        "Arduino Uno, 赤色LED, 抵抗セット, 超音波センサーHC-SR04, ブザー",
        "M5Stack Basic, Groveケーブル, 距離センサー, サーボ, 紙箱",
        "Raspberry Pi Pico, LED, タクトスイッチ, OLEDっぽい小さい画面, 抵抗",
        "部品名は分からない。青い基板、光るやつ、抵抗っぽい小さい部品、線がある",
        "スターターキット一式。Arduino互換ボード、センサーいろいろ、ブレッドボード",
        "ESP32とUSBケーブルだけはある。LEDと抵抗は買う必要があるかも",
        "M5StickC, Grove温湿度センサー, 両面テープ, 小さい箱",
    ],
    "skills": [
        "gnd_common:0.10, led_polarity:0.20, gpio:0.15, debugging:0.10",
        "gnd_common:0.45, led_polarity:0.62, gpio:0.38, serial_monitor:0.22",
        "gpio:0.55, resistor_usage:0.40, serial_monitor:0.52, project_decomposition:0.30",
        "debugging:0.18, upload_error_reading:0.12, enclosure_design:0.05",
        "",
    ],
    "previous": [
        "",
        "前回はLEDの向きを逆にしていて光らなかった",
        "USBケーブルが充電専用で書き込めなかった",
        "GNDを別の列に刺していてセンサー値が変わらなかった",
        "抵抗なしでLEDをつなぎそうになったので止めた",
        "シリアルモニタの速度が違って文字化けした",
    ],
}


ROBUST_SCENARIOS: list[dict[str, Any]] = [
    {
        "topic": "inventory",
        "stages": ["parts_check", "orient"],
        "questions": [
            "手元にある部品を、分かる範囲で一行で書けますか？",
            "部品名があいまいでも大丈夫なので、机の上にあるものを並べてください。",
            "今持っているボード、LED、センサー、線、抵抗を分かる範囲で教えてください。",
            "スターターキットっぽいものがあれば、中身をざっくり書けますか？",
        ],
        "answers": {
            "inventory_report": [
                "ESP32、LED、抵抗、ブレッドボード、ジャンパ線があります",
                "ArduinoとLEDと抵抗セット、あと超音波センサーっぽいものがあります",
                "M5StackとGroveケーブル、距離センサー、ブザーがあります",
                "青い基板とLEDと線はあります。抵抗はたぶんあります",
                "スターターキット一式がありますが、名前は全部は分からないです",
            ],
            "unknown": [
                "部品名がよく分からないです",
                "箱にいろいろ入ってますが、どれが何か自信ないです",
                "LEDっぽいものはあります。抵抗かどうか分からない部品もあります",
            ],
            "photo_request": [
                "写真で見てもらいたいです",
                "机に並べて写真を送った方が早そうです",
                "部品名が読めないので画像で確認したいです",
            ],
        },
    },
    {
        "topic": "gnd",
        "stages": ["minimal_circuit", "debug_triage"],
        "questions": [
            "センサー、LED、ボードのGNDは同じGND列につながっていますか？",
            "ボードのGNDとブレッドボードのマイナス列はつながっていますか？",
            "黒や青のジャンパ線は、全部同じGNDラインに集まっていますか？",
            "GNDが共通になっているか、今見える範囲で教えてください。",
        ],
        "answers": {
            "confirmed": [
                "同じGND列につながっています",
                "ボードのGNDからマイナス列に線が出ています",
                "LEDの短い足側もセンサーのGNDも同じ列にしました",
                "たぶん同じところにつながっています",
                "さっき別でしたが、つなぎ直しました",
            ],
            "negative": [
                "別の列に刺さっているかもしれません",
                "GNDの線がセンサー側に行っていないです",
                "同じ列ではない気がします",
            ],
            "unknown": [
                "同じ列か分からないです",
                "ブレッドボードの列のつながりがまだ自信ないです",
                "黒い線はありますがGNDか分かりません",
            ],
            "photo_request": [
                "写真で確認してほしいです",
                "配線がごちゃっとしているので画像で見てもらいたいです",
            ],
        },
    },
    {
        "topic": "led",
        "stages": ["minimal_circuit", "debug_triage"],
        "questions": [
            "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
            "LEDの短い足はGND側、長い足は抵抗側になっていますか？",
            "LEDの向きは、足の長さを見て確認できていますか？",
            "LEDの極性を、今見えている状態で教えてください。",
        ],
        "answers": {
            "confirmed": [
                "長い足が抵抗を通ってGPIO側につながっています",
                "短い足をGND側にしました",
                "向きは大丈夫そうです",
                "たぶん合っています。長い足がプラス側です",
                "逆だったので直しました",
            ],
            "negative": [
                "長い足がGND側かもしれません",
                "向きを見ずに刺してしまいました",
                "たぶん逆です",
            ],
            "unknown": [
                "足を切ってしまって長さが分からないです",
                "LEDの向きが分からないです",
                "どっちが長い足だったか自信ないです",
            ],
            "problem_report": [
                "向きを直しても光りません",
                "LEDが一瞬だけ光って消えます",
                "LEDは光らないけどボードのランプは光っています",
            ],
        },
    },
    {
        "topic": "resistor",
        "stages": ["minimal_circuit", "debug_triage"],
        "questions": [
            "LEDとGPIOの間に220Ω前後の抵抗は入っていますか？",
            "LEDをGPIOへ直接つながず、抵抗を一本はさんでいますか？",
            "抵抗っぽい部品はLEDの足とGPIOの間に入っていますか？",
            "抵抗なしでLEDを直結していないか確認できますか？",
        ],
        "answers": {
            "confirmed": [
                "220Ωの抵抗を一本はさんでいます",
                "抵抗は入っています。色の帯がある部品です",
                "GPIOとLEDの間に抵抗を入れました",
                "直接ではなく抵抗経由になっています",
            ],
            "negative": [
                "直接つないでいました",
                "抵抗が見つからないです",
                "ジャンパ線だけでつないでいるかも",
            ],
            "unknown": [
                "どれが抵抗か分からないです",
                "抵抗値が合っているか自信ないです",
                "色の帯が読めません",
            ],
        },
    },
    {
        "topic": "board",
        "stages": ["firmware_upload", "parts_check"],
        "questions": [
            "使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？",
            "コードを書き込む先のマイコン名は分かりますか？",
            "Arduino IDEで選ぶ予定のボード名を教えてください。",
            "手元の基板はESP32系、Arduino系、M5Stack系のどれに近いですか？",
        ],
        "answers": {
            "board_report": [
                "ESP32です",
                "Arduino Unoを使っています",
                "M5Stack Basicです",
                "Picoっぽいです",
                "ESP32 DevKitと書いてあります",
            ],
            "unknown": [
                "ボード名が分からないです",
                "ESP32っぽいけど自信ないです",
                "青い基板です。名前は読めません",
            ],
            "problem_report": [
                "ボードを選ぶところで迷っています",
                "選ぶボード名が見つかりません",
                "USBを挿しても認識されません",
            ],
        },
    },
    {
        "topic": "usb_port",
        "stages": ["firmware_upload", "debug_triage"],
        "questions": [
            "Arduino IDEやエディタで、ボード名とポートは選べていますか？",
            "PC側でCOMポート、またはシリアルポートは表示されていますか？",
            "USBケーブルはデータ通信対応で、書き込み先ポートが見えていますか？",
            "コードのアップロードは完了しましたか？エラーは出ていますか？",
        ],
        "answers": {
            "confirmed": [
                "COMポートが見えています",
                "書き込みできました",
                "ボード名とポートを選べました",
                "アップロード完了と出ました",
            ],
            "negative": [
                "ポートが出ません",
                "書き込みできていません",
                "USBを挿しても反応しないです",
            ],
            "problem_report": [
                "Failed to connect と出ます",
                "A fatal error occurred と出ました",
                "checkpoint loadedで止まっています",
                "コンパイルは通るけど書き込みで失敗します",
            ],
            "unknown": [
                "どれがポートか分からないです",
                "エラー文が長くて読めません",
            ],
        },
    },
    {
        "topic": "serial",
        "stages": ["observe_serial", "debug_triage"],
        "questions": [
            "シリアルモニタに起動メッセージや数値は出ていますか？",
            "Serial Monitorに start やセンサー値は表示されていますか？",
            "手を近づけたり離したりしたとき、シリアルの値は変わりますか？",
            "シリアルモニタの表示を、そのまま一行で教えてください。",
        ],
        "answers": {
            "confirmed": [
                "start と数値が出ています",
                "手を近づけると値が変わります",
                "シリアルに距離っぽい数字が出ています",
                "起動メッセージと数値が出ています",
            ],
            "negative": [
                "何も出ません",
                "値がずっと0です",
                "文字化けしています",
            ],
            "problem_report": [
                "シリアルモニタを開くとエラーになります",
                "リセットすると一瞬だけ出て消えます",
                "値が変わらないです",
            ],
            "unknown": [
                "シリアルモニタの開き方が分からないです",
                "どの数字を見ればいいか分かりません",
            ],
        },
    },
    {
        "topic": "extension",
        "stages": ["standard_build", "enclosure", "extension"],
        "questions": [
            "次は見た目、音、センサー追加のどれを足したいですか？",
            "完成版に近づけるなら、光・音・動き・外装のどれを足しますか？",
            "作品らしさを出すために、一つだけ追加するとしたら何にしますか？",
            "ここで完成ログにするか、もう一機能だけ足すか、どちらにしますか？",
        ],
        "answers": {
            "confirmed": [
                "ブザーを追加したいです",
                "光り方をかわいくしたいです",
                "紙箱に入れて机に置きたいです",
                "ここで完成にします",
                "次はサーボで少し動かしたいです",
            ],
            "unknown": [
                "何を足すか迷っています",
                "完成に近づけたいけど方向が決まりません",
            ],
            "problem_report": [
                "ブザーを足したらLEDも消えました",
                "サーボをつないだら電源が落ちます",
                "外装に入れたら線が抜けました",
            ],
            "photo_request": [
                "見た目を写真で相談したいです",
                "ケースに入れた状態を見てほしいです",
            ],
        },
    },
]


ROBUST_REPLY_TEMPLATES = {
    "inventory": {
        "good": [
            ("部品情報を受け取りました", "書いてくれた部品を所持品として扱います。名前があいまいなものは、不足部品と代替候補に分けて見積もります。", "次は最小回路に使うLED、抵抗、GNDだけを確認します。"),
            ("その部品で始められます", "まずは手元にあるものを優先して、買い足しを少なくする構成にします。分からない部品はあとで写真チェックに回せます。", "USBを抜いた状態で、LEDと抵抗を探してください。"),
        ],
        "warn": [
            ("部品名が曖昧でも進めます", "全部の名前が分からなくても大丈夫です。初心者向けのスターター構成として扱い、危ない部品は使わない前提で進めます。", "分かる部品だけ使って、最小構成から始めます。"),
            ("写真確認に回せます", "部品の名前が読めないときは、無理に判断せず画像で確認する方が安全です。今は推定で進めます。", "まずはLED、抵抗、USBケーブルだけを分けてください。"),
        ],
    },
    "gnd": {
        "good": [
            ("GND共有は大丈夫そうです", "ボード、LED、センサーの基準がそろっているので、次はコード側のピン番号確認へ進めます。", "コードのGPIO番号と配線先を合わせます。"),
            ("基準線はそろいました", "GNDが共通になっていると、センサー値やLEDの反応を正しく見られます。ここは大きな山を越えています。", "次は確認用コードを書き込みます。"),
        ],
        "warn": [
            ("GNDを先にそろえましょう", "GNDが別れていると、コードが合っていても反応しません。USBを抜いてから、GND線だけを同じ列へまとめます。", "つなぎ直したら「GNDを同じ列にしました」と答えてください。"),
            ("ここは止まって正解です", "GNDは初心者が一番つまずきやすい場所です。色ではなく、同じ列につながっているかを見ます。", "写真で確認したい場合は、そのまま写真チェックに回せます。"),
        ],
    },
    "led": {
        "good": [
            ("LEDの向きは大丈夫そうです", "長い足が抵抗を通ってGPIO側、短い足がGND側なら、次は抵抗とピン番号を合わせます。", "コード側のGPIO番号と配線先を確認します。"),
            ("LED極性はクリアです", "向きが合っているので、最小回路の確認を前へ進められます。", "確認用コードを書き込んでLEDだけを試します。"),
        ],
        "warn": [
            ("LEDの向きを見直しましょう", "LEDは向きがあります。USBを抜いてから、短い足をGND側、長い足を抵抗側にします。", "直したら「向きを直した」と答えてください。"),
            ("LEDだけで一度確認します", "センサーやブザーを足す前に、LED単体で光るかを見ると原因を絞れます。", "LEDと抵抗だけの最小回路に戻します。"),
        ],
    },
    "resistor": {
        "good": [
            ("抵抗は入っています", "LEDをGPIOへ直接つないでいないので、安全側で進められます。次はコードとピンの一致を見ます。", "GPIO番号をコードと配線でそろえます。"),
            ("LED保護はOKです", "抵抗が直列に入っていれば、まずはLEDだけの動作確認に進めます。", "確認用コードを書き込みます。"),
        ],
        "warn": [
            ("抵抗なしでは進めません", "LEDをGPIOに直結すると故障リスクがあります。220Ω前後の抵抗を一本はさんでください。", "抵抗を入れたら、LEDだけで再確認します。"),
            ("抵抗を探しましょう", "色の帯がある小さい部品が抵抗です。向きはありません。LEDとGPIOの間に入れます。", "分からなければ部品写真チェックに回します。"),
        ],
    },
    "board": {
        "good": [
            ("ボード情報を反映しました", "使うマイコンに合わせて、ピン番号と書き込み手順を変えます。", "次はボード設定とポートを確認します。"),
            ("このボードで進めます", "ボード名が分かると、コードの書き方とピン割り当てを安全に決められます。", "IDEで同じボード名を選んでください。"),
        ],
        "warn": [
            ("ボード名は焦らなくて大丈夫です", "名前が分からない場合は、USB端子の形や基板の印字から推定します。今はスターター構成で進めます。", "写真か印字を見て、ESP32/Arduino/M5Stack/Picoのどれに近いか確認します。"),
            ("認識問題として扱います", "USBを挿しても出ない場合、ボード設定より先にケーブルとドライバを疑います。", "別のUSBケーブルがあれば試してください。"),
        ],
    },
    "usb_port": {
        "good": [
            ("書き込み準備はできています", "ボードとポートが見えているので、確認用コードを書き込んでシリアル出力を見ます。", "書き込み後、シリアルモニタを開いてください。"),
            ("PC接続は大丈夫そうです", "COMポートが出ているなら、次はコードが動いているかをログで確認します。", "起動メッセージが出るか見ます。"),
        ],
        "warn": [
            ("ポート認識を先に直します", "ポートが出ない状態では書き込みできません。充電専用ケーブル、ドライバ、ボード選択を順に確認します。", "まず別のUSBケーブルで試してください。"),
            ("書き込みエラーとして切り分けます", "エラー文が出ているなら、全文ではなく最初の1行だけで原因を絞れます。", "エラーの先頭1行を貼ってください。"),
        ],
    },
    "serial": {
        "good": [
            ("ログが見えています", "起動メッセージや数値が出ているので、コードは動いています。次は値が変わるかを見ます。", "手を近づけたり離したりして値の変化を見ます。"),
            ("観察できる状態です", "シリアルに反応があるなら、作品化へ進む準備ができています。", "次はブザーや外装などを一つだけ足します。"),
        ],
        "warn": [
            ("シリアル出力を先に出しましょう", "何も出ない場合、書き込み、速度設定、Serial.beginのどれかで止まっている可能性があります。", "速度をコードと同じにして、もう一度開きます。"),
            ("値が変わらない原因を絞ります", "ずっと0や同じ値なら、センサー配線かピン番号が合っていない可能性があります。", "センサーのVCC、GND、SIGを確認します。"),
        ],
    },
    "extension": {
        "good": [
            ("作品化の方向が見えました", "拡張は一度に一つだけ足すと、失敗しても戻しやすいです。", "追加前の動くコードを保存してから進めます。"),
            ("次の一機能を決めました", "光、音、動き、外装のうち一つに絞ると、完成率が上がります。", "追加する部品を一つだけ選びます。"),
        ],
        "warn": [
            ("拡張で崩れた状態として扱います", "追加後に動かなくなった場合は、追加した部品だけを外して最小構成へ戻します。", "まず変更前の動く状態に戻しましょう。"),
            ("完成を急がず一つ戻します", "作品化で一気に増やすと原因が追えなくなります。追加は一機能だけにします。", "最後に足した部品を外して確認します。"),
        ],
    },
    "mood": {
        "good": [
            ("作りたい方向が見えました", "まだ曖昧でも大丈夫です。気分から作品候補を3つに絞ります。", "一番おすすめ、安い案、少し挑戦案を出します。"),
        ],
        "warn": [
            ("決まっていなくても始められます", "作りたいものが分からない状態も入力として扱います。予算か持ち物から候補を作ります。", "まずは予算だけ選んで進めます。"),
        ],
    },
    "budget": {
        "good": [
            ("予算を反映しました", "安すぎる見積もりにならないよう、ケーブル、抵抗、ジャンパ線、外装も含めて考えます。", "予算内で最小構成から出します。"),
        ],
        "warn": [
            ("予算未定でも進めます", "0円で試す案、1000円以内の案、5000円以内の案に分けて候補を出します。", "手元部品を優先して見積もります。"),
        ],
    },
}


def generate_records(samples: int, seed: int = 131, robust_ratio: float = 0.7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(samples * 30, 1000)
    while len(records) < samples and attempts < max_attempts:
        attempts += 1
        record = generate_robust_record(rng) if rng.random() < robust_ratio else generate_base_record(rng)
        key = dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    if len(records) < samples:
        raise RuntimeError(f"Could only generate {len(records)} unique tutorial response records out of {samples}.")
    return records


def generate_base_record(rng: random.Random) -> dict[str, Any]:
        project_id = rng.choice(list(PROJECTS))
        current_stage = rng.choice(STAGES[:-1])
        question = question_for_stage(current_stage, rng)
        topic = topic_for_question(question)
        kind = choose_kind(topic, current_stage, rng)
        answer = mutate_answer(answer_for(topic, kind, rng), rng)
        next_stage = choose_next_stage(question, current_stage, kind)
        source = build_source({
            "projectId": project_id,
            "projectTitle": PROJECTS[project_id],
            "currentStage": current_stage,
            "question": question,
            "answer": answer,
            "inventory": rng.choice(INVENTORY_CONTEXTS),
            "budget": rng.choice([0, 1000, 3000, 5000, 10000, 20000]),
            "symptom": symptom_for_kind(kind, answer),
            "skill": rng.choice(["gnd_common:0.20, led_polarity:0.35", "gpio:0.55, serial_monitor:0.30", "debugging:0.25", ""]),
            "previous": rng.choice(["", "前回はLEDの向きで詰まった", "USBケーブルが充電専用だった", "部品名が分からず止まった"]),
        })
        target = build_target(project_id, question, answer, current_stage, next_stage, kind, rng)
        return {
            "source": source,
            "target": target,
            "nextStage": next_stage,
            "kind": target_kind(target),
            "topic": topic,
            "currentStage": current_stage,
            "generator": "base",
        }


def generate_robust_record(rng: random.Random) -> dict[str, Any]:
    scenario = rng.choice(ROBUST_SCENARIOS)
    topic = scenario["topic"]
    project_id = rng.choice(list(PROJECTS))
    current_stage = rng.choice(scenario["stages"])
    question = mutate_question(rng.choice(scenario["questions"]), rng)
    kind = choose_robust_kind(topic, scenario, rng)
    answer = mutate_robust_answer(rng.choice(scenario["answers"][kind]), topic, kind, rng)
    next_stage = choose_next_stage(question, current_stage, kind)
    if current_stage == "extension" and kind == "confirmed" and any(token in answer for token in ["完成", "ここで終わり", "終わり"]):
        next_stage = "completion_log"
    source = build_source({
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": current_stage,
        "question": question,
        "answer": answer,
        "inventory": rng.choice(ROBUST_CONTEXTS["inventories"]),
        "budget": rng.choice([0, 1000, 1500, 3000, 5000, 8000, 10000, 20000]),
        "symptom": symptom_for_kind(kind, answer),
        "skill": rng.choice(ROBUST_CONTEXTS["skills"]),
        "previous": rng.choice(ROBUST_CONTEXTS["previous"]),
    })
    target = build_robust_target(project_id, question, answer, current_stage, next_stage, kind, topic, rng)
    return {
        "source": source,
        "target": target,
        "nextStage": next_stage,
        "kind": target_kind(target),
        "topic": topic,
        "currentStage": current_stage,
        "generator": "robust",
    }


def dedupe_key(record: dict[str, Any]) -> str:
    source = re.sub(r"\s+", " ", record["source"]).strip()
    target = re.sub(r"\s+", " ", record["target"]).strip()
    return f"{source}\n---\n{target}"


def choose_robust_kind(topic: str, scenario: dict[str, Any], rng: random.Random) -> str:
    available = list(scenario["answers"])
    if topic in {"gnd", "led", "resistor", "usb_port", "serial"}:
        preferred = rng.choices(["confirmed", "negative", "unknown", "problem_report", "photo_request"], weights=[34, 24, 18, 16, 8], k=1)[0]
    elif topic == "inventory":
        preferred = rng.choices(["inventory_report", "unknown", "photo_request"], weights=[54, 28, 18], k=1)[0]
    elif topic == "board":
        preferred = rng.choices(["board_report", "unknown", "problem_report"], weights=[55, 24, 21], k=1)[0]
    elif topic == "extension":
        preferred = rng.choices(["confirmed", "unknown", "problem_report", "photo_request"], weights=[45, 25, 20, 10], k=1)[0]
    else:
        preferred = rng.choice(available)
    return preferred if preferred in available else rng.choice(available)


def mutate_question(question: str, rng: random.Random) -> str:
    prefixes = ["", "", "確認です。", "次だけ見たいです。", "いま見えている範囲で、", "焦らなくて大丈夫です。"]
    suffixes = ["", "", "そのまま一行で答えてください。", "分からなければ分からないでOKです。", "写真で見たい場合もそう書いてください。"]
    text = f"{rng.choice(prefixes)}{question}"
    if rng.random() < 0.36:
        text = f"{text} {rng.choice(suffixes)}"
    if rng.random() < 0.12:
        text = text.replace("GND", rng.choice(["GND", "gnd", "グランド"]))
    if rng.random() < 0.1:
        text = text.replace("GPIO", rng.choice(["GPIO", "ピン", "gpio"]))
    return re.sub(r"\s+", " ", text).strip()


def mutate_robust_answer(answer: str, topic: str, kind: str, rng: random.Random) -> str:
    prefixes = [
        "", "", "", "今見た感じ、", "たぶん、", "すみません、", "写真だと、", "さっき直して、",
        "完全には自信ないですが、", "いまの状態は、",
    ]
    suffixes = [
        "", "", "", "このまま進めていいですか？", "次に何を見ればいいですか？",
        "ちょっと不安です。", "たぶん合ってると思います。", "一応そう見えます。",
    ]
    text = f"{rng.choice(prefixes)}{answer}"
    if rng.random() < 0.42:
        text = f"{text}。{rng.choice(suffixes)}"
    if rng.random() < 0.18 and kind in {"confirmed", "unknown"}:
        text = f"{text} でも少し自信ないです"
    if rng.random() < 0.13 and topic in {"usb_port", "serial", "led"}:
        text = f"{text} / 画面には {rng.choice(['start', '0', 'Failed to connect', 'COM3', 'Upload done'])} と出ています"
    if rng.random() < 0.12:
        text = introduce_light_typo(text, rng)
    if rng.random() < 0.08:
        text = text.replace("です", rng.choice(["です", "っす", "ですー"]))
    return re.sub(r"\s+", " ", text).strip()


def introduce_light_typo(text: str, rng: random.Random) -> str:
    replacements = [
        ("つながっています", "つながってます"),
        ("分からない", "わからない"),
        ("シリアル", "serial"),
        ("ポート", "port"),
        ("抵抗", "ていこう"),
        ("ブレッドボード", "ブレボ"),
        ("アップロード", "upload"),
        ("GND", "gnd"),
    ]
    rng.shuffle(replacements)
    for before, after in replacements:
        if before in text:
            return text.replace(before, after, 1)
    if len(text) > 8:
        index = rng.randrange(2, len(text) - 2)
        return text[:index] + text[index + 1 :]
    return text


def build_robust_target(project_id: str, question: str, answer: str, current_stage: str, next_stage: str, kind: str, topic: str, rng: random.Random) -> str:
    style = "good" if kind in {"confirmed", "inventory_report", "board_report"} else "warn" if kind in {"negative", "unknown", "problem_report", "photo_request"} else "info"
    template_group = ROBUST_REPLY_TEMPLATES.get(topic, ROBUST_REPLY_TEMPLATES["mood"])
    template_key = "good" if style == "good" else "warn"
    title, body, instruction = rng.choice(template_group.get(template_key, template_group["warn"]))
    answer_snippet = summarize_answer(answer)
    project_title = PROJECTS[project_id]
    if rng.random() < 0.46:
        body = f"{body} 入力は「{answer_snippet}」として受け取りました。"
    if rng.random() < 0.34:
        body = f"{body} {project_title}では、いきなり完成形にせず最小構成から進めます。"
    if rng.random() < 0.24:
        instruction = f"{instruction} 迷ったら「分からない」と答えて大丈夫です。"
    return f"style:{style}\ntitle:{title}\nbody:{body}\nnext:{instruction}\nstage:{next_stage}"


def question_for_stage(stage: str, rng: random.Random) -> str:
    table = {
        "orient": rng.choice(QUESTION_BANK["mood"] + QUESTION_BANK["budget"]),
        "parts_check": rng.choice(QUESTION_BANK["inventory"]),
        "minimal_circuit": rng.choice(QUESTION_BANK["gnd"] + QUESTION_BANK["led"] + QUESTION_BANK["resistor"]),
        "firmware_upload": rng.choice(QUESTION_BANK["board"] + QUESTION_BANK["usb_port"]),
        "observe_serial": rng.choice(QUESTION_BANK["serial"]),
        "debug_triage": rng.choice(QUESTION_BANK["gnd"] + QUESTION_BANK["led"] + QUESTION_BANK["resistor"] + QUESTION_BANK["usb_port"] + QUESTION_BANK["serial"]),
        "standard_build": rng.choice(QUESTION_BANK["gnd"] + QUESTION_BANK["extension"]),
        "enclosure": rng.choice(QUESTION_BANK["extension"]),
        "extension": rng.choice(QUESTION_BANK["extension"]),
    }
    return table.get(stage, rng.choice(QUESTION_BANK["inventory"]))


def choose_kind(topic: str, stage: str, rng: random.Random) -> str:
    if topic == "inventory":
        return rng.choices(["inventory_report", "unknown", "photo_request"], weights=[62, 25, 13], k=1)[0]
    if topic == "board":
        return rng.choices(["board_report", "unknown", "problem_report"], weights=[66, 18, 16], k=1)[0]
    if topic in {"gnd", "led", "resistor", "usb_port", "serial"}:
        return rng.choices(["confirmed", "negative", "unknown", "photo_request", "problem_report"], weights=[38, 25, 18, 8, 11], k=1)[0]
    if topic == "extension":
        return rng.choices(["confirmed", "unknown", "problem_report", "photo_request"], weights=[54, 25, 13, 8], k=1)[0]
    return rng.choices(["confirmed", "unknown", "problem_report"], weights=[48, 34, 18], k=1)[0]


def answer_for(topic: str, kind: str, rng: random.Random) -> str:
    bank = ANSWER_BANK.get(topic, {})
    if kind in bank:
        return rng.choice(bank[kind])
    if kind == "photo_request":
        return rng.choice(["写真で確認したいです", "画像を送った方がよさそうです", "配線を見てもらえますか"])
    if kind == "problem_report":
        return rng.choice(["動きません", "エラーが出ています", "途中で止まります", "反応がありません"])
    if kind == "negative":
        return rng.choice(["いいえ", "まだできていません", "違うかもしれません"])
    if kind == "unknown":
        return rng.choice(["分からないです", "自信がないです", "どこを見ればいいか分かりません"])
    return rng.choice(["はい", "できました", "確認しました"])


def mutate_answer(answer: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "今の状態は、", "たぶん、", "確認したら、", "すみません、", "写真を見る限り、"])
    suffix = rng.choice(["", "", "。", "、次に何をすればいいですか？", "。ちょっと不安です。", "。このまま進めていいですか？"])
    if rng.random() < 0.12:
        answer = answer.replace("GND", rng.choice(["GND", "gnd", "グランド"]))
    if rng.random() < 0.08:
        answer = answer.replace("ブレッドボード", "ブレボ")
    return f"{prefix}{answer}{suffix}"


def build_source(payload: dict[str, Any]) -> str:
    return "\n".join([
        f"project: {payload.get('projectId')} {payload.get('projectTitle')}",
        f"stage: {payload.get('currentStage')}",
        f"budget: {payload.get('budget')}",
        f"inventory: {payload.get('inventory')}",
        f"symptom: {payload.get('symptom')}",
        f"skill: {payload.get('skill')}",
        f"previous: {payload.get('previous')}",
        f"question: {payload.get('question')}",
        f"answer: {payload.get('answer')}",
        "task: generate empathetic autonomous tutorial response, next instruction, and next stage",
    ])


def build_target(project_id: str, question: str, answer: str, current_stage: str, next_stage: str, kind: str, rng: random.Random) -> str:
    topic = topic_for_question(question)
    project_title = PROJECTS[project_id]
    answer_snippet = summarize_answer(answer)
    style = "good" if kind in {"confirmed", "inventory_report", "board_report"} else "warn" if kind in {"negative", "unknown", "problem_report", "photo_request"} else "info"

    if topic == "gnd":
        if kind == "confirmed":
            title = choose(rng, ["GND共有はクリアです", "GNDは同じ基準で見られています", "GNDラインは大丈夫そうです"])
            body = choose(rng, [
                f"「{answer_snippet}」なら、{project_title}の最小回路は次の確認へ進めます。",
                "GNDがそろっているので、LEDやセンサーの基準電位はひとまず合っています。",
                "ここが合っていると、次はコード側のピン確認に集中できます。",
            ])
            instruction = choose(rng, ["ボードとポートを確認して、シリアル出力つきコードを書き込みます。", "次はGPIO番号とコードの定義を合わせます。", "確認コードを書き込んで start 表示を見ます。"])
        elif kind == "photo_request":
            title = choose(rng, ["写真でGNDを確認しましょう", "配線写真でGNDラインを見ます", "画像チェックに切り替えます"])
            body = choose(rng, [
                "ボードのGNDピン、ブレッドボードのマイナス列、LEDやセンサーのGND側が一枚に入ると判断しやすいです。",
                "真上から撮ると、同じ列かどうかをかなり見分けやすくなります。",
                "線の色より、刺さっている列が同じかを見ます。",
            ])
            instruction = choose(rng, ["USBを抜いてから真上の写真を撮ってください。", "マイコンのGND表示が読める距離で撮ってください。", "写真が難しければ、GND線だけを一本ずつ指で追ってください。"])
        else:
            title = choose(rng, ["GNDを同じ列につなぎ直します", "まずGND共有を直しましょう", "ここはGND確認で止まって正解です"])
            body = choose(rng, [
                "USBを抜いて、LEDの短い足側、センサーのGND、ボードのGNDを同じGND列へまとめます。",
                f"「{answer_snippet}」なら、先にGNDをそろえるのが一番早いです。",
                "GNDが別々だと、コードが合っていても反応しません。",
            ])
            instruction = choose(rng, ["つなぎ直せたら「つなぎ直した」と答えてください。", "まずUSBを抜いて、GND線だけを同じ列へ移してください。", "直したあと、LEDだけの最小確認に戻ります。"])
    elif topic == "inventory":
        if kind in {"unknown", "photo_request"}:
            title = choose(rng, ["部品が分からなくても進めます", "不明な部品はスターター構成で補います", "部品名が曖昧でも大丈夫です"])
            body = choose(rng, [
                "スターター構成として扱い、必要部品込みで見積もります。ここで止まらず最小回路へ進みます。",
                "分かる部品だけ採用し、不足しそうなものはBOMで別枠にします。",
                "写真確認に回せますが、まずはLED・抵抗・USB・ブレッドボード前提で進められます。",
            ])
            instruction = choose(rng, ["LED・抵抗・GNDだけの最小確認から始めます。", "次は最小回路へ進みます。", "部品表では不足候補を明示します。"])
        else:
            title = choose(rng, ["部品情報を受け取りました", "所持部品を反映しました", "この部品で進めます"])
            body = choose(rng, [
                f"「{answer_snippet}」を所持品として扱います。不足分は部品表で分けて出します。",
                "書いてくれた部品から、買うものと後回しでよいものを分けます。",
                f"{project_title}の最小構成に使える部品を優先して使います。",
            ])
            instruction = choose(rng, ["次はUSBを抜いた状態で最小回路を確認します。", "まずLEDと抵抗だけで動作確認します。", "GND共有の確認へ進みます。"])
    elif topic == "board":
        title = choose(rng, ["ボード情報を反映しました", "使うマイコンを受け取りました", "このボード設定で進めます"])
        body = choose(rng, [
            "このボードに合わせてピン番号と確認コードを組み立てます。",
            f"「{answer_snippet}」として扱い、ピン割り当てを安全側で選びます。",
            "ボードが分からない場合は、印字やUSB端子の形も手がかりにします。",
        ])
        instruction = choose(rng, ["次は最小回路とコードのピン番号を合わせます。", "ボード設定とポートを確認します。", "確認コードを書き込む準備に進みます。"])
    elif topic == "resistor":
        if kind == "confirmed":
            title = choose(rng, ["抵抗は入っています", "LED保護はできています", "抵抗の確認はクリアです"])
            body = choose(rng, ["LEDをGPIOへ直結していないので、まず安全な形です。", "抵抗が直列に入っていれば、次はピン番号確認へ進めます。", f"「{answer_snippet}」なら、そのまま次へ進めます。"])
            instruction = choose(rng, ["コードのGPIO番号と配線先を一致させます。", "次は書き込みとシリアル確認へ進みます。", "LEDだけで点灯確認します。"])
        else:
            title = choose(rng, ["抵抗を直列に入れます", "抵抗なしでは進めません", "LED保護を先に入れます"])
            body = choose(rng, ["LEDとGPIOの間に220Ω前後の抵抗を入れてください。抵抗なしの直結は避けます。", "安全のため、まず抵抗を一本はさみます。向きはありません。", "抵抗が見つからなければ、部品表で買い足し候補にします。"])
            instruction = choose(rng, ["抵抗を足したらLEDだけで再確認します。", "USBを抜いて抵抗を追加してください。", "抵抗がない場合は購入リストへ回します。"])
    elif topic == "led":
        if kind == "confirmed":
            title = choose(rng, ["LEDの向きは大丈夫そうです", "LED極性は確認できています", "LED配線は次へ進めそうです"])
            body = choose(rng, ["長い足が抵抗を通ってGPIO側、短い足がGND側なら次へ進めます。", "LEDの向きが合っていれば、次は抵抗とピン番号だけ見ます。", f"「{answer_snippet}」ならLED極性は一旦クリアです。"])
            instruction = choose(rng, ["抵抗とコードのピン番号を確認します。", "確認コードを書き込みます。", "LEDだけで点灯確認します。"])
        else:
            title = choose(rng, ["LEDの向きを直してから進みます", "LED極性を見直します", "LEDの足の向きで止まっています"])
            body = choose(rng, ["LEDは向きがあります。長い足をGPIO側、短い足をGND側にします。", "足を切って分からない場合は、別のLEDで試す方が早いです。", "USBを抜いてからLEDを挿し直します。"])
            instruction = choose(rng, ["USBを抜いて向きを直し、LEDだけで点灯確認します。", "短い足をGND側へ戻します。", "別のLEDで最小確認しても大丈夫です。"])
    elif topic in {"serial", "usb_port"}:
        if kind == "confirmed":
            title = choose(rng, ["ログが見えています", "書き込み確認は進められます", "PCとの接続は大丈夫そうです"])
            body = choose(rng, ["起動メッセージや値が見えているので、コードは最低限動いています。", "ポートとボードが選べているので、次は反応の変化を見ます。", f"「{answer_snippet}」なら次の観察へ進めます。"])
            instruction = choose(rng, ["入力に応じて値が変わるか観察します。", "手を近づける、暗くするなどで反応を見ます。", "標準構成へ進む前にログを一つ保存します。"])
        else:
            title = choose(rng, ["ログを見る準備をします", "PC接続を一つずつ確認します", "書き込み前の接続確認に戻ります"])
            body = choose(rng, ["シリアルモニタを開き、速度をSerial.beginの値に合わせます。", "ポートが見えない場合は、データ通信対応USBケーブルかどうかを確認します。", "書き込みエラーはボード設定、ポート、ケーブルの順に見ます。"])
            instruction = choose(rng, ["起動メッセージが出るかだけ確認します。", "まずポートが見えるか確認します。", "別のUSBケーブルで試せるなら試します。"])
    elif topic == "extension":
        if kind == "problem_report":
            title = choose(rng, ["拡張で崩れた状態として受け取りました", "追加部品で問題が出ています", "一度最小構成へ戻します"])
            body = choose(rng, ["追加した機能を一つ外して、動いていた最小構成に戻します。", "変更点を一つに絞ると復旧しやすいです。", "拡張は一機能ずつ足します。"])
            instruction = choose(rng, ["追加した部品を外して、前の動く状態を確認します。", "変更前コードに戻せるか確認します。", "まずLEDだけで生存確認します。"])
            style = "warn"
        else:
            title = choose(rng, ["次の拡張を決めました", "作品化の方向が見えました", "一機能だけ足して進めます"])
            body = choose(rng, [f"{project_title}に足すなら、今は一機能だけに絞るのが安全です。", "見た目、音、センサーのうち一つだけ足します。", "完成感を出すなら外装か光の演出が効果的です。"])
            instruction = choose(rng, ["追加する部品を一つだけ選びます。", "次は標準構成の配線へ進みます。", "動く状態を保存してから拡張します。"])
            style = "good" if kind == "confirmed" else "info"
    elif kind == "problem_report":
        title = choose(rng, ["動かない状態として受け取りました", "トラブルとして診断します", "復旧優先で進めます"])
        body = choose(rng, ["原因を広げず、物理接続、電源、コードの順に一つずつ潰します。", "まず最後に変更した箇所だけを疑います。", "危険な電源系ではなく、USB給電の範囲で確認します。"])
        instruction = choose(rng, ["まずUSBを抜いてGND共有とピン番号を確認します。", "エラー文があれば一行だけ貼ってください。", "最小構成へ戻して動作を確認します。"])
        style = "warn"
    else:
        title = choose(rng, ["回答を受け取りました", "状況を反映しました", "次の短い作業へ進めます"])
        body = choose(rng, ["今の回答を次の作業に反映します。迷わないように確認点を一つに絞ります。", f"「{answer_snippet}」として受け取りました。", "曖昧なところは安全側に倒して進めます。"])
        instruction = choose(rng, ["次のカードで短い作業だけを進めます。", "一つだけ確認してから先へ進みます。", "分からなければ写真チェックに回せます。"])

    body = enrich_body(body, topic, project_title, current_stage, next_stage, answer_snippet, rng)
    instruction = enrich_instruction(instruction, topic, current_stage, next_stage, rng)
    return f"style:{style}\ntitle:{title}\nbody:{body}\nnext:{instruction}\nstage:{next_stage}"


def enrich_body(body: str, topic: str, project_title: str, current_stage: str, next_stage: str, answer_snippet: str, rng: random.Random) -> str:
    additions: list[str] = []
    if rng.random() < 0.38:
        additions.append(choose(rng, [
            "テスターがなくても、まず目視確認から進めます。",
            "一度に複数箇所を直すと原因が追えないので、ここでは一つだけ見ます。",
            "初心者向けには、先に安全なUSB給電の範囲で確認します。",
            "ここを急がずに見ると、後のデバッグがかなり楽になります。",
            "写真で確認する場合も、この観点で見ると判断しやすいです。",
        ]))
    if rng.random() < 0.3:
        additions.append(choose(rng, [
            f"{project_title}は、最小構成が動いてから作品らしさを足す方が成功しやすいです。",
            f"今は{project_title}の完成形ではなく、動く最小版を守ります。",
            f"この確認は{project_title}の次ステップにも再利用できます。",
        ]))
    if rng.random() < 0.26:
        additions.append(choose(rng, [
            f"入力は「{answer_snippet}」として受け取りました。",
            f"今の回答は{next_stage}へ進めるための材料にします。",
            f"{current_stage}の途中として扱い、次の確認を短くします。",
        ]))
    if rng.random() < 0.22:
        additions.append(topic_hint(topic, rng))
    return " ".join([body, *[item for item in additions if item]]).strip()


def enrich_instruction(instruction: str, topic: str, current_stage: str, next_stage: str, rng: random.Random) -> str:
    additions: list[str] = []
    if rng.random() < 0.28:
        additions.append(choose(rng, [
            "USBを挿したまま配線を差し替えないでください。",
            "作業前にUSBを抜いてから触ります。",
            "終わったら一回だけ確認コードで試します。",
            "うまくいかなければ写真チェックに切り替えます。",
            "答えは短くて大丈夫です。",
        ]))
    if rng.random() < 0.22:
        additions.append(choose(rng, [
            f"次の段階は{next_stage}として扱います。",
            f"{current_stage}には戻らず、確認点を一つ進めます。",
            "ここで結果を制作ログにも残します。",
            "迷ったら「分からない」で進めます。",
        ]))
    if rng.random() < 0.18:
        additions.append(topic_next_hint(topic, rng))
    return " ".join([instruction, *[item for item in additions if item]]).strip()


def topic_hint(topic: str, rng: random.Random) -> str:
    table = {
        "gnd": ["GNDは全部の基準なので、違う列だとLEDもセンサーも反応しにくくなります。", "線の色ではなく、刺さっている列が同じかを見ます。"],
        "led": ["LEDは向きがあるので、足の長さか平らな側を手がかりにします。", "LED単体で確認できると、後のセンサー追加が楽です。"],
        "resistor": ["抵抗は向きがないので、LEDとGPIOの間に入っていれば大丈夫です。", "抵抗なしの直結は避けて、安全側で進めます。"],
        "usb_port": ["ポートが見えない時は、充電専用ケーブルが原因のことがあります。", "ボード設定とポート設定は別々に確認します。"],
        "serial": ["シリアルは、コードが動いているかを見る窓として使います。", "文字化けする時は速度設定を疑います。"],
        "inventory": ["部品名が曖昧でも、不足候補として扱えば進められます。", "分かる部品だけで最小構成を作ります。"],
        "extension": ["拡張は一度に一つだけ足すと、失敗しても戻しやすいです。", "作品らしさは外装・音・光のどれか一つから足します。"],
    }
    return choose(rng, table.get(topic, ["ここでは一つの確認に絞ります。"]))


def topic_next_hint(topic: str, rng: random.Random) -> str:
    table = {
        "gnd": ["直せたら「GNDつなぎ直した」と返してください。", "写真で見るなら真上から撮ります。"],
        "led": ["次はLEDだけで点灯確認します。", "向きが不安なら別のLEDで試しても大丈夫です。"],
        "resistor": ["抵抗がなければ購入リストに回します。", "入れたらGPIO番号の確認へ進みます。"],
        "usb_port": ["ポートが出たら書き込み確認へ進みます。", "別ケーブルがあれば交換して試します。"],
        "serial": ["値が出たら、手を近づけて変化を見る段階です。", "何も出なければポートと速度を見直します。"],
        "inventory": ["足りないものはBOMに分けて出します。", "次は最小回路の確認へ進みます。"],
        "extension": ["変更前の動く状態を残してから足します。", "次は追加部品を一つだけ選びます。"],
    }
    return choose(rng, table.get(topic, ["次の確認へ進みます。"]))


def topic_for_question(question: str) -> str:
    if any(token in question for token in ["部品", "手元", "持っている", "所持", "机の上"]):
        return "inventory"
    if any(token in question for token in ["GND", "グランド", "マイナス列", "GNDライン"]):
        return "gnd"
    if "抵抗" in question:
        return "resistor"
    if "LED" in question:
        return "led"
    if any(token in question for token in ["ポート", "USB", "書き込み", "ケーブル"]):
        return "usb_port"
    if any(token in question for token in ["ボード", "マイコン", "Arduino", "ESP32", "M5Stack", "Pico"]):
        return "board"
    if any(token in question for token in ["シリアル", "値", "起動メッセージ", "Serial"]):
        return "serial"
    if any(token in question for token in ["見た目", "音", "センサー追加", "外装", "ログ保存"]):
        return "extension"
    if any(token in question for token in ["予算", "何円"]):
        return "budget"
    return "mood"


def choose(rng: random.Random, values: list[str]) -> str:
    return rng.choice(values)


def summarize_answer(answer: str) -> str:
    text = re.sub(r"\s+", " ", answer).strip()
    return text[:46] + ("..." if len(text) > 46 else "")


def choose_next_stage(question: str, current_stage: str, kind: str) -> str:
    topic = topic_for_question(question)
    if topic == "inventory":
        return "minimal_circuit"
    if topic == "gnd":
        return "firmware_upload" if kind == "confirmed" else "minimal_circuit"
    if topic in {"led", "resistor"}:
        return "firmware_upload" if kind == "confirmed" else "minimal_circuit"
    if topic in {"board", "usb_port"}:
        return "observe_serial" if kind in {"confirmed", "board_report"} else "debug_triage"
    if topic == "serial":
        return "standard_build" if kind == "confirmed" else "debug_triage"
    if kind in {"problem_report", "photo_request"}:
        return "debug_triage"
    order = STAGES
    if kind == "confirmed" and current_stage in order and order.index(current_stage) < len(order) - 1:
        return order[order.index(current_stage) + 1]
    return current_stage


def symptom_for_kind(kind: str, answer: str) -> str:
    if kind != "problem_report":
        return "none"
    lowered = answer.lower()
    if "led" in lowered or "光" in answer:
        return "led_not_lighting"
    if "upload" in lowered or "ポート" in answer or "書き込" in answer or "connect" in lowered:
        return "upload_failed"
    if "値" in answer or "sensor" in lowered or "センサー" in answer:
        return "sensor_static"
    if "checkpoint loaded" in lowered or "止ま" in answer:
        return "app_stuck_after_checkpoint"
    return "unknown"


def target_kind(target: str) -> str:
    for line in target.splitlines():
        if line.startswith("style:"):
            return line.split(":", 1)[1].strip()
    return "info"


def collect_texts(records: list[dict[str, Any]]) -> list[str]:
    return [item["source"] for item in records] + [item["target"] for item in records]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TutorialResponseDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_source_length: int, max_target_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        source_ids, source_mask = self.tokenizer.encode(record["source"], self.max_source_length, "<query>")
        target_ids, target_mask = self.tokenizer.encode(record["target"], self.max_target_length, "<log>")
        decoder_input = target_ids[:-1]
        labels = target_ids[1:]
        label_mask = target_mask[1:]
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.bool),
            "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
        }


def collate_response_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}
