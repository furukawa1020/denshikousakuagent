# Lチカのつづき / MakerGraph AI

電子工作初心者の曖昧な制作意図を、作品候補・部品表・回路・コード・デバッグ・次作品推薦へつなげるプロトタイプです。

## Deep Learning Core

`ai_models/makergraph` に実際の PyTorch Transformer を入れています。

- `MakerIntentTransformer`: ユーザー発話と作品/部品/回路/失敗ログ文書を同じ埋め込み空間へ写像
- `MakerGraphDualEncoder`: query tower と project tower を対照学習する二塔型推薦モデル
- `ProjectGraphSequenceTransformer`: intent から graph token を生成する Transformer decoder の土台
- Loss: symmetric InfoNCE + difficulty/budget/novelty profile regression
- CUDA: `--device cuda --amp` で mixed precision training

## Colab / CUDA Training

Colabで `notebooks/maker_intent_transformer_colab.ipynb` を開き、ランタイムをGPUにしてください。

```bash
pip install -r requirements-ml.txt
python -m ai_models.makergraph.train_intent_encoder \
  --data data/intent_training_seed.jsonl \
  --output runs/maker_intent_transformer \
  --epochs 40 \
  --batch-size 5 \
  --device cuda \
  --amp
```

推論:

```bash
python -m ai_models.makergraph.infer \
  --model-dir runs/maker_intent_transformer \
  --text "作りたいものは分からない。予算は5000円。かわいいものがいい。" \
  --device cuda \
  --top-k 3
```

## Local API

標準ライブラリだけで起動できます。学習済みチェックポイントが `runs/maker_intent_transformer/best.pt` にある場合は、APIからTransformer推論も呼べます。

```bash
python backend/server.py
```

Health check:

```bash
curl http://127.0.0.1:8765/api/health
```

Transformer endpoint:

```bash
curl -X POST http://127.0.0.1:8765/api/ai/intent/transformer \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"ESP32を持っている。かわいい机上作品を作りたい\", \"topK\":3}"
```
