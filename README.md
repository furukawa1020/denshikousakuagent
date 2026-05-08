# Lチカのつづき / MakerGraph AI

電子工作初心者の曖昧な制作意図を、作品候補・部品表・回路・コード・デバッグ・次作品推薦へつなげるプロトタイプです。

## Deep Learning Core

`ai_models/makergraph` に実際の PyTorch Transformer を入れています。

- `MakerIntentTransformer`: ユーザー発話と作品/部品/回路/失敗ログ文書を同じ埋め込み空間へ写像
- `MakerGraphDualEncoder`: query tower と project tower を対照学習する二塔型推薦モデル
- `ProjectGraphSequenceTransformer`: intent から graph token を生成する Transformer decoder の土台
- `NeuralBOMEstimator`: 作品グラフから必要部品のマルチラベル推定と価格分位点を出すTransformer
- `WireCheckNet`: 配線写真を部品bbox・ジャンパ線端点・危険ミスへ変換するCNN/ViT + Transformer decoder
- Loss: symmetric InfoNCE + difficulty/budget/novelty profile regression
- CUDA: `--device cuda --amp` で mixed precision training

## Local RTX 5060 Setup

Windows + RTX 5060では、NVIDIAドライバが新しければPyTorchのCUDA 13.0 wheelを使えます。

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install pillow
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Dataset Generation

seedだけでは小さすぎるので、制作意図・予算・所持部品・安全制約を組み合わせた合成データを生成します。

```bash
python -m ai_models.makergraph.generate_synthetic_intent_data \
  --output data/intent_training_synthetic.jsonl \
  --records-per-project 160
```

## Colab / CUDA Training

Colabで `notebooks/maker_intent_transformer_colab.ipynb` を開き、ランタイムをGPUにしてください。

```bash
pip install -r requirements-ml.txt
python -m ai_models.makergraph.train_intent_encoder \
  --data data/intent_training_synthetic.jsonl \
  --output runs/maker_intent_transformer \
  --epochs 40 \
  --batch-size 5 \
  --device cuda \
  --amp
```

Project Graph Generator:

```bash
python -m ai_models.makergraph.train_project_graph_generator \
  --data data/intent_training_synthetic.jsonl \
  --output runs/project_graph_transformer \
  --epochs 40 \
  --batch-size 8 \
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

評価:

```bash
python -m ai_models.makergraph.evaluate_retrieval \
  --model-dir runs/maker_intent_transformer \
  --data data/intent_training_synthetic.jsonl \
  --device cuda \
  --top-k 3
```

Graph生成:

```bash
python -m ai_models.makergraph.generate_project_graph \
  --model-dir runs/project_graph_transformer \
  --text "植物を枯らすので水やり通知を作りたい。自動ポンプは怖い。" \
  --device cuda
```

WireCheckNet:

```bash
python -m ai_models.wirecheck.train_wirecheck \
  --output runs/wirechecknet \
  --samples 8000 \
  --epochs 25 \
  --batch-size 32 \
  --device cuda \
  --amp
```

8GB VRAMで重い場合は `--batch-size 16` に下げます。まず動作確認だけなら:

```bash
python -m ai_models.wirecheck.train_wirecheck \
  --output runs/wirechecknet_smoke \
  --samples 256 \
  --epochs 2 \
  --batch-size 8 \
  --device cuda \
  --amp
```

WireCheckNet推論:

```bash
python -m ai_models.wirecheck.infer_wirecheck \
  --model-dir runs/wirechecknet \
  --synthetic \
  --device cuda
```

Neural BOM Estimator:

```bash
python -m ai_models.bom.generate_synthetic_bom_data \
  --output data/bom_training_gpu.jsonl \
  --records-per-project 600

python -m ai_models.bom.train_bom_estimator \
  --data data/bom_training_gpu.jsonl \
  --output runs/bom_estimator \
  --epochs 24 \
  --batch-size 32 \
  --device cuda \
  --amp

python -m ai_models.bom.infer_bom \
  --model-dir runs/bom_estimator \
  --text "水やり通知を作りたい。ESP32とLEDと抵抗は持っている。予算5000円。" \
  --device cuda
```

## Local API

標準ライブラリだけで起動できます。学習済みチェックポイントが `runs/maker_intent_transformer/best.pt` にある場合は、APIからTransformer推論も呼べます。

```bash
python backend/server.py
```

ローカルGPU学習済みモデルをAPIから使う場合は、PyTorchを入れた仮想環境で起動します。

```bash
.\.venv\Scripts\python.exe backend/server.py
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

```bash
curl -X POST http://127.0.0.1:8765/api/ai/project-graph/transformer \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"ESP32を持っている。かわいい机上作品を作りたい\"}"
```

```bash
curl -X POST http://127.0.0.1:8765/api/ai/wirechecknet \
  -H "Content-Type: application/json" \
  -d "{\"synthetic\":true}"
```

```bash
curl -X POST http://127.0.0.1:8765/api/ai/bom/estimator \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"水やり通知を作りたい。ESP32とLEDと抵抗は持っている。予算5000円。\"}"
```
