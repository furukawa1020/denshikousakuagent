# RunPod Deploy

ローカルPCを公開せず、GPUクラウド上でそのまま動かす一番簡単な構成です。

## 何がそのままか

- 画面: `frontend/index.html`
- API: `backend/server.py`
- モデル: `runs/*/best.pt`
- ポート: `8765`

RunPod PodのHTTP Proxyで `8765` を公開します。

## 1. Dockerイメージを作ってpush

`runs/` もイメージに含めます。モデル込みなので、push先はprivate registry推奨です。

例: GitHub Container Registry

```powershell
docker login ghcr.io
powershell -ExecutionPolicy Bypass -File scripts\build_runpod_image.ps1 -Image ghcr.io/YOUR_USER/denshi-craft-agent:gpu
```

## 2. RunPodでPod作成

RunPodで新しいPodを作成します。

- Image: `ghcr.io/YOUR_USER/denshi-craft-agent:gpu`
- GPU: RTX 4090 / A40 / A5000 / A6000 / L40S など
- Expose HTTP Ports: `8765`
- Container Disk: 20GB以上推奨

RunPodのHTTP Proxy URLは次の形式です。

```text
https://<POD_ID>-8765.proxy.runpod.net
```

## 3. 確認

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_remote.ps1 -BaseUrl https://<POD_ID>-8765.proxy.runpod.net
```

`cuda: true`、`checkpoints` が全部 `true` なら成功です。

## 注意

- HTTP ProxyはWebアプリ/API向けで、自動HTTPS URLが出ます。
- RunPodのProxyにはタイムアウトがあるため、長時間処理は避け、今のような短い推論レスポンスに向いています。
- モデルをイメージに含めるので、公開registryではなくprivate推奨です。
