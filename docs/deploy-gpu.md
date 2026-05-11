# GPU Deployment

今のアプリ構造はそのままです。`backend/server.py` がフロント配信とAPIを担当し、`runs/` にある学習済みチェックポイントをCUDAで読み込みます。

## 推奨構成

- フロント: `frontend/index.html`
- API: `backend/server.py`
- GPU推論: PyTorch CUDA 13.0 wheel
- モデル: `runs/*/best.pt`
- 公開: そのままLAN公開、またはCaddy/Nginx/Tailscale/Cloudflare TunnelなどでHTTPS化

## ローカルRTX 5060で直接起動

```powershell
.\scripts\start_gpu_backend.ps1 -HostName 0.0.0.0 -Port 8765
```

確認:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/health
powershell -ExecutionPolicy Bypass -File scripts\smoke_agent_pipeline.ps1
```

## Docker ComposeでGPU起動

前提:

- NVIDIAドライバ
- Docker
- NVIDIA Container Toolkit
- `runs/` ディレクトリが存在すること

起動:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\docker_gpu_up.ps1 -Build
```

確認:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_agent_pipeline.ps1
```

Docker Composeでは `compose.gpu.yml` の `deploy.resources.reservations.devices` でNVIDIA GPUを予約します。`runs/` はイメージへ焼き込まず、`/app/runs:ro` としてマウントします。

## 外部公開

HTTPSで公開したい場合は、GPUサーバ上でAPIを `127.0.0.1:8765` に立て、Caddyなどでリバースプロキシします。

```caddyfile
your-domain.example {
  reverse_proxy 127.0.0.1:8765
}
```

例は `deploy/Caddyfile.example` にあります。

## 重要

`runs/` は `.gitignore` 対象です。別マシンへ持っていく場合は、リポジトリとは別に `runs/` をコピーしてください。これがないと `/api/health` の `runtime.checkpoints` が `false` になります。
