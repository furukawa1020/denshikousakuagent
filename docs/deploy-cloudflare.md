# Cloudflare Deploy

この構成では、Cloudflare Pages が `frontend/` を配信し、Cloudflare Worker が `/api/*` をGPUバックエンドへ中継します。深層学習モデルは `runs/` のチェックポイントを使うため、GPU推論はRTX 5060搭載PC、または別GPUサーバで動かします。

## 1. GPUバックエンドを起動

```powershell
.\scripts\start_gpu_backend.ps1 -HostName 0.0.0.0 -Port 8765
```

外部公開する場合は Cloudflare Tunnel などで `http://localhost:8765` をHTTPSのURLへつなぎます。例は `cloudflare/tunnel-config.example.yml` にあります。

## 2. WorkerにGPU API URLを設定

```powershell
npm install
npx wrangler secret put GPU_API_BASE
```

値には `https://api.example.com` のようなGPUバックエンドのベースURLを入れます。

## 3. Workerをデプロイ

```powershell
npm run deploy:worker
```

## 4. Pagesをデプロイ

```powershell
npm run deploy:pages
```

同一ドメインで使う場合は、Pagesのドメインに対してWorker routeを `/api/*` で割り当てます。別ドメインで使う場合は、フロント側で以下のようにAPIベースを指定できます。

```javascript
localStorage.setItem("DENSHI_API_BASE", "https://denshi-craft-agent-api.YOUR_SUBDOMAIN.workers.dev");
```

## 5. 動作確認

```powershell
Invoke-RestMethod https://YOUR_WORKER_DOMAIN/api/health
```

`gpuApiConfigured: true` が返れば、Worker側の中継設定は入っています。
