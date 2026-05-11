# External Publish

今のアプリ構成は崩しません。GPU推論バックエンドは `127.0.0.1:8765` だけで待ち受け、外部公開はHTTPSリバースプロキシまたはTunnelで行います。

## まず確認

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_public_readiness.ps1
```

`cudaAvailable: true`、`allCheckpointsPresent: true` が外部公開前の最低条件です。

## 方式A: 独自ドメイン + Caddy

一番本番らしい構成です。

### 1. GPUバックエンドをローカル限定で起動

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_public_gpu_backend.ps1
```

この起動では `HOST=127.0.0.1` にしているので、8765番を直接インターネットへ晒しません。

### 2. Caddyを入れる

```powershell
winget install CaddyServer.Caddy
```

### 3. DNSとルーターを設定

- ドメインのAレコードを自宅/サーバのグローバルIPへ向ける
- ルーターでTCP `80` と `443` をこのPCへポート転送する
- Windows Defender FirewallでTCP `80` と `443` を許可する

### 4. Caddyfileを作る

`deploy/Caddyfile.public.example` をコピーして、以下を置き換えます。

- `your-domain.example`
- `your-email@example.com`

必要ならBasic認証も有効にします。

```powershell
caddy hash-password --plaintext "your-strong-password"
```

生成されたhashをCaddyfileの `basic_auth` に貼ります。

### 5. HTTPSで公開

```powershell
caddy run --config deploy\Caddyfile.public
```

確認:

```powershell
Invoke-RestMethod https://your-domain.example/api/health
```

## 方式B: Tunnel

ルーターのポート開放ができない場合はTunnelを使います。

代表例:

- Tailscale Funnel
- Cloudflare Tunnel
- ngrok

どれを使っても、転送先は同じです。

```text
http://127.0.0.1:8765
```

## 公開時の注意

- `http://グローバルIP:8765` のように8765番を直接公開しない
- 公開前にBasic認証かアクセス制限を入れる
- `runs/` はGit管理外なので、GPUサーバへ移す時は必ずコピーする
- 電源断で止まるので、長期運用はGPU VPSまたは常時起動PCにする
