# Easy External Share

今のアプリ構成は変えません。RTX 5060のGPU推論はこのPCで動かしたまま、Tunnelで外部URLだけ発行します。

## 一番ラク: Cloudflare Quick Tunnel

特徴:

- アカウントなしで試せる
- DNS設定なし
- ルーターのポート開放なし
- URLは一時的な `https://*.trycloudflare.com`

手順:

```powershell
winget install Cloudflare.cloudflared
powershell -ExecutionPolicy Bypass -File scripts\share_trycloudflare.ps1 -StartBackend
```

表示された `https://...trycloudflare.com` を共有します。

## Tailscale Funnel

Tailscaleを使っているならこれも楽です。公開URLは `*.ts.net` になります。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\share_tailscale_funnel.ps1 -StartBackend
```

## 注意

- どちらもGPU処理はこのPCで実行されます。
- PCをスリープすると止まります。
- Quick TunnelのURLは一時URLです。固定URLが必要ならCloudflare Tunnelのnamed tunnel、Tailscale Funnel、またはAWS GPUへ移します。
- 公開URLを誰でも叩けるので、長時間公開する場合はBasic認証やアクセス制限を追加してください。
