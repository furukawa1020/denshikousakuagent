# Railway / AWS Deployment

## 結論

- Railway: CPUデモなら可能。GPU/CUDA本番には向かない。
- AWS: GPU本番ならEC2 GPUが一番まっすぐ。ECS GPUも可能。

今のアプリ構成は崩しません。フロントは既存の `frontend/index.html`、APIは `backend/server.py` のままです。

## Railwayでできること

RailwayはDockerfileデプロイ、Config as Code、CLI deployに対応しています。公式ドキュメント上、通常のサービスはCPU/RAM/Volume中心のリソースとして説明されています。自前CUDA GPUを割り当てる前提の記載は見当たらないため、RailwayはCPUデモ用途にします。

追加済み:

- `Dockerfile.railway`
- `requirements-cpu.txt`
- `railway.toml`

注意:

- Railwayでは `runs/` が必要です。
- `.gitignore` では `runs/` を除外しているため、RailwayへGitHub連携するだけではモデルが入りません。
- `railway up --ignore-gitignore` でアップロードするか、別途モデル配布方法を用意してください。
- フロントが `device: cuda` を送っても、GPUがない環境ではバックエンド側でCPUへ降格します。

### Railway CLI

```powershell
npm install -g @railway/cli
railway login
railway init
railway up --ignore-gitignore
```

Railwayが `railway.toml` を読み、`Dockerfile.railway` でCPU版を起動します。

## AWSでGPU本番

一番簡単なのはEC2 GPUインスタンスにこのリポジトリと `runs/` を置き、Docker Composeで起動する方法です。

推奨候補:

- g4dn.xlarge: NVIDIA T4、軽量推論向け
- g5.xlarge: NVIDIA A10G、余裕あり
- g6系: 新しめのNVIDIA GPU、対応リージョンなら候補

### EC2手順

1. Ubuntu 24.04 または AWS Deep Learning AMI のGPUインスタンスを作る
2. Security GroupでTCP `80` / `443` を許可する
3. リポジトリと `runs/` を配置する
4. Docker/NVIDIA Container Toolkitを入れる
5. 起動する

```bash
docker compose -f compose.gpu.yml up --build
```

確認:

```bash
curl http://127.0.0.1:8765/api/health
```

### HTTPS

Caddyを使う場合:

```bash
caddy run --config deploy/Caddyfile.public
```

`deploy/Caddyfile.public.example` の `your-domain.example` と `your-email@example.com` を置き換えてください。

## AWS ECS GPU

ECSでもGPUタスクは可能です。AWS公式では、GPU対応EC2 container instance、ECS GPU-optimized AMI、タスク定義の `resourceRequirements` でGPU数を指定する方式が説明されています。

ただし初回はECSよりEC2単体のほうが速いです。ECS化は、常時運用、ローリングデプロイ、ログ監視、複数台運用が必要になってからで十分です。
