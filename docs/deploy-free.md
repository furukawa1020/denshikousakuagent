# Free Deployment

ローカルPCを使わず、無料で外部公開するなら Hugging Face Spaces の CPU Basic が一番現実的です。

## できること

- 今の画面とAPIをそのままDocker Spaceとして公開
- 無料CPU Basicで起動
- GPUがなくてもバックエンド側でCPUへ自動降格
- `runs/` のチェックポイント込みでデプロイ

## できないこと

- 無料で常時GPUを占有すること
- 今のDockerアプリをそのままHugging Face ZeroGPUへ載せること

ZeroGPUは無料GPUアクセスがありますが、ホスト側はGradio SDK前提で、個人アカウントで自分のZeroGPU Spaceを作るにはPROが必要です。今の構成を崩さないならCPU Basicが素直です。

## 手順

### 1. Spaceを作る

Hugging Faceで新しい Space を作ります。

- SDK: Docker
- Hardware: CPU Basic
- Visibility: Public または Private

### 2. デプロイ用フォルダを作る

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_hf_space.ps1
```

`dist/hf-space` が作られます。

### 3. Spaceへpush

```powershell
cd dist\hf-space
git init
git lfs install
git lfs track "*.pt"
git add .
git commit -m "Deploy L Chika No Tsuzuki"
git remote add origin https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
git push -u origin main
```

## 確認

SpaceのURLを開くか、以下を叩きます。

```powershell
Invoke-RestMethod https://YOUR_USER-YOUR_SPACE.hf.space/api/health
```

CPU Basicなので、`cudaAvailable` は `false` で正常です。

## 注意

- 初回ビルドは時間がかかります。
- 無料CPU Spaceは無操作が続くとスリープします。
- GPU速度で見せたい場合は有料GPUへ上げるか、RunPod/AWSへ移します。
