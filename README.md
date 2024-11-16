# CogVideoX Factory 🧪

24GB以下のGPUメモリでCogファミリーのビデオモデルをカスタムビデオ生成用にファインチューニング ⚡️📼

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">お使いのブラウザはビデオタグをサポートしていません。</video></td>
</tr>
</table>

## クイックスタート

リポジトリをクローンし、必要な要件がインストールされていることを確認します：`pip install -r requirements.txt` およびdiffusersをソースからインストール `pip install git+https://github.com/huggingface/diffusers`

次にデータセットをダウンロードします：

```bash
# `huggingface_hub`をインストール
huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney
```

次に、テキストから動画へのLoRAファインチューニングを開始します（ハイパーパラメータ、データセットのルート、その他の設定オプションは必要に応じて変更してください）：

```bash
# テキストから動画へのCogVideoXモデルのLoRAファインチューニング用
./train_text_to_video_lora.sh

# テキストから動画へのCogVideoXモデルの完全ファインチューニング用
./train_text_to_video_sft.sh

# 画像から動画へのCogVideoXモデルのLoRAファインチューニング用
./train_image_to_video_lora.sh
```

LoRAがHF Hubに保存・プッシュされ、`my-awesome-name/my-awesome-lora`という名前が付けられているとします。ファインチューニングされたモデルを使って推論を行うことができます：

```diff
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name="cogvideox-lora")
+ pipe.set_adapters(["cogvideox-lora"], [1.0])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4", fps=8)
```

マルチ解像度ビデオでトレーニングされた画像から動画へのLoRAの場合、以下の行も追加する必要があります（詳細は[この](https://github.com/a-r-r-o-w/cogvideox-factory/issues/26)Issueを参照）：

```python
from diffusers import CogVideoXImageToVideoPipeline

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
).to("cuda")

# ...

del pipe.transformer.patch_embed.pos_embedding
pipe.transformer.patch_embed.use_learned_positional_embeddings = False
pipe.transformer.config.use_learned_positional_embeddings = False
```

LoRAが正しくマウントされているかどうかは[こちら](tests/test_lora_inference.py)で確認できます。

以下では、このリポジトリで探求された追加のオプションについて詳しく説明します。これらはすべて、メモリ要件を可能な限り削減することで、ビデオモデルのファインチューニングをできるだけアクセスしやすくすることを試みています。

## データセットの準備とトレーニング

トレーニングを開始する前に、[データセット仕様](assets/dataset.md)に従ってデータセットが準備されているかどうかを確認してください。[CogVideoXモデルファミリー](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)に対応したテキストから動画生成および画像から動画生成に適したトレーニングスクリプトを提供しています。トレーニングは、トレーニングしたいタスクに応じて`train*.sh`スクリプトを使用して開始できます。テキストから動画へのLoRAファインチューニングを例に取ります。

[以下、環境設定やトレーニングの詳細な手順が続きます...]

## メモリ要件

[メモリ要件の詳細な表とグラフが続きます...]

## TODO

- [x] スクリプトをDDPと互換性のあるようにする
- [ ] スクリプトをFSDPと互換性のあるようにする
- [x] スクリプトをDeepSpeedと互換性のあるようにする
- [ ] vLLM搭載のキャプショニングスクリプト
- [x] `prepare_dataset.py`でのマルチ解像度/フレームサポート
- [ ] 潜在的な高速化のためのトレース分析と可能な限り多くの同期の削除
- [ ] QLoRA（優先）およびその他の高使用LoRA手法のサポート
- [x] bitsandbytesのメモリ効率の良いオプティマイザでスクリプトをテスト
- [x] CPUOffloadOptimizer等でスクリプトをテスト
- [ ] torchao量子化と低ビットメモリオプティマイザでスクリプトをテスト（現在AdamW (8/4-bit torchao)でエラー）
- [ ] AdamW (8-bit bitsandbytes) + CPUOffloadOptimizer（勾配オフロード付き）でスクリプトをテスト（現在エラー発生）
- [ ] [Sage Attention](https://github.com/thu-ml/SageAttention)（後方パスのサポートのため著者と協力し、A100向けに最適化）

> [!IMPORTANT]
> スクリプトをできるだけメモリフレンドリーにすることが目標であるため、マルチGPUトレーニングは保証していません。
