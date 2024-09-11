# sbv2-api

このプロジェクトは Style-Bert-ViTS2 を ONNX 化したものを Rust で実行するのを目的としています。

学習したい場合は、Style-Bert-ViTS2 学習方法 などで調べるとよいかもしれません。

JP-Extra しか対応していません。(基本的に対応する予定もありません)

## ONNX 化する方法

```sh
cd convert
# (何かしらの方法でvenv作成(推奨))
pip install -r requirements.txt
python convert_deberta.py
python convert_model.py --style_file ../../style-bert-vits2/model_assets/something/style_vectors.npy --config_file ../../style-bert-vits2/model_assets/something/config.json --model_file ../../style-bert-vits2/model_assets/something/something_eXXX_sXXXX.safetensors
```

## Todo

- [x] WebAPI の実装
- [x] Rust ライブラリの実装
- [ ] 余裕があれば PyO3 使って Python で利用可能にする
- [x] GPU 対応(優先的に CUDA)
- [ ] WASM 変換(ort がサポートやめたので、中止)

## 構造説明

- `sbv2_api` - 推論用 REST API
- `sbv2_core` - 推論コア部分
- `docker` - docker ビルドスクリプト

## API の起動方法

```sh
cargo run -p sbv2_api -r
```

### CUDA での起動

```sh
cargo run -p sbv2_api -r -F cuda,cuda_tf32
```

### Dynamic Link サポート

```sh
ORT_DYLIB_PATH=./libonnxruntime.dll cargo run -p sbv2_api -r -F dynamic
```

### models をインストール

https://huggingface.co/googlefan/sbv2_onnx_models/tree/main
を models フォルダとして配置

### .env ファイルの作成

```sh
cp .env.sample .env
```

### テストコマンド

```sh
curl -XPOST -H "Content-type: application/json" -d '{"text": "こんにちは","ident": "tsukuyomi"}' 'http://localhost:3000/synthesize' --output "output.wav"
curl http://localhost:3000/models
```

## 謝辞

- [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) - このコードの書くにあたり、ベースとなる部分を参考にさせていただきました。
- [Googlefan](https://github.com/Googlefan256) - 彼にモデルを ONNX ヘ変換および効率化をする方法を教わりました。
