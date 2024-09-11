# sbv2-api
このプロジェクトはStyle-Bert-ViTS2をONNX化したものをRustで実行するのを目的としています。

学習したい場合は、Style-Bert-ViTS2 学習方法 などで調べるとよいかもしれません。

JP-Extraしか対応していません。(基本的に対応する予定もありません)

## ONNX化する方法
```sh
cd convert
# (何かしらの方法でvenv作成(推奨))
pip install -r requirements.txt
python convert_deberta.py
python convert_model.py --style_file ../../style-bert-vits2/model_assets/something/style_vectors.npy --config_file ../../style-bert-vits2/model_assets/something/config.json --model_file ../../style-bert-vits2/model_assets/something/something_eXXX_sXXXX.safetensors
```

## Todo
- [x] WebAPIの実装
- [x] Rustライブラリの実装
- [ ] 余裕があればPyO3使ってPythonで利用可能にする
- [x] GPU対応(優先的にCUDA)
- [ ] WASM変換(ortがサポートやめたので、中止)

## 構造説明
- `sbv2_api` - 推論用 REST API
- `sbv2_core` - 推論コア部分
- `docker` - dockerビルドスクリプト

## APIの起動方法
```sh
cargo run -p sbv2_api -r
```

### CUDAでの起動
```sh
cargo run -p sbv2_api -r -F cuda,cuda_tf32
```

### Dynamic Linkサポート
```sh
ORT_DYLIB_PATH=./libonnxruntime.dll cargo run -p sbv2_api -r -F dynamic
```

### テストコマンド
```sh
curl -XPOST -H "Content-type: application/json" -d '{"text": "こんにちは","ident": "something"}' 'http://localhost:3000/synthesize'
curl http://localhost:3000/models
```

## 謝辞
- [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) - このコードの書くにあたり、ベースとなる部分を参考にさせていただきました。
- [Googlefan](https://github.com/Googlefan256) - 彼にモデルをONNXヘ変換および効率化をする方法を教わりました。
