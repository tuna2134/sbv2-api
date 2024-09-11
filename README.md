# sbv2-api
このプロジェクトはStyle-Bert-ViTS2をONNX化したものをRustで実行するのを目的としています。つまり推論しか行いません。

学習したいのであれば、Style-Bert-ViT2で調べてやってください。

注意：JP-Extraしか対応していません。

## ONNX化する方法
dabertaとstbv2本体をonnx化する必要があります。

あくまで推奨ですが、onnxsimを使うことをお勧めします。
onnxsim使うことでモデルのサイズを軽くすることができます。

## onnxモデルの配置方法
- `models/daberta.onnx` - DaBertaのonnxモデル
- `models/sbv2.onnx` - `Style-Bert-ViT2`の本体

## Todo
- [x] WebAPIの実装
- [x] Rustライブラリの実装
- [ ] 余裕があればPyO3使ってPythonで利用可能にする
- [ ] GPU対応(優先的にCUDA)
- [ ] WASM変換

## ディレクトリー説明
- `sbv2_api` - Style-Bert-VITS2の推論Web API
- `sbv2_core` - Style-Bert-VITS2の推論コア部分

## APIの起動方法
```bash
cargo run -p sbv2_api -r
```

### CUDAでの起動
```bash
cargo run -p sbv2_api -r -F cuda,cuda_tf32
```

### テストコマンド
```bash
curl -XPOST -H "Content-type: application/json" -d '{"text": "こんにちは"}' 'http://localhost:3000/synthesize'
```

## 謝辞
- [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) - このコードの書くにあたり、ベースとなる部分を参考にさせていただきました。
- [Googlefan](https://github.com/Googlefan256) - 彼にモデルをONNXヘ変換および効率化をする方法を教わりました。