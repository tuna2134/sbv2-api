# sbv2-api
このプロジェクトはStyle-Bert-ViTS2をONNX化したものをRustで実行するのを目的としています。つまり推論しか行いません。

学習したいのであれば、Style-Bert-ViT2で調べてやってください。

## ONNX化する方法
dabertaとstbv2本体をonnx化する必要があります。

あくまで推奨ですが、onnxsimを使うことをお勧めします。
onnxsim使うことでモデルのサイズを軽くすることができます。

## onnxモデルの配置方法
- `models/daberta.onnx` - DaBertaのonnxモデル
- `models/sbv2.onnx` - `Style-Bert-ViT2`の本体

## Todo
- [ ] WebAPIの実装
- [ ] Rustライブラリの実装
- [ ] 余裕があればPyO3使ってPythonで利用可能にする

## 謝辞
- [litagin02](https://github.com/litagin02/Style-Bert-VITS2) - このコードのベースとなる部分を参考にさせていただきました。
- [Googlefan](https://github.com/Googlefan256) - 彼にモデルをONNXヘ変換および効率化をする方法を教わりました。