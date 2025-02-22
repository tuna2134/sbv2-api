# 変換方法

## 初心者向け準備

わかる人は飛ばしてください。

1. pythonを入れます。3.11.8で動作確認をしていますが、最近のバージョンなら大体動くはずです。

4. `cd convert`

3. `python -m venv venv`

4. `source venv/bin/activate`

5. `pip install -r requirements.txt`

## モデル変換

1. 変換したいモデルの`.safetensors`で終わるファイルの位置を特定してください。

2. 同様に`config.json`、`style_vectors.npy`というファイルを探してください。

3. 以下のコマンドを実行します。
```sh
python convert_model.py --style_file "ここにstyle_vectors.npyの場所" --config_file "同様にconfig.json場所" --model_file "同様に.safetensorsで終わるファイルの場所"
```

4. `models/名前.sbv2`というファイルが出力されます。GUI版のモデルファイルに入れてあげたら使えます。

## Deberta変換

意味が分からないならおそらく変換しなくてもいいってことです。

venvを用意し、requirementsを入れて、`python convert_model.py`を実行するだけです。

`models/deberta.onnx`と`models/tokenizer.json`が出力されたら成功です。