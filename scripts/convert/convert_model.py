import numpy as np
import json
from io import BytesIO
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.models.infer import get_net_g, get_text
from style_bert_vits2.models.hyper_parameters import HyperParameters
import torch
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
import os
from tarfile import open as taropen, TarInfo
from zstandard import ZstdCompressor
from style_bert_vits2.tts_model import TTSModel
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--style_file", required=True)
parser.add_argument("--config_file", required=True)
parser.add_argument("--model_file", required=True)
args = parser.parse_args()
style_file = args.style_file
config_file = args.config_file
model_file = args.model_file

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

array = np.load(style_file)
data = array.tolist()
hyper_parameters = HyperParameters.load_from_json(config_file)
out_name = hyper_parameters.model_name

with open(f"../../models/style_vectors_{out_name}.json", "w") as f:
    json.dump(
        {
            "data": data,
            "shape": array.shape,
        },
        f,
    )
text = "今日はいい天気ですね。"

bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
    text,
    Languages.JP,
    hyper_parameters,
    "cpu",
    assist_text=None,
    assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
    given_phone=None,
    given_tone=None,
)

tts_model = TTSModel(
    model_path=model_file,
    config_path=config_file,
    style_vec_path=style_file,
    device="cpu",
)
device = "cpu"
style_id = tts_model.style2id[DEFAULT_STYLE]


def get_style_vector(style_id, weight):
    style_vectors = np.load(style_file)
    mean = style_vectors[0]
    style_vec = style_vectors[style_id]
    style_vec = mean + (style_vec - mean) * weight
    return style_vec


style_vector = get_style_vector(style_id, DEFAULT_STYLE_WEIGHT)

x_tst = phones.to(device).unsqueeze(0)
tones = tones.to(device).unsqueeze(0)
lang_ids = lang_ids.to(device).unsqueeze(0)
bert = bert.to(device).unsqueeze(0)
ja_bert = ja_bert.to(device).unsqueeze(0)
en_bert = en_bert.to(device).unsqueeze(0)
x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
style_vec_tensor = torch.from_numpy(style_vector).to(device).unsqueeze(0)

model = get_net_g(
    model_file,
    hyper_parameters.version,
    device,
    hyper_parameters,
)


def forward(x, x_len, sid, tone, lang, bert, style, length_scale, sdp_ratio, noise_scale, noise_scale_w):
    return model.infer(
        x,
        x_len,
        sid,
        tone,
        lang,
        bert,
        style,
        sdp_ratio=sdp_ratio,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
    )


model.forward = forward

torch.onnx.export(
    model,
    (
        x_tst,
        x_tst_lengths,
        torch.LongTensor([0]).to(device),
        tones,
        lang_ids,
        bert,
        style_vec_tensor,
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.6777),
        torch.tensor(0.8),
    ),
    f"../../models/model_{out_name}.onnx",
    verbose=True,
    dynamic_axes={
        "x_tst": {0: "batch_size", 1: "x_tst_max_length"},
        "x_tst_lengths": {0: "batch_size"},
        "sid": {0: "batch_size"},
        "tones": {0: "batch_size", 1: "x_tst_max_length"},
        "language": {0: "batch_size", 1: "x_tst_max_length"},
        "bert": {0: "batch_size", 2: "x_tst_max_length"},
        "style_vec": {0: "batch_size"},
    },
    input_names=[
        "x_tst",
        "x_tst_lengths",
        "sid",
        "tones",
        "language",
        "bert",
        "style_vec",
        "length_scale",
        "sdp_ratio",
        "noise_scale",
        "noise_scale_w"
    ],
    output_names=["output"],
)
os.system(f"onnxsim ../../models/model_{out_name}.onnx ../../models/model_{out_name}.onnx")
onnxfile = open(f"../../models/model_{out_name}.onnx", "rb").read()
stylefile = open(f"../../models/style_vectors_{out_name}.json", "rb").read()
version = bytes("1", "utf8")
with taropen(f"../../models/tmp_{out_name}.sbv2tar", "w") as w:

    def add_tar(f, b):
        t = TarInfo(f)
        t.size = len(b)
        w.addfile(t, BytesIO(b))

    add_tar("version.txt", version)
    add_tar("model.onnx", onnxfile)
    add_tar("style_vectors.json", stylefile)
open(f"../../models/{out_name}.sbv2", "wb").write(
    ZstdCompressor(threads=-1, level=22).compress(
        open(f"../../models/tmp_{out_name}.sbv2tar", "rb").read()
    )
)
os.unlink(f"../../models/tmp_{out_name}.sbv2tar")
