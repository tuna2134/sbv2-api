from transformers.convert_slow_tokenizer import BertConverter
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--model", default="ku-nlp/deberta-v2-large-japanese-char-wwm")
args = parser.parse_args()
model_name = args.model

bert_models.load_tokenizer(Languages.JP, model_name)
tokenizer = bert_models.load_tokenizer(Languages.JP)
converter = BertConverter(tokenizer)
tokenizer = converter.converted()
tokenizer.save("../../models/tokenizer.json")


class ORTDeberta(nn.Module):
    def __init__(self, model_name):
        super(ORTDeberta, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, input_ids, token_type_ids, attention_mask):
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        res = self.model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        return res


model = ORTDeberta(model_name)
inputs = AutoTokenizer.from_pretrained(model_name)(
    "今日はいい天気ですね", return_tensors="pt"
)

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]),
    "../../models/deberta.onnx",
    input_names=["input_ids", "token_type_ids", "attention_mask"],
    output_names=["output"],
    verbose=True,
    dynamic_axes={"input_ids": {1: "batch_size"}, "attention_mask": {1: "batch_size"}},
)
os.system("onnxsim ../../models/deberta.onnx ../../models/deberta.onnx")