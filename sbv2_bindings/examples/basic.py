from sbv2_bindings import TTSModel


def main():
    with open("../models/debert.onnx", "rb") as f:
        bert = f.read()
    with open("../models/tokenizer.json", "rb") as f:
        tokenizer = f.read()
    print("Loading models...")
    
    model = TTSModel(bert, tokenizer)

    with open("../models/amitaro.sbv2", "rb") as f:
        model.load_sbv2file(f.read())
    print("All setup is done!")
    
    style_vector = model.get_style_vector("amitaro", 0, 1.0)
    with open("output.wav", "wb") as f:
        f.write(model.synthesize("こんにちは", "amitaro", style_vector, 0.0, 0.5))


if __name__ == "__main__":
    main()