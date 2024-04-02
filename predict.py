import os

import fire
import torch

from config import GlobalConfig
from model import Transformer
from process_data import get_tokens
from utils import Predictor
from utils import load_dict_from_pickle


data_config = GlobalConfig.data_config
model_config = GlobalConfig.model_config


def predict(
    model_path: str,
    num_tokens: int,
    compute_device: torch.device,
    generate: bool = False,
    data_tensors: str =data_config.data_tensors_path,
    temperature: float=1.0,
):
    print("Using device {} for inference".format(compute_device))
    checkpoint = torch.load(model_path)
    config = checkpoint["config"]
    data_config = config.data
    model_config = config.model
    model = Transformer(
        vocab_size=data_config.vocab_size,
        embedding_dim=model_config.embedding_dim,
        seq_length=data_config.seq_length,
        num_blocks=model_config.num_blocks,
        num_heads_in_block=model_config.num_heads_in_block,
        dropout=model_config.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(compute_device)
    model.eval()

    idx_to_word = load_dict_from_pickle(os.path.join(data_tensors, "idx_to_word.pkl"))
    word_to_idx = load_dict_from_pickle(os.path.join(data_tensors, "word_to_idx.pkl"))
    predictor = Predictor(
        model,
        idx_to_word,
        word_to_idx,
        compute_device,
        temperature,
        config.data.seq_length,
    )
    if not generate:
        input_str = input("Prompt: ")
        output = predictor.predict_tokens(get_tokens(input_str), num_tokens)
    else:
        output = predictor.generate_text(num_tokens, config.data.seq_length)
    output = Predictor.beautify_output(output)
    print("Generated text: \n{}".format(output))


if __name__ == "__main__":
    fire.Fire(predict)
