import itertools
import os
import pickle
import re

import contractions

from config import GlobalConfig
from utils import save_dict_as_pickle

config = GlobalConfig.config
data_config = GlobalConfig.data_config

token_linebreak = " [SEP] "
token_number = " [NUM] "

number_regex = re.compile(r"(?:- ?)?\d+\.\d+|(?:- ?)?\d+")
sent_regex = re.compile(r"(?:\.|\?|!)(?: \n?)?")
punc_regex = re.compile(
    r""";|:|,|"|\{|\}|\[|\]|\'|\(|\)|“|”|’|‘|\/|-|…|@|™|—|_|\\|\*"""
)
non_ascii_regex = re.compile(r"[^\x00-\x7F]+")
hashtag_regex = re.compile(r"(?<!\S)#(\S+)")


def filter_text(text: str):
    text = contractions.fix(text, slang=False)
    text = number_regex.sub(token_number, text)
    text = punc_regex.sub(" ", text)
    text = non_ascii_regex.sub(" ", text)
    text = sent_regex.sub(token_linebreak, text)
    text = hashtag_regex.sub(" ", text)
    return text


def get_tokens(poem_text: str):
    poem_text = poem_text.lower()
    tokens = filter_text(poem_text).split()
    tokens = [token for token in tokens if len(token.strip()) != 0]
    return tokens


if __name__ == "__main__":
    data_dir = data_config.data_path
    output_dir = data_config.data_tensors_path
    txt_files = os.listdir(data_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    vocab: list[str] = []
    tokenized_sentences: list[list[str]] = []
    with open(
        os.path.join(data_dir, "dataset_articles.txt"), "r", encoding="utf-8"
    ) as text_file:
        for line in text_file:
            tokens = get_tokens(line)
            vocab += tokens
            tokenized_sentences.append(tokens)
    print(f"{len(tokenized_sentences)} sentences read.")
    vocab = list(set(vocab))
    vocab = list(sorted(vocab))

    index_to_word: dict[int, str] = dict(zip(range(1, len(vocab) + 1), vocab))
    word_to_index: dict[str, int] = dict(zip(vocab, range(1, len(vocab) + 1)))
    save_dict_as_pickle(index_to_word, os.path.join(output_dir, "idx_to_word.pkl"))
    save_dict_as_pickle(word_to_index, os.path.join(output_dir, "word_to_idx.pkl"))
    config.data.vocab_size = len(index_to_word) + 1
    print(f"{config.data.vocab_size} words in vocabulary")
    GlobalConfig.save_global_config(config)

    tokenized_ds = itertools.chain(*tokenized_sentences)
    idx_tokenized_ds = [word_to_index[word] for word in tokenized_ds]

    with open(
        os.path.join(data_config.data_tensors_path, "sequences.pkl"), "wb"
    ) as file:
        pickle.dump(idx_tokenized_ds, file)
