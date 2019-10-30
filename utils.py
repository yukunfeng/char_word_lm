#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Misc utils
"""

import logging
import torch
import numpy as np


def get_logger(log_file=None):
    """
    Logger from opennmt
    """

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger


def word_ids_to_sentence(id_tensor, vocab, word_len=20):
    """Converts a sequence of word ids to a sentence
    id_tensor: torch-based tensor
    vocab: torchtext vocab
    """
    all_strings = ""
    for row in id_tensor:
        row_string = ""
        for col in row:
            word = vocab.itos[col][0:word_len]
            word = word.ljust(word_len)
            row_string += word + " "
        all_strings += row_string + "\n"
    return all_strings


def probability_lookup(id_tensor, field, most_n_word=30):
    """
    probability_lookup, id_tensor are logits before softmax
    """
    softmax = torch.nn.Softmax(dim=0)
    probabilities = softmax(id_tensor)[0: most_n_word]
    numbers, indexs = id_tensor.sort(descending=True)
    probabilities = softmax(numbers)[0: most_n_word]
    numbers = numbers[0: most_n_word]
    indexs = indexs[0: most_n_word]
    word_list = []
    for index in indexs:
        word = field.vocab.itos[index]
        word_list.append(word)
    return numbers, probabilities, word_list, indexs


def save_word_embedding(vocab, emb, file_name):
    """Saving word emb"""
    with open(file_name, 'x') as fh:
        fh.write(f"{emb.size(0)} {emb.size(1)}\n")
        for word, vec in zip(vocab, emb):
            str_vec = [f"{x.item():5.4f}" for x in vec]
            line = word + " " + " ".join(str_vec) + "\n"
            fh.write(line)

def load_word_embedding(file_path):
    embeddings = []

    with open(file_path, 'r') as fh:
        for count, line in enumerate(fh, 0):
            line = line.strip()
            # Skip empty lines
            if line == "":
                continue
            items = line.split()

            # Skip first line if it is miklov-style vectors
            if count == 0 and len(items) == 2:
                continue
                
            embedding = [float(val) for val in items[1:]]
            embeddings.append(embedding)

    return torch.tensor(embeddings)

def save_word_embedding_test():
    vocab = ["a", "b", "c"]
    emb = torch.rand(len(vocab), 5)
    save_word_embedding(vocab, emb, "vec.txt")


def load_word_embedding_test():
    emb = load_word_embedding("tmp.vec")
    print(emb)


if __name__ == "__main__":
    # Unit test
    #  save_word_embedding_test()
    load_word_embedding_test()
