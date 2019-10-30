from collections import Counter
import collections
import os
import torch


def get_word_len(counter, confidence):
    word2len = {}
    
    character_num = 0
    for word in counter.keys():
        word_len = len(word)
        word2len[word] = word_len
        character_num += (word_len * counter[word])
    word_num = sum(counter.values())
    avg_len_per_word = character_num / word_num

    sorted_word2len = collections.OrderedDict(
        sorted(word2len.items(), reverse=False, key=lambda t: t[1])
    )
    accumulated_len = 0
    for word, word_len in sorted_word2len.items():
        freq = counter[word]
        accumulated_len += freq
        percentage = accumulated_len / word_num
        if percentage > confidence:
            break
    most_long_word_len = word_len
    return avg_len_per_word, most_long_word_len


def test_get_word_len():
    counter = Counter()
    file_path = "/home/lr/yukun/pytorch_examples/word_lm/data/50lm/penn/valid.txt"
    lines = [
        counter.update(line.strip().split())
        for line in open(file_path, 'r').readlines()
    ]
    avg_len_per_word, most_long_word_len = get_word_len(counter, 0.998)
    #  most_long_word_len = min(most_long_word_len, 30)
    print(avg_len_per_word)
    print(most_long_word_len)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class CharNgrams(object):
    def __init__(self, n, add_begin_end, max_len, specials=None):
        self.specials = specials
        self.counter = Counter()
        self.chars2idx = {}
        # add special ngrams
        self.pad = "<pad>"
        self.idx2chars = [self.pad]
        self.chars2idx[self.pad] = len(self.idx2chars) - 1
        self.unk_char = "<unk_chars>"
        self.idx2chars.append(self.unk_char)
        self.chars2idx[self.unk_char] = len(self.idx2chars) - 1
        self.pad_index = self.chars2idx[self.pad]
        self.unk_char_index = self.chars2idx[self.unk_char]
        # adding specials
        if specials is not None:
            for special in specials:
                self.idx2chars.append(special)
                self.chars2idx[special] = len(self.idx2chars) - 1

        self.n = n
        self.max_len = max_len
        self.add_begin_end = add_begin_end

    def get_ngrams(self, word):
        if self.specials is not None and word in self.specials:
            return [word]
        if self.add_begin_end:
            word = f"${word}^"
        n = self.n
        chars_list = [word[i:i+n] for i in range(len(word)-n+1)]
        return chars_list

    def get_ngrams_index(self, word, padding=True):
        chars_list = self.get_ngrams(word)
        chars_list = chars_list[0:self.max_len]
        real_length = len(chars_list)
        if padding:
            chars_list.extend([self.pad] * (self.max_len - len(chars_list)))
        index_list = []
        for chars in chars_list:
            index = self.unk_char_index
            if chars in self.chars2idx:
                index = self.chars2idx[chars]
            index_list.append(index)
        return index_list, real_length

    def add_word(self, word):
        chars_list = self.get_ngrams(word)
        chars_list = chars_list[0:self.max_len]
        self.counter.update(chars_list)
        for chars in chars_list:
            if chars not in self.chars2idx:
                self.idx2chars.append(chars)
                self.chars2idx[chars] = len(self.idx2chars) - 1


class Corpus(object):
    def __init__(self, path, use_ngram=True, max_gram_n=3,
            add_begin_end=True, max_ngram_len=20, input_freq=None, input_extra_unk="<input_extra_unk>"):
        """
        input_extra_unk: when using fixed input vocab decided by input_freq. If this param is None
        and input_freq is 1. This tag will not be appended to input vocab and thus can tie input
        and output word embedding.
        """
        self.dictionary = Dictionary()
        self.input_dict = Dictionary()
        self.dict_for_ngram = Dictionary()
        # sometimes unk_tag appears in the corpus
        self.unk_tag = "<unk>"
        # real tags to represents words not appearing in training data for input data
        self.input_extra_unk = input_extra_unk
        self.eos_tag = "<eos>"
        train_path = os.path.join(path, 'train.txt')
        self.counter = Counter()
        lines = [
            self.counter.update(line.strip().split())
            for line in open(train_path, 'r').readlines()
        ]

        if input_freq is None:
            type_token_ratio = f"{len(self.counter.keys()) / sum(self.counter.values()):5.2f}"
            self.type_token = float(type_token_ratio) * 100
            self.type_token = int(self.type_token)
            if self.type_token <= 5:
                self.input_freq = 5
            elif self.type_token >= 10:
                self.input_freq = 10
            else:
                self.input_freq = self.type_token
            print(f"automatically chosen input_freq: {self.input_freq}")
        else:
            self.input_freq = input_freq

        self.use_ngram = use_ngram
        if self.use_ngram:
            avg_len_per_word, most_long_word_len = get_word_len(self.counter, 0.99999)
            most_long_word_len = min(most_long_word_len + 1, 40)
            gram_n = min(max_gram_n, int(avg_len_per_word))
            print(f"max length of word:{most_long_word_len}")
            print(f"n value in n-gram: {gram_n}")
            specials = [self.unk_tag, self.eos_tag]
            self.char_ngrams = CharNgrams(
                gram_n,
                add_begin_end,
                most_long_word_len,
                specials
            )
        if self.use_ngram:
            self.train, self.train_ngram = self.tokenize(
                train_path,
                add_to_vocab=True,
                return_ngram=True
            )
            self.valid, self.valid_ngram = self.tokenize(
                os.path.join(path, 'valid.txt'),
                return_ngram=True
            )
            self.test, self.test_ngram = self.tokenize(
                os.path.join(path, 'test.txt'),
                return_ngram=True
            )
        else:
            self.train = self.tokenize(train_path, add_to_vocab=True)
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

        # preprare for fixed-vocab input
        self.train_fixed = self.get_fixed_input_data(self.train)
        self.valid_fixed = self.get_fixed_input_data(self.valid)
        self.test_fixed = self.get_fixed_input_data(self.test)
        if self.input_extra_unk is not None and self.input_extra_unk in self.input_dict.word2idx:
            self.input_unseen_idx = self.input_dict.word2idx[self.input_extra_unk]
        else:
            self.input_unseen_idx = self.input_dict.word2idx[self.unk_tag]

        # add ngram input
        if self.use_ngram:
            self.ngram_train, self.ngram_train_len = self.get_ngram_data(self.train_ngram)
            self.ngram_test, self.ngram_test_len = self.get_ngram_data(self.test_ngram)
            self.ngram_valid, self.ngram_valid_len = self.get_ngram_data(self.valid_ngram)

    def get_fixed_input_data(self, data):
        fixed_data = torch.zeros(
            data.size(0),
            dtype=data.dtype
        )
        for word_int_index, word_int in enumerate(data, 0):
            word_str = self.dictionary.idx2word[word_int]
            if word_str not in self.input_dict.word2idx:
                word_str = self.unk_tag
                if self.input_extra_unk in self.input_dict.word2idx:
                    word_str = self.input_extra_unk
            fixed_data[word_int_index] = self.input_dict.word2idx[word_str]
        return fixed_data

    def get_ngram_data(self, data):
        ngram_data = torch.zeros(
            data.size(0),
            self.char_ngrams.max_len,
            dtype=data.dtype
        )
        ngram_length = torch.zeros(
            data.size(0),
            dtype=data.dtype
        )
        for word_int_index, word_int in enumerate(data, 0):
            word_str = self.dict_for_ngram.idx2word[word_int]
            ngram_list, real_length = self.char_ngrams.get_ngrams_index(word_str)
            ngram_data[word_int_index] = torch.tensor(ngram_list, dtype=data.dtype)
            ngram_length[word_int_index] = real_length
        return ngram_data, ngram_length


    def id_to_words(self, idx_list):
        word_list = []
        for idx in idx_list:
            word = self.dictionary.idx2word[idx]
            word_list.append(word)
        return word_list

    def tokenize(self, path, add_to_vocab=False, return_ngram=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + [self.eos_tag]
                tokens += len(words)
                for word in words:
                    if add_to_vocab:
                        self.dictionary.add_word(word)
                        if word in self.counter and self.counter[word] >= self.input_freq:
                            self.input_dict.add_word(word)
                        elif word == self.eos_tag:
                            self.input_dict.add_word(word)
                    if self.use_ngram:
                        self.char_ngrams.add_word(word)
                        self.dict_for_ngram.add_word(word)

        if add_to_vocab:
            # no unk_tag in this corpus
            if self.unk_tag not in self.dictionary.word2idx:
                self.dictionary.add_word(self.unk_tag)
                self.dict_for_ngram.add_word(self.unk_tag)
                self.input_dict.add_word(self.unk_tag)
            else:
                # unk_tag already exists in this corpus. Define a another tag for fixed input
                # vocab. Thus the orginal unk_tag will be treated as normal word.
                if self.input_extra_unk is not None:
                    self.input_dict.add_word(self.input_extra_unk)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            if self.use_ngram:
                ids_for_ngram = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + [self.eos_tag]
                for word in words:
                    if self.use_ngram:
                        ids_for_ngram[token] = self.dict_for_ngram.word2idx[word]
                    if word not in self.dictionary.word2idx:
                        word = self.unk_tag
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        if return_ngram:
            return ids, ids_for_ngram
        else:
            return ids


def test_charngrams():
    cn = CharNgrams(2, True, 8)
    cn.add_word("happniess")
    cn.add_word("am")
    print("dict")
    print(cn.chars2idx)
    print(cn.idx2chars)
    print(f"max_len: {cn.max_len}")
    word = "happniess"
    print(f"word {word} ngrams")
    print(cn.get_ngrams(word))
    word = "happysdfsdfsdfdsf"
    print(f"word {word} ngrams index")
    print(cn.get_ngrams_index(word))


if __name__ == "__main__":
    #  test_charngrams()
    test_get_word_len()
