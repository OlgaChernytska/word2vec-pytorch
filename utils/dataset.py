import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset
from torchtext.data import to_map_style_dataset
from torchtext.vocab import vocab
from torchtext.datasets import WikiText2, WikiText103

from utils.constants import CBOW_N_WORDS, MIN_WORD_FREQUENCY


class CBOW_Dataset(Dataset):
    def __init__(self, name, set_type, data_dir, vocab=None):
        self.data_iter = self._get_data_iterator(name, set_type, data_dir)
        self.tokens = self._get_tokens_from_iterator(self.data_iter)
        if vocab != None:
            self.vocab = vocab
        else:
            self.vocab = self._build_vocabulary(self.tokens)
        self.vocab_size = len(self.vocab.get_stoi())

    def _get_data_iterator(self, dataset_name, set_type, data_dir):
        if dataset_name == "WikiText2":
            data_iter = WikiText2(root=data_dir, split=(set_type))
        elif dataset_name == "WikiText103":
            data_iter = WikiText103(root=data_dir, split=(set_type))
        else:
            raise ValueError("Choose dataset from: WikiText2, WikiText103")
        data_iter = to_map_style_dataset(data_iter)
        return data_iter

    def _get_tokens_from_iterator(self, data_iter):
        tokens = []
        for line in data_iter:
            line_lower = line.lower()
            tokens += line_lower.split()
        return tokens

    def _build_vocabulary(self, tokens):
        tokens_counter = Counter(tokens)
        vocabulary = vocab(tokens_counter, min_freq=MIN_WORD_FREQUENCY)
        vocabulary.set_default_index(vocabulary["<unk>"])
        return vocabulary

    def __len__(self):
        return len(self.tokens) - CBOW_N_WORDS + 1

    def __getitem__(self, idx):
        token_sequence = self.tokens[idx : (idx + CBOW_N_WORDS)]
        token_id_sequence = [self.vocab[x] for x in token_sequence]
        output = token_id_sequence.pop(CBOW_N_WORDS // 2)
        inputs = token_id_sequence

        inputs_vector = np.zeros((CBOW_N_WORDS - 1, self.vocab_size))
        for i, id_ in enumerate(inputs):
            inputs_vector[i, id_] = 1

        output = torch.tensor(output, dtype=torch.long)
        inputs_vector = torch.tensor(inputs_vector, dtype=torch.float32)
        return inputs_vector, output