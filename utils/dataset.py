import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset
from torchtext.data import to_map_style_dataset
from torchtext.vocab import vocab
from torchtext.datasets import WikiText2, WikiText103

from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    SKIPGRAM_CONTEXT_SAMPLING_POWER,
)


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
        return len(self.tokens) - CBOW_N_WORDS * 2

    def __getitem__(self, idx):
        token_sequence = self.tokens[idx : (idx + CBOW_N_WORDS * 2 + 1)]
        token_id_sequence = self.vocab(token_sequence)
        output = token_id_sequence.pop(CBOW_N_WORDS)
        inputs = token_id_sequence

        output = torch.tensor(output, dtype=torch.long)
        inputs = torch.tensor(inputs, dtype=torch.long)
        return inputs, output
    
    
class SkipGram_Dataset(Dataset):
    def __init__(self, name, set_type, data_dir, vocab=None):
        self.data_iter = self._get_data_iterator(name, set_type, data_dir)
        self.tokens = self._get_tokens_from_iterator(self.data_iter)
        if vocab != None:
            self.vocab = vocab
        else:
            self.vocab = self._build_vocabulary(self.tokens)
        self.vocab_size = len(self.vocab.get_stoi())
        self.token_probs = self._get_neighbouring_token_probs()

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

    def _get_neighbouring_token_probs(self):
        probs = np.concatenate(
            (
                np.arange(1, SKIPGRAM_N_WORDS + 1),
                np.array([0]),
                np.arange(SKIPGRAM_N_WORDS, 0, -1),
            )
        )
        probs = probs ** SKIPGRAM_CONTEXT_SAMPLING_POWER
        probs = probs / probs.sum()
        return probs

    def __len__(self):
        return len(self.tokens) - SKIPGRAM_N_WORDS * 2

    def __getitem__(self, idx):
        token_sequence = self.tokens[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
        token_id_sequence = self.vocab(token_sequence)
        input_ = token_id_sequence[SKIPGRAM_N_WORDS]
        output = np.random.choice(token_id_sequence, p=self.token_probs)

        output = torch.tensor(output, dtype=torch.long)
        input_ = torch.tensor(input_, dtype=torch.long)
        return input_, output