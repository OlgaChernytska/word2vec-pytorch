def get_all_tokens(data_iter, tokenizer) -> list:
    all_tokens = []
    for data in data_iter:
        tokens = tokenizer(data)
        all_tokens.extend(tokens)
    return all_tokens


def get_vocab_dictionary(all_tokens: list, vocab_size: int, min_word_freq: int) -> dict:
    counter = Counter(all_tokens)
    large_tuples = [(k, v) for (k, v) in counter.items() if v > min_word_freq]
    sorted_tuples = sorted(large_tuples, key=lambda x: x[1], reverse=True)
    topn_tuples = sorted_tuples[:vocab_size]
    vocab_dict = dict(topn_tuples)
    return vocab_dict


def create_vocab(data_iter, tokenizer, vocab_size: int, min_word_freq: int):
    all_tokens = get_all_tokens(data_iter, tokenizer)
    vocab_dict = get_vocab_dictionary(all_tokens, vocab_size=5000, min_instances=50)
    vocabulary = vocab(vocab_dict)
    return vocabulary