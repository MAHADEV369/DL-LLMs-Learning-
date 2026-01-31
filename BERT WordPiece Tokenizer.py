def wordpiece_tokenizer(text, vocab, unk_token='[UNK]'):
    vocab_set = set(vocab)
    tokens = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = text_len
        cur_token = None

        while start < end:
            substr = text[start:end]
            if start > 0:
                substr = "##" + substr

            if substr in vocab_set:
                cur_token = substr
                break

            end -= 1

        if cur_token is None:
            return [unk_token]

        tokens.append(cur_token)
        start = end

    return tokens
