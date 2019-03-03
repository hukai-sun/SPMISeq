import re
import pickle
import numpy as np

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(br"\d")

def load_embedding_dict(embedding, embedding_path, normalize_digits=True):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimention, caseless
    """
    print("loading embedding: %s from %s" % (embedding, embedding_path))

    if embedding == 'glove':
        # loading GloVe
        embedd_dim = -1
        embedd_dict = dict()
        with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub(b"0", tokens[0].encode('utf-8')).decode('utf-8') if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'senna':
        # loading Senna
        embedd_dim = -1
        embedd_dict = dict()
        with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub(b"0", tokens[0].encode('utf-8')).decode('utf-8') if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'sskip':
        embedd_dim = -1
        embedd_dict = dict()
        with open(embedding_path, 'r') as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                try:
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if len(tokens) < embedd_dim:
                        continue

                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1

                    embedd = np.empty([1, embedd_dim], dtype=np.float32)
                    start = len(tokens) - embedd_dim
                    word = ' '.join(tokens[0:start])
                    embedd[:] = tokens[start:]
                    word = DIGIT_RE.sub(b"0", word.encode('utf-8')).decode('utf-8') if normalize_digits else word
                    embedd_dict[word] = embedd
                except UnicodeDecodeError:
                    continue
        return embedd_dict, embedd_dim
    elif embedding == 'polyglot':
        words, embeddings = pickle.load(open(embedding_path, 'rb'))
        _, embedd_dim = embeddings.shape
        embedd_dict = dict()
        for i, word in enumerate(words):
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = embeddings[i, :]
            word = DIGIT_RE.sub(b"0", word.encode('utf-8')).decode('utf-8') if normalize_digits else word
            embedd_dict[word] = embedd
        return embedd_dict, embedd_dim

    else:
        raise ValueError("embedding should choose from [senna, glove, sskip, polyglot]")

def construct_word_embedding_table(num_words, word_dim, embedd_dict, word_alphabet, unknown_id=0):
    scale = np.sqrt(3.0 / word_dim)
    table = np.empty([num_words, word_dim], dtype=np.float32)
    table[unknown_id, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
    oov = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            embedding = embedd_dict[word]
        elif word.lower() in embedd_dict:
            embedding = embedd_dict[word.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding
    print('oov: %d' % oov)
    
    return table

def construct_label_embedding_table(num_labels, label_dim, embedd_type):

    if embedd_type == 'one_hot':
        table = np.eye(num_labels, dtype=np.float32)
    elif embedd_type == 'random':
        scale = np.sqrt(3.0 / label_dim)
        table = np.random.uniform(-scale, scale, [num_labels, label_dim]).astype(np.float32)
    else:
        raise ValueError("embedd type should choose from [one_hot, random]")
    return table
