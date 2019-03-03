import os.path
import random
import numpy as np
from .reader import CoNLL03Reader
from .alphabet import Alphabet
from .embedding import load_embedding_dict

import torch
import re

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(br"\d")

# Special vocabulary symbols - we always put them at the start.
PAD = "_PAD"
PAD_POS = "_PAD_POS"
PAD_CHUNK = "_PAD_CHUNK"
PAD_NER = "_PAD_NER"
PAD_CHAR = "_PAD_CHAR"
_START_VOCAB = [PAD,]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 1

_buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 140]

class CoNLL03_Dataset(object):
    def __init__(self, train_path, eval_paths, logger):
        super(CoNLL03_Dataset, self).__init__()

        self._train_path = train_path
        self._eval_paths = eval_paths
        self._logger = logger
        
        self._word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
        self._char_alphabet = Alphabet('character', defualt_value=True)
        self._pos_alphabet = Alphabet('pos')
        self._chunk_alphabet = Alphabet('chunk')
        self._ner_alphabet = Alphabet('ner')
        
        self._char_alphabet.add(PAD_CHAR)
        self._pos_alphabet.add(PAD_POS)
        self._chunk_alphabet.add(PAD_CHUNK)
        self._ner_alphabet.add(PAD_NER)

        self._word_embedding_dict = None
    
    def read_embedding(self, embedding_type, embedding_path):

        embedd_word_dict, _ = load_embedding_dict(embedding_type, embedding_path)

        self._word_embedding_dict = embedd_word_dict

    def create_alphabets(self, alphabet_directory, max_vocabulary_size=50000,
                     min_occurence=1, normalize_digits=True):
        
        def expand_vocab():
            vocab_set = set(vocab_list)
            for data_path in self._eval_paths:
                # logger.info("Processing data: %s" % data_path)
                with open(data_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if len(line) == 0:
                            continue

                        tokens = line.split(' ')
                        word = DIGIT_RE.sub(b"0", tokens[0].encode('utf-8')).decode('utf-8') if normalize_digits else tokens[0]
                        pos = tokens[1]
                        chunk = tokens[2]
                        ner = tokens[3]

                        self._pos_alphabet.add(pos)
                        self._chunk_alphabet.add(chunk)
                        self._ner_alphabet.add(ner)

                        if word not in vocab_set and (word in self._word_embedding_dict or word.lower() in self._word_embedding_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)



        vocab = dict()
        with open(self._train_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split(' ')
                for char in tokens[0]:
                    self._char_alphabet.add(char)

                word = DIGIT_RE.sub(b"0", tokens[0].encode('utf-8')).decode('utf-8') if normalize_digits else tokens[0]
                pos = tokens[1]
                chunk = tokens[2]
                ner = tokens[3]

                self._pos_alphabet.add(pos)
                self._chunk_alphabet.add(chunk)
                self._ner_alphabet.add(ner)

                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if self._word_embedding_dict is not None:
            for word in vocab.keys():
                if word in self._word_embedding_dict or word.lower() in self._word_embedding_dict:
                    vocab[word] += min_occurence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        self._logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        self._logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        self._logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if self._eval_paths is not None and self._word_embedding_dict is not None:
            expand_vocab()

        for word in vocab_list:
            self._word_alphabet.add(word)
            if word in singletons:
                self._word_alphabet.add_singleton(self._word_alphabet.get_index(word))

        self._word_alphabet.save(alphabet_directory)
        self._char_alphabet.save(alphabet_directory)
        self._pos_alphabet.save(alphabet_directory)
        self._chunk_alphabet.save(alphabet_directory)
        self._ner_alphabet.save(alphabet_directory)

        self._word_alphabet.close()
        self._char_alphabet.close()
        self._pos_alphabet.close()
        self._chunk_alphabet.close()
        self._ner_alphabet.close()

        self._logger.info("Word Alphabet Size (Singleton): %d (%d)" % (self._word_alphabet.size(), self._word_alphabet.singleton_size()))
        self._logger.info("Character Alphabet Size: %d" % self._char_alphabet.size())
        self._logger.info("POS Alphabet Size: %d" % self._pos_alphabet.size())
        self._logger.info("Chunk Alphabet Size: %d" % self._chunk_alphabet.size())
        self._logger.info("NER Alphabet Size: %d" % self._ner_alphabet.size())
        
    def read_data(self, source_path, max_size=None, normalize_digits=True):
        data = [[] for _ in _buckets]
        max_char_length = [0 for _ in _buckets]
        print('Reading data from %s' % source_path)
        counter = 0
        reader = CoNLL03Reader(source_path, self._word_alphabet, self._char_alphabet, self._pos_alphabet, self._chunk_alphabet, self._ner_alphabet)
        inst = reader.getNext(normalize_digits)
        while inst is not None and (not max_size or counter < max_size):
            counter += 1
            if counter % 10000 == 0:
                print("reading data: %d" % counter)

            inst_size = inst.length()
            sent = inst.sentence
            for bucket_id, bucket_size in enumerate(_buckets):
                if inst_size < bucket_size:
                    data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.chunk_ids, inst.ner_ids])
                    max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                    if max_char_length[bucket_id] < max_len:
                        max_char_length[bucket_id] = max_len
                    break

            inst = reader.getNext(normalize_digits)
        reader.close()
        print("Total number of data: %d" % counter)
        return data, max_char_length

    def read_data_to_tensor(self, source_path, device, max_size=None, normalize_digits=True, has_start_node=True):

        data, max_char_length = self.read_data(source_path, max_size=max_size, normalize_digits=normalize_digits)
        bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

        data_tensor = []

        for bucket_id in range(len(_buckets)):
            bucket_size = bucket_sizes[bucket_id]
            if bucket_size == 0:
                data_tensor.append((1, 1))
                continue

            bucket_length = _buckets[bucket_id]
            char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
            wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
            pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            chid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

            masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
            single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

            lengths = np.empty(bucket_size, dtype=np.int64)

            for i, inst in enumerate(data[bucket_id]):
                wids, cid_seqs, pids, chids, nids = inst

                if not has_start_node :
                    pids_nostart = [pos -1 for pos in pids]
                    chids_nostart = [chunk -1 for chunk in chids]
                    nids_nostart = [ner -1 for ner in nids]

                inst_size = len(wids)
                lengths[i] = inst_size
                # word ids
                wid_inputs[i, :inst_size] = wids
                wid_inputs[i, inst_size:] = PAD_ID_WORD
                for c, cids in enumerate(cid_seqs):
                    cid_inputs[i, c, :len(cids)] = cids
                    cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
                cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
                # pos ids
                pid_inputs[i, :inst_size] = pids if has_start_node else pids_nostart
                pid_inputs[i, inst_size:] = PAD_ID_TAG
                # chunk ids
                chid_inputs[i, :inst_size] = chids if has_start_node else chids_nostart 
                chid_inputs[i, inst_size:] = PAD_ID_TAG
                # ner ids
                nid_inputs[i, :inst_size] = nids if has_start_node else nids_nostart
                nid_inputs[i, inst_size:] = PAD_ID_TAG
                # masks
                masks[i, :inst_size] = 1.0
                for j, wid in enumerate(wids):
                    if self._word_alphabet.is_singleton(wid):
                        single[i, j] = 1

            words = torch.from_numpy(wid_inputs).to(device)
            chars = torch.from_numpy(cid_inputs).to(device)
            pos = torch.from_numpy(pid_inputs).to(device)
            chunks = torch.from_numpy(chid_inputs).to(device)
            ners = torch.from_numpy(nid_inputs).to(device)
            masks = torch.from_numpy(masks).to(device)
            single = torch.from_numpy(single).to(device)
            lengths = torch.from_numpy(lengths).to(device)

            data_tensor.append((words, chars, pos, chunks, ners, masks, single, lengths))

        return data_tensor, bucket_sizes

    def get_batch_tensor(self, data, batch_size, unk_replace=0.):
        
        data_tensor, bucket_sizes = data
        total_size = float(sum(bucket_sizes))
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number = np.random.random_sample()
        bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
        bucket_length = _buckets[bucket_id]

        words, chars, pos, chunks, ners, masks, single, lengths = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        batch_size = min(bucket_size, batch_size)
        index = torch.randperm(bucket_size).long()[:batch_size]
        if words.is_cuda:
            index = index.cuda()

        words = words[index]
        if unk_replace:
            ones = single.new_zeros((batch_size, bucket_length)).fill_(1)
            noise = masks.new_zeros((batch_size, bucket_length)).bernoulli_(unk_replace).long()
            words = words * (ones - single[index] * noise)

        return words, chars[index], pos[index], chunks[index], ners[index], masks[index], lengths[index]

    def iterate_batch_tensor(self, data, batch_size, unk_replace=0., shuffle=False):
        data_tensor, bucket_sizes = data

        bucket_indices = np.arange(len(_buckets))
        if shuffle:
            np.random.shuffle((bucket_indices))

        for bucket_id in bucket_indices:
            bucket_size = bucket_sizes[bucket_id]
            bucket_length = _buckets[bucket_id]
            if bucket_size == 0:
                continue

            words, chars, pos, chunks, ners, masks, single, lengths = data_tensor[bucket_id]
            if unk_replace:
                ones = single.new_zeros((bucket_size, bucket_length)).fill_(1)
                noise = masks.new_zeros((bucket_size, bucket_length)).bernoulli_(unk_replace).long()
                words = words * (ones - single * noise)

            indices = None
            if shuffle:
                indices = torch.randperm(bucket_size).long()
                if words.is_cuda:
                    indices = indices.cuda()
            for start_idx in range(0, bucket_size, batch_size):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + batch_size)
                yield words[excerpt], chars[excerpt], pos[excerpt], chunks[excerpt], ners[excerpt], \
                    masks[excerpt], lengths[excerpt]
