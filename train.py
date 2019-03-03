import os 
import argparse
import torch 
import time
import uuid
import datetime
import json 

from data.conll03_data import CoNLL03_Dataset
from models.RNNT_trainer import RNNT_Trainer
from models.NCRFT_trainer import NCRFT_Trainer 
from utils.logger import init_logging
from data.writer import CoNLL03Writer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir')
    parser.add_argument('--train_path', default='raw/rawtext/train.bioes.txt')
    parser.add_argument('--dev_path', default='raw/rawtext/valid.bioes.txt')
    parser.add_argument('--test_path', default='raw/rawtext/test.bioes.txt')
    parser.add_argument('--embedding_type', choices=['senna', 'glove', 'sskip', 'polyglot'], default='glove')
    parser.add_argument('--embedding_path', default='raw/embedd/glove.6B.100d.txt')

    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--load_checkpoint', action='store_true', default=False)
    parser.add_argument('--load_checkpoint_path')
    parser.add_argument('--model_name', choices=['RNNT', 'NCRFT'], default='RNNT')
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--tmp_path')
    parser.add_argument('--decode_mode', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--loss', choices=['local-joint', 'local-separate', 'global-joint-normal', \
        'global-joint-logsoftmax', 'global-separate-normal', 'global-separate-logsoftmax'])

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(args.gpu_id)
    
    logger, args.root_dir = init_logging('log-%s'%args.model_name, time.time())

    dataset = CoNLL03_Dataset(args.train_path, [args.dev_path, args.test_path], logger)
    
    dataset.read_embedding(args.embedding_type, args.embedding_path)

    print("Create Alphabets...")
    alphabet_path = os.path.join(args.root_dir, 'alphabet')
    dataset.create_alphabets(alphabet_path, max_vocabulary_size=50000)
    
    args.checkpoint_path = os.path.join(args.root_dir, 'checkpoint')
    os.makedirs(args.checkpoint_path)

    args.tmp_path = os.path.join(args.root_dir, 'tmp')
    os.makedirs(args.tmp_path)

    writer = CoNLL03Writer(dataset._word_alphabet, dataset._char_alphabet, dataset._pos_alphabet, \
            dataset._chunk_alphabet, dataset._ner_alphabet)

    with open(os.path.join(args.checkpoint_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    if args.model_name == 'RNNT':
        trainer = RNNT_Trainer(args, dataset, logger, writer)
    elif args.model_name == 'NCRFT':
        trainer = NCRFT_Trainer(args, dataset, logger, writer)

    trainer.train()

if __name__ == "__main__":
    main()
            



