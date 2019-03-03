import torch
import torch.nn as nn
import torch.optim as optim

import time
import uuid
import datetime
import os
from os import makedirs
from os.path import join, exists
import sys

from .NCRFT import NCRFT
from data.embedding import construct_word_embedding_table, construct_label_embedding_table
from utils.logger import save_checkpoint
from hparams import hparams, hparams_debug_string

uid = uuid.uuid4().hex[:6]

def evaluate(output_file):
    tmp_dir = os.path.dirname(output_file)
    score_file = os.path.join(tmp_dir, "score_%s" % str(uid))
    os.system("eval/conll03eval.v2 < %s > %s" % (output_file, score_file))
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


class NCRFT_Trainer(object):
    def __init__(self, config, data_loader, logger, writer):
        super(NCRFT_Trainer, self).__init__()

        self.config = config
        self.data_loader = data_loader
        self.logger = logger
        self.writer = writer

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")

        self.logger.info(hparams_debug_string())

        print("build model...")
        self.build_model()

    def build_model(self):

        print("Construct word and label table...")
        num_words = self.data_loader._word_alphabet.size()
        num_chars = self.data_loader._char_alphabet.size()
        num_labels = self.data_loader._ner_alphabet.size()
        embedd_word_dict = self.data_loader._word_embedding_dict

        word_table = construct_word_embedding_table(num_words, hparams.word_dim, embedd_word_dict, self.data_loader._word_alphabet, unknown_id=0)
        label_table = construct_label_embedding_table(num_labels, hparams.label_dim, hparams.label_embedd_type)        
        word_embedd_tensor = torch.from_numpy(word_table)
        label_embedd_tensor = torch.from_numpy(label_table)

        print("Constructing network...")
        self.net = NCRFT(hparams, num_words, num_chars, num_labels, embedd_word_table=word_embedd_tensor, \
                embedd_char_table=None, embedd_label_table=label_embedd_tensor, device=self.device, norm_type=self.config.loss).to(self.device)

        self.lr = hparams.learning_rate
    
        if hparams.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0., nesterov=True)
        elif hparams.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError()

        if self.config.load_checkpoint and self.config.load_checkpoint_path:
            pretrained_model_checkpoint = torch.load(self.config.load_checkpoint_path)
            self.net.load_state_dict(pretrained_model_checkpoint['state_dict'])
            self.logger.info("succeed to load pretrained model from %s"%self.config.load_checkpoint_path)

    def train(self):

        self.data_train = self.data_loader.read_data_to_tensor(self.data_loader._train_path, self.device, max_size=None, normalize_digits=True, has_start_node=True)
        self.data_dev = self.data_loader.read_data_to_tensor(self.data_loader._eval_paths[0], self.device, max_size=None, normalize_digits=True, has_start_node=True)
        self.data_test = self.data_loader.read_data_to_tensor(self.data_loader._eval_paths[1], self.device, max_size=None, normalize_digits=True, has_start_node=True)

        num_data = sum(self.data_train[1])

        num_batches = num_data // hparams.batch_size + 1
                
        best_epoch = 0
        best_dev_f1 = 0.0
        early_epoch = 0
                
        for epoch in range(1, hparams.num_epochs + 1):
            train_err = 0.
            train_data_er = 0.
            train_sample_er = 0. 
            train_total = 0.

            start_time = time.time()
            num_back = 0


            for batch in range(1, num_batches + 1):
                word, char, _, _, labels, masks, lengths = self.data_loader.get_batch_tensor(self.data_train, hparams.batch_size, unk_replace=hparams.unk_replace)

                samples = self.sampling(word, char, masks, lengths)
                self.net.train()

                sos_mat = labels.new_zeros((labels.size(0), 1)).fill_(hparams.sos).long()
                input_label = torch.cat([sos_mat, labels], dim=1)
                input_label = input_label[:, :-1]

                sample_sos_mat = sos_mat.repeat(1, hparams.num_samples).view(-1, hparams.num_samples, 1)
                input_sample = torch.cat([sample_sos_mat, samples], dim=2)
                input_sample = input_sample[:, :, :-1]

                self.optimizer.zero_grad()
                
                loss, data_loss, sample_loss = self.net.loss(word, char, input_label, input_sample, labels, samples, masks, lengths)
                loss.backward()
                
                if hparams.clip_grad:
                    nn.utils.clip_grad_norm_(self.net.parameters(), hparams.clip_grad_norm)
                self.optimizer.step()

                num_inst = masks.size(0)
                train_err += loss.item() * num_inst
                train_data_er += data_loss.item() * num_inst
                train_sample_er += sample_loss.item() * num_inst
                train_total += num_inst

                time_ave = (time.time() - start_time) / batch
                time_left = (num_batches - batch) * time_ave

                # update log
                if batch % 100 == 0:
                    sys.stdout.write("\b" * num_back)
                    sys.stdout.write(" " * num_back)
                    sys.stdout.write("\b" * num_back)
                    log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (batch, num_batches, train_err / train_total, time_left)
                    sys.stdout.write(log_info)
                    sys.stdout.flush()
                    num_back = len(log_info)

            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            self.logger.info('epoch: %d, train: %d, loss: %.4f, data loss: %.4f, sample loss: %.4f, time: %.2fs, lr:%.4f' % (epoch,num_batches, train_err / train_total, train_data_er / train_total, train_sample_er / train_total, time.time() - start_time, self.lr))

            # evaluate performance on dev data
            tmp_filename = os.path.join(self.config.tmp_path, 'dev%d' % (epoch))
            f1 = self.eval('dev', epoch, tmp_filename)

            if best_dev_f1 <= f1:
                
                best_dev_f1 = f1
                best_epoch = epoch
                early_epoch = 0

                save_checkpoint({
                'epoch': best_epoch,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict()},
                join(self.config.checkpoint_path ,"%s_%d"%(self.config.model_name, best_epoch)))
            
            else:
                if epoch > hparams.early_start:
                    early_epoch += 1
                
                if early_epoch > hparams.early_patience:
                    self.logger.info("early stopped, best epoch: %d"%best_epoch)
                    os._exit(0)
            
            # evaluate performance on test data
            tmp_filename = os.path.join(self.config.tmp_path, 'test%d' % (epoch))
            _ = self.eval('test', epoch, tmp_filename)

            if epoch % hparams.schedule == 0:
                self.lr = hparams.learning_rate / (1.0 + epoch * hparams.decay_rate)
                
                if hparams.optimizer == 'SGD':
                    self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0, nesterov=True)
                
                elif hparams.optimizer == 'Adam':
                    self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
                else:
                    NotImplementedError()


    def eval(self, name, epoch, tmp_filename):

        # evaluate performance on data
        self.net.eval()
        self.writer.start(tmp_filename)
        
        if name == 'dev':
            eval_data = self.data_dev
        else:
            eval_data = self.data_test

        for batch in self.data_loader.iterate_batch_tensor(eval_data, hparams.batch_size):
            word, char, pos, chunk, labels, masks, lengths = batch
            
            if self.config.decode_mode == 'greedy':
                preds = self.net.greedy_search(word, char, masks, lengths)
                self.writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(), preds.data.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
            
            elif self.config.decode_mode == 'beam':
                preds = self.net.beam_search(hparams.beam_width, word, char, masks, lengths)
                preds = preds[:, 0, :]
                self.writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(), preds.data.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
            
            else:
                raise NotImplementedError()


        self.writer.close()
        acc, precision, recall, f1 = evaluate(tmp_filename)
        self.logger.info('epoch: %d, %s acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (epoch, name, acc, precision, recall, f1))

        return f1


    def sampling(self, input_word, input_char, mask, length, trans_hx=None, pred_hx=None):

        self.net.eval()
        samples = self.net.beam_search(hparams.num_samples, input_word, input_char, mask, length, trans_hx=trans_hx, pred_hx=pred_hx)

        return samples

        
        

        