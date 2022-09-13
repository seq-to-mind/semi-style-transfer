# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import random
import re

import math
import numpy as np
import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F

from transformers import RobertaForSequenceClassification, RobertaTokenizer
from utils import get_batches
from global_config import pretrained_style_model

device = 'cuda' if cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_sc(model, valid_loader, loss_fn, epoch, tokenizer):
    """ Evaluation function for style classifier """
    model.eval()
    total_acc = 0.
    total_num = 0.
    total_loss = 0.
    with torch.no_grad():
        for batch in valid_loader:
            x_batch = [i[0] for i in batch]
            x_batch = tokenizer(x_batch, add_special_tokens=True, padding=True, return_tensors="pt").data
            y_batch = torch.tensor([i[1] for i in batch]).to(device)
            logits = model(x_batch["input_ids"].to(device), attention_mask=x_batch["attention_mask"].to(device)).logits

            total_loss += loss_fn(logits, y_batch)
            _, y_hat = torch.max(logits, dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)
    model.train()
    print('[Info] Epoch {:02d}-valid: {}'.format(
        epoch, 'acc {:.4f}% | loss {:.4f}').format(
        total_acc / total_num * 100, total_loss / total_num))

    return total_acc / total_num, total_loss / total_num


def process_text(input_str):
    # process text for GYAFC data.
    input_str = re.sub("\s+", " ", input_str.replace("\t", " ")).strip()[:200]
    return input_str


def main():
    parser = argparse.ArgumentParser('Style Classifier TextCNN')
    parser.add_argument('-lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('-seed', default=100, type=int, help='pseudo random number seed')
    parser.add_argument('-min_count', default=0, type=int, help='minmum number of corpus')
    parser.add_argument('-max_len', default=30, type=int, help='maximum tokens in a batch')
    parser.add_argument('-log_step', default=100, type=int, help='print log every x steps')
    parser.add_argument('-eval_step', default=3000, type=int, help='early stopping training')
    parser.add_argument('-batch_size', default=64, type=int, help='maximum sents in a batch')
    parser.add_argument('-epoch', default=10, type=int, help='force stop at specified epoch')

    opt = parser.parse_args()

    setup_seed(opt.seed)

    tokenizer = RobertaTokenizer.from_pretrained(pretrained_style_model)

    corpus_name = "Yelp"

    if corpus_name in ["Yelp", "amazon"]:
        with open(corpus_name + '_data/sentiment.train.0', 'r') as f:
            tmp = f.readlines()
            train_0 = [[i.strip().lower(), 0] for i in tmp]
        with open(corpus_name + '_data/sentiment.train.1', 'r') as f:
            tmp = f.readlines()
            train_1 = [[i.strip().lower(), 1] for i in tmp]
        with open(corpus_name + '_data/sentiment.test.0', 'r') as f:
            tmp = f.readlines()
            valid_0 = [[i.strip().lower(), 0] for i in tmp]
        with open(corpus_name + '_data/sentiment.test.1', 'r') as f:
            tmp = f.readlines()
            valid_1 = [[i.strip().lower(), 1] for i in tmp]
    elif corpus_name in ["GYAFC", ]:
        train_0, train_1 = [], []
        train_data_files = {"GYAFC_data/Entertainment_Music/train/informal": "GYAFC_data/Entertainment_Music/train/formal",
                            "GYAFC_data/Family_Relationships/train/informal": "GYAFC_data/Family_Relationships/train/formal", }

        for one_file_name in train_data_files.keys():
            tmp_list_0 = open(one_file_name, encoding="utf-8").readlines()
            train_0.extend([[process_text(i), 0] for i in tmp_list_0])
            tmp_list_1 = open(train_data_files[one_file_name], encoding="utf-8").readlines()
            train_1.extend([[process_text(i), 1] for i in tmp_list_1])

        valid_0, valid_1 = [], []
        valid_data_files = {"GYAFC_data/Entertainment_Music/tune/informal": "GYAFC_data/Entertainment_Music/tune/formal",
                            "GYAFC_data/Family_Relationships/tune/informal": "GYAFC_data/Family_Relationships/tune/formal", }

        for one_file_name in valid_data_files.keys():
            tmp_list_0 = open(one_file_name, encoding="utf-8").readlines()
            valid_0.extend([[process_text(i), 0] for i in tmp_list_0])
            tmp_list_1 = open(valid_data_files[one_file_name], encoding="utf-8").readlines()
            valid_1.extend([[process_text(i), 1] for i in tmp_list_1])

    print('[Info] {} instances from train_0 set'.format(len(train_0)))
    print('[Info] {} instances from train_1 set'.format(len(train_1)))

    train_set = train_0 + train_1
    random.seed(100)
    random.shuffle(train_set)
    valid_set = valid_0 + valid_1
    random.seed(100)
    random.shuffle(valid_set)

    train_batches = get_batches(train_set, opt.batch_size)
    valid_batches = get_batches(valid_set, opt.batch_size)

    model = RobertaForSequenceClassification.from_pretrained(pretrained_style_model)
    model.to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002, betas=(0.9, 0.98), eps=1e-09)

    loss_fn = nn.CrossEntropyLoss()

    print('[Info] Built a model with {} parameters'.format(
        sum(p.numel() for p in model.parameters())))
    print('[Info]', opt)

    tab = 0
    avg_acc = 0
    total_acc = 0.
    total_num = 0.
    total_loss = 0.
    start = time.time()
    train_steps = 0

    for e in range(opt.epoch):

        model.train()
        for idx, batch in enumerate(train_batches):
            x_batch = [i[0] for i in batch]
            x_batch = tokenizer(x_batch, add_special_tokens=True, padding=True, return_tensors="pt").data
            y_batch = torch.tensor([i[1] for i in batch]).to(device)

            train_steps += 1

            optimizer.zero_grad()
            logits = model(x_batch["input_ids"].to(device), attention_mask=x_batch["attention_mask"].to(device)).logits
            loss = loss_fn(logits, y_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()

            y_hat = logits.argmax(dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)

            if train_steps % opt.log_step == 0:
                lr = optimizer.param_groups[0]['lr']
                print('[Info] Epoch {:02d}-{:05d}: | average acc {:.4f}% | '
                      'average loss {:.4f} | lr {:.6f} | second {:.2f}'.format(
                    e, train_steps, total_acc / total_num * 100,
                                    total_loss / (total_num), lr, time.time() - start))
                start = time.time()

            if train_steps % opt.eval_step == 0:
                valid_acc, valid_loss = evaluate_sc(model, valid_batches, loss_fn, e, tokenizer)
                if avg_acc < valid_acc or True:
                    avg_acc = valid_acc
                    save_path = 'saved_models/TextBERT_' + corpus_name + '/TextBERT_best.chkpt' + str(train_steps) + "_" + str(valid_acc)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 10:
                        exit()


if __name__ == '__main__':
    main()
