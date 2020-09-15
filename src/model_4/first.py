# domain classifier

import json
import random
import copy
import logging
import argparse
import os
import yaml
from datetime import datetime
from tqdm import tqdm, trange
from utils import set_seeds, clean
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer, AutoConfig

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import WEIGHTS_NAME, CONFIG_NAME


class BERTDANN(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.tokenizer = tokenizer
        self.sample_counter = 0
        self.max_seq_length = 128

        with open(f'{args.data_dir}/sports.json', 'r', encoding='utf-8') as f:
            sports_data = json.load(f)
        with open(f"{args.data_dir}/tv.json", 'r', encoding='utf-8') as f:
            tv_data = json.load(f)
        with open(f"{args.data_dir}/movie_train.json", 'r', encoding='utf-8') as f:
            movie_data = json.load(f)
       
        self.sports_data = [(line['text'], 1) for line in sports_data if len(line['text']) > 5]
        self.tv_data = [(line['text'], 1) for line in tv_data if len(line['text']) > 5]
        self.movie_data = [(line['text'], 0) for line in movie_data if type(line['text'])==str and len(line['text']) > 5]
        
        target_train_size = min(int(0.9 * len(self.sports_data)), int(0.9 * len(self.tv_data)))
        movie_train_size = int(0.9 * len(self.movie_data))
        
        if mode == 'train':
            self.movie_data = self.movie_data[:movie_train_size]
            self.sports_data = self.sports_data[:target_train_size]
            if args.target_data == 'sports':
                self.dataset = self.sports_data + self.movie_data
                random.shuffle(self.dataset)
            else:
                self.dataset = self.tv_data[:target_train_size] + self.movie_data
                random.shuffle(self.dataset)
        else:
            self.movie_data = self.movie_data[movie_train_size:]
            self.sports_data = self.sports_data[target_train_size:]
            if args.target_data == 'sports':
                self.dataset = self.sports_data + self.movie_data
                random.shuffle(self.dataset)
            else:
                self.dataset = self.tv_data[target_train_size:] + self.movie_data
                random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        guid = self.sample_counter
        self.sample_counter += 1

        tokens_a = ""
        while tokens_a == "":
            tokens_a, label= self.random_sent(idx)
            tokens_a = self.tokenizer.tokenize(clean(tokens_a))
        
        example = InputExample(guid=guid, tokens_a=tokens_a, label=label)
        
        features = convert_example_to_features(example, self.tokenizer, self.max_seq_length)
        
        tensors = (torch.tensor(features.input_ids),
                    torch.tensor(features.input_mask),
                    torch.tensor(features.token_type_ids),
                    torch.tensor(features.label))

        return tensors
    
    def random_sent(self, idx):
        line, label = self.dataset[idx]
        return line, label


class InputExample:
    def __init__(self, 
                guid, 
                tokens_a, 
                tokens_b="", 
                label=None):
                
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.label = label

class InputFeatures:
    def __init__(self, 
                input_ids, 
                input_mask, 
                token_type_ids, 
                label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label = label


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b # None
    
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = []
    token_type_ids = []
    tokens.append("[CLS]")
    token_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        token_type_ids.append(0)
    tokens.append("[SEP]")
    token_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        token_type_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length

    if example.guid < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print(
                "segment_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        print("movie = 0 | sports = 1: %s " % (example.label))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             token_type_ids=token_type_ids,
                             label=example.label)
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
