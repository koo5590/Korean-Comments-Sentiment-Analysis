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
        self.max_seq_length = args.max_seq_length
        self.mode = mode

        with open(f'{args.data_dir}/{args.model4_train_dir}.json', 'r', encoding='utf-8') as f:
            sports_data = json.load(f)
        # with open(f"{args.scond_classifier_output}/source_target.json", 'r', encoding='utf-8') as f:
        #     movie_data = json.load(f)
        with open(f"{args.scond_classifier_output}/target_with_0.5.json", 'r', encoding='utf-8') as f:
            movie_data = json.load(f)
        
        with open(f"./data/movie_test.json", 'r', encoding='utf-8') as f:
            target = json.load(f)
        
        # most dissimilar
        # movie_data = sorted(movie_data, key=lambda x: x['target_prob'])
        # random
        random.shuffle(movie_data)
        # print(movie_data[:10])
        # raise Exception
        self.sports_data = [line['text'] for line in sports_data if len(line['text']) > 5]
        self.movie_data = [(line['text'], line['sentiment']) for line in movie_data if type(line['text'])==str and len(line['text']) > 5]
        # self.target =  [(line['text'], line['sentiment']) for line in target [:15000] if type(line['text'])==str and len(line['text']) > 5]
        # self.target =  [(line['text'], line['sentiment']) for line in target [:116] if type(line['text'])==str and len(line['text']) > 5]
        val = int(0.1 * len(self.movie_data))
        self.target =  [(line['text'], line['sentiment']) for line in target[:val] if type(line['text'])==str and len(line['text']) > 5]

        movie_train_size = int(0.9 * len(self.movie_data))
        print(val)
        if self.mode == 'train':
            self.dataset = self.movie_data
            # self.target = self.sports_data + self.sports_data[:10000]
            self.target = self.sports_data[:len(self.dataset)]
            
            # self.dataset = self.movie_data[:100]
            # self.target = self.sports_data[:100]
            random.shuffle(self.target)
        else:
            self.dataset = self.target
            self.target = self.movie_data
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        guid = self.sample_counter
        self.sample_counter += 1


        tokens_a = ""
        while tokens_a == "":
            tokens_a, label = self.random_sent(idx, 'source')
            tokens_a = self.tokenizer.tokenize(clean(tokens_a))
        
        if self.mode == 'train':
            tokens_b = ""
            while tokens_b == "":
                tokens_b = self.random_sent(idx, 'target')
                tokens_b = self.tokenizer.tokenize(clean(tokens_b[0]))
        
            example = InputExample(guid=guid, tokens_a=tokens_a, tokens_b=tokens_b, label=label)
            
            features_a = convert_example_to_features(example, self.tokenizer, self.max_seq_length, 'a', self.mode)
            features_b = convert_example_to_features(example, self.tokenizer, self.max_seq_length, 'b', self.mode)
            
            tensors = (
                        (torch.tensor(features_a.input_ids),
                        torch.tensor(features_a.input_mask),
                        torch.tensor(features_a.label)),
                        
                        (torch.tensor(features_b.input_ids),
                        torch.tensor(features_b.input_mask),
                        torch.tensor(features_a.label))
                    )
        else:
            example = InputExample(guid=guid, tokens_a=tokens_a, tokens_b="", label=label)
            features_a = convert_example_to_features(example, self.tokenizer, self.max_seq_length, 'a', self.mode)
            tensors = (
                torch.tensor(features_a.input_ids),
                torch.tensor(features_a.input_mask),
                torch.tensor(features_a.label)
            )
        return tensors
    
    def random_sent(self, idx, domain):
        if domain == 'source':
            line, label = self.dataset[idx]
            return line, label
        else:
            line = self.target[idx]
            return line

class InputExample:
    def __init__(self, 
                guid, 
                tokens_a, 
                tokens_b, 
                label=None):
                
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.label = label

class InputFeatures:
    def __init__(self, 
                input_ids, 
                input_mask,
                label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label


def convert_example_to_features(example, tokenizer, max_seq_length, mode, train='train'):
    if mode == 'a':
        tokens_a = example.tokens_a
    else:
        tokens_a = example.tokens_b
    
    _truncate_seq_pair(tokens_a, "", max_seq_length - 3)

    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    if example.guid < 5 and train=='train' and mode=='a':
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))


    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
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

