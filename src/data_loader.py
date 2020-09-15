from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import json
import random
from utils import clean

class BuildDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self. tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        sentiment = self.data[idx]['sentiment']

        if not isinstance(text, str):
            text = ""
        text = clean(text)
        
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            truncation = True
        )

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask'] if 'attention_mask' in encoded else None
        token_type_ids = encoded['token_type_ids'] if 'token_type_ids' in encoded else None
        
        input = {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'label':sentiment
        }

        for k, v in input.items():
            input[k] = torch.tensor(v)
        
        return input['input_ids'], input['attention_mask'], input['token_type_ids'], input['label']

    def __len__(self):
        return len(self.data)

# TODO yaml 사용하여 파싱 짜기 / max_length, batch size, epoch 등
def build_loader(args, tokenizer, mode):
    
    if mode == 'train':
        with open(f'{args.data_dir}/{args.train_data_dir}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.shuffle(data)
        train_size = int(0.9 * len(data))
        valid_size = int(0.99 * len(data))
        
        train_data = data[:train_size]
        valid_data = data[valid_size:]
        
        train_dataset = BuildDataset(train_data, tokenizer)
        valid_dataset = BuildDataset(valid_data, tokenizer)
    
        train_iterator = DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
        valid_iterator = DataLoader(dataset=valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=args.eval_batch_size)

        return train_dataset, valid_dataset, train_iterator, valid_iterator
    else:
        with open(f'{args.data_dir}/{args.test_data_dir}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.shuffle(data)
        
        test_dataset = BuildDataset(data, tokenizer)
        test_iterator = DataLoader(dataset=test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)

        return test_iterator
