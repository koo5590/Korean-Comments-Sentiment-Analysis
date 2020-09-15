import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel


from utils import init_logger, compute_metrics, load_tokenizer
from data_loader import build_loader
import torch.nn as nn
import json


logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, args, tokenizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.model = BertModel.from_pretrained(args.test_model_dir).to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)
        self.classifier = nn.Linear(768, 2).to(self.device)
        
        self.tokenizer = tokenizer
        self.args = args

    def predict(self):
        logger.info("***** Model Loaded *****")
        test_loader = build_loader(self.args, self.tokenizer, 'test')
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(test_loader, desc="Predicting"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids":batch[2]
                        }
            
                outputs = self.model(**inputs)
                pooled_output = outputs[1]
                logits = self.classifier(pooled_output)
           
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, batch[3].detach().cpu().numpy(), axis=0)
    
        results = {}
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        # with open(f'{self.args.test_data_dir}_test.json', 'r', encoding='utf-8') as f, \
        #     open(f'wrong_sports.txt', "w", encoding="utf-8") as fw:
        #     data = json.load(f)
        #     for line, pred in zip(data, preds):
        #         if line['sentiment'] != pred:
        #             fw.write(f"{line['text']}\t{line['sentiment']}\t{pred}\n")

        # # Write to output file
        # with open(self.args.output_file, "w", encoding="utf-8") as f:
        #     for pred in preds:
        #         f.write("{}\n".format(pred))

        logger.info("Prediction Done!")
