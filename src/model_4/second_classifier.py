import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification,  AutoTokenizer


from utils import init_logger, binary_accuracy, load_tokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler
from model_4.second import BERTDANN
import json
import numpy as np

logger = logging.getLogger()

class SecondClassifier:
    def __init__(self, args, tokenizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AutoModelForSequenceClassification.from_pretrained(args.first_domain_classifier_output).to(self.device)
     
        self.tokenizer = tokenizer
        self.args = args

    def classifier(self):
        logger.info("***** Model Loaded *****")
        test_dataset = BERTDANN(self.args, self.tokenizer, 'train')
        test_loader = DataLoader(dataset=test_dataset, sampler=RandomSampler(test_dataset), batch_size=self.args.eval_batch_size)

        self.model.eval()

        preds, data, sentiment = [], [], []
        for batch in tqdm(test_loader, desc="Predicting"):
            sentiment.extend(batch[3].detach().cpu().numpy())
            data.extend(list(batch[4]))

            batch = tuple(t.to(self.device) for t in batch if type(t) != tuple)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": None}
            
                outputs = self.model(**inputs)
                logits = outputs[0]

                preds.append(logits)

        preds = torch.cat(preds, dim=0)
        temp = torch.sigmoid(preds)[:,1].detach().cpu().numpy()
    
        rounded_preds = torch.round(torch.sigmoid(preds)).argmax(axis=1)
        rounded_preds = rounded_preds.detach().cpu().numpy()
      
        for idx, lab in enumerate(temp):
            if lab > 0.2:
                rounded_preds[idx] = 1
   
        assert len(rounded_preds) == len(data)
        assert len(data) == len(sentiment)

        res = []
        for label, line, senti in zip(rounded_preds, data, sentiment):
            if label == 1:
                res.append({
                            'text':line,
                            'sentiment':int(senti)
                            })
        logger.info("  %s = %s", 'Number target similar', len(res))

        with open(self.args.scond_classifier_output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent='\t')

        logger.info("Prediction Done!")
