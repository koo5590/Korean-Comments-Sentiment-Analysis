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
from new_model_4.second import BERTDANN
import json
import numpy as np

logger = logging.getLogger()

class SecondClassifier_n:
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
        preds = None
        out_label_ids = None
        preds_t, data, sentiment = [], [], []
        for batch in tqdm(test_loader, desc="Predicting"):
            sentiment.extend(batch[3].detach().cpu().numpy())
            data.extend(list(batch[4]))

            batch = tuple(t.to(self.device) for t in batch if type(t) != tuple)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": None}
            
                outputs = self.model(**inputs)
                logits = outputs[0]
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            preds_t.append(logits)
    
        preds_t = torch.cat(preds_t, dim=0)
        tar_prob = torch.sigmoid(preds_t)[:,1].detach().cpu().numpy()
      
        rounded_preds = np.argmax(preds, axis=1)
      

        # for idx, lab in enumerate(tar_prob):
        #     if lab > 0.45:
        #         rounded_preds[idx] = 1
   
        assert len(rounded_preds) == len(data)
        assert len(data) == len(sentiment)

        res = []
        all_data = []
        mani = []
        for label, line, senti, prob in zip(rounded_preds, data, sentiment, tar_prob):
            if label == 1:
                res.append({
                            'text':line,
                            'sentiment':int(senti),
                            'target_prob':float(prob)
                            })
            if prob > 0.5 or label == 1:
                mani.append({
                            'text':line,
                            'sentiment':int(senti),
                            'target_prob':float(prob)
                            })
            all_data.append({
                            'text':line,
                            'sentiment':int(senti),
                            'target_prob':float(prob)
                            })
        logger.info("  %s = %s", 'Number target similar', len(res))
        logger.info("  %s = %s", 'Number target similar and > 0.5', len(mani))
        res = sorted(res, key=lambda x: x['target_prob'], reverse=True)
        all_data = sorted(all_data, key=lambda x: x['target_prob'], reverse=True)
        mani = sorted(mani, key=lambda x: x['target_prob'], reverse=True)
        if not os.path.exists(self.args.scond_classifier_output):
            os.makedirs(self.args.scond_classifier_output)
        with open(os.path.join(self.args.scond_classifier_output,'target_only.json'), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent='\t')

        with open(os.path.join(self.args.scond_classifier_output,'source_target.json'), "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent='\t')
        with open(os.path.join(self.args.scond_classifier_output,'target_with_0.5.json'), "w", encoding="utf-8") as f:
            json.dump(mani, f, ensure_ascii=False, indent='\t')
        logger.info("Prediction Done!")
