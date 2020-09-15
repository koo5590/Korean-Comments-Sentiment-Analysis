import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification,  AutoTokenizer
from LSTMModel import SentimentClassifier

from utils import init_logger, binary_accuracy, load_tokenizer, compute_metrics
from data_loader import build_loader


logger = logging.getLogger(__name__)

class Tester:
    def __init__(self, args, tokenizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.args = args

        if args.model_2:
            self.model = SentimentClassifier(args).to(self.device)
            self.model.load_state_dict(torch.load(args.test_model_dir, map_location=self.device))
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(args.test_model_dir).to(self.device)

    def test(self):
        logger.info("***** Model Loaded *****")
        test_loader = build_loader(self.args, self.tokenizer, 'test')
        preds = None
        out_label_ids = None
        nb_eval_steps = 0

        self.model.eval()

        for batch in tqdm(test_loader, desc="Predicting"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                            "labels": None}
                outputs = self.model(**inputs)
                logits = outputs[0]
            
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

        # Write to output file
        with open(self.args.test_output_file, "w", encoding="utf-8") as f:
            for pred in preds:
                f.write("{}\n".format(pred))

        logger.info("Prediction Done!")
