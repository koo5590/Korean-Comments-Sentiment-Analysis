import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer, AutoConfig

from LSTMModel import SentimentClassifier
from data_loader import build_loader
from utils import compute_metrics, format_time, init_logger

import os
from tqdm import tqdm
import logging
import numpy as np

logger = logging.getLogger(__name__)

# TODO Kobert의 tokenizer 는?
class Trainer:
    def __init__(self, args, tokenizer):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'device: {self.device}')

        self.tokenizer = tokenizer
        
        if args.model_2:    
            self.model = SentimentClassifier(self.args).to(self.device)
            if args.second_finetuning:
                self.model.load_state_dict(torch.load(args.test_model_dir, map_location=self.device))
        else:
            self.config = AutoConfig.from_pretrained(self.args.bert_type,
                                                            num_labels=2, 
                                                            finetuning_task='nsmc',
                                                            id2label={str(i): label for i, label in enumerate([0, 1])},
                                                            label2id={label: i for i, label in enumerate([0, 1])})
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.bert_type, config=self.config).to(self.device)
        
    def train(self):
            train_dataset, valid_dataset, train_dataloader, valid_dataloader = build_loader(self.args, self.tokenizer, 'train')
            
            if self.args.max_steps > 0:
                t_total = self.args.max_steps
                self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            else:
                t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_dataloader))
            logger.info("  Num train data = %d", len(train_dataset))
            logger.info("  Num valid data = %d | 0.01", len(valid_dataset))
            logger.info("  Num Epochs = %d", self.args.num_train_epochs)
            logger.info("  Total train batch size = %d", self.args.train_batch_size)
            logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)
            logger.info("  Logging steps = %d", self.args.logging_steps)
            logger.info("  saving steps = %d", self.args.save_steps)

            self.model.zero_grad()
            best_valid_loss = float('inf')
            global_step = 0

            for epoch_idx in range(int(self.args.num_train_epochs)):
                logger.info(f"========== {epoch_idx + 1} : {self.args.num_train_epochs} ==========")
                
                epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                epoch_train_loss, epoch_valid_loss = 0, 0
                epoch_valid_accuracy, valid_cnt = 0, 0

                for step, batch in enumerate(epoch_iterator):
                    self.model.train()
                    optimizer.zero_grad()

                    batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                    inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'labels': batch[3]}
                    inputs['token_type_ids'] = batch[2]
                    outputs = self.model(**inputs)
                    loss = outputs[0]

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    loss.backward()

                    epoch_train_loss += loss.item()
                    
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1
                        
                        if self.args.logging_steps > 0 and global_step  % self.args.logging_steps == 0:
                            valid_loss, valid_accuracy = self.evaluate(valid_dataloader, "valid")
                            epoch_valid_loss += valid_loss
                            epoch_valid_accuracy += valid_accuracy
                            valid_cnt += 1
                        
                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            if valid_loss < best_valid_loss:
                                self.save_model(optimizer, best_valid_loss)
                    
                    if 0 < self.args.max_steps < global_step:
                        epoch_iterator.close()
                        break
                
                if 0 < self.args.max_steps < global_step:
                    break

                epoch_train_loss = epoch_train_loss / global_step 
                epoch_valid_loss = epoch_valid_loss / valid_cnt
                epoch_valid_accuracy = epoch_valid_accuracy / valid_cnt
                if epoch_valid_loss < best_valid_loss:
                    best_valid_loss = epoch_valid_loss    
                    self.save_model(optimizer, best_valid_loss)
                logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'train_loss', epoch_train_loss)
                logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'valid_loss', epoch_valid_loss)
                logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'valid_accuracy', epoch_valid_accuracy)

    def evaluate(self, eval_dataloader, mode):
        logger.info("  ***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results['loss'], results['acc']

    def save_model(self, optimizer, loss):
        if not os.path.exists(self.args.save_model_dir):
            os.makedirs(self.args.save_model_dir)
        if self.args.model_2:
            torch.save(self.model.state_dict(), os.path.join(self.args.save_model_dir, 'model_sports.pt'))
        else:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(self.args.save_model_dir)
        
        torch.save(self.args, os.path.join(self.args.save_model_dir, 'training_args.bin'))
        torch.save(self.args, os.path.join(self.args.save_model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.save_model_dir)