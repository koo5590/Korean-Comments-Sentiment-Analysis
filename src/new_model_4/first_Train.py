import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoConfig, BertModel

import os
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

from new_model_4.first import BERTDANN
from utils import compute_metrics

logger = logging.getLogger(__name__)

class FirstTrainer:
    def __init__(self, args, tokenizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.tokenizer = tokenizer

        self.config = AutoConfig.from_pretrained(self.args.bert_type,
                                                        num_labels=2, 
                                                        finetuning_task='nsmc',
                                                        id2label={str(i): label for i, label in enumerate([0, 1])},
                                                        label2id={label: i for i, label in enumerate([0, 1])},
                                                        return_dict=True)
        self.model = BertModel.from_pretrained(self.args.bert_type, config=self.config).to(self.device)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob).to(self.device)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(self.config.hidden_dropout_prob),
        #     nn.Linear(768, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 2),
        # ).to(self.device)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels).to(self.device)
       
    def train(self):
            train_dataset = BERTDANN(self.args, self.tokenizer, 'train')
            train_dataloader = DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.args.train_batch_size)
            valid_dataset = BERTDANN(self.args, self.tokenizer, 'valid')
            valid_dataloader = DataLoader(dataset=valid_dataset, sampler=RandomSampler(valid_dataset), batch_size=self.args.eval_batch_size)
            
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
            # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.max_steps, num_cycles=3)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
            loss_fct = nn.CrossEntropyLoss()
            
            logger.info("***** Running training *****")
            logger.info("  lr = %f", self.args.learning_rate)
            logger.info("  Num examples = %d", len(train_dataloader))
            logger.info("  Num data = %d | movie = %d | sports = %d", len(train_dataset), len(train_dataset.movie_data), len(train_dataset.sports_data))
            logger.info("  Num Epochs = %d", self.args.num_train_epochs)
            logger.info("  Total train batch size = %d", self.args.train_batch_size)
            logger.info("  Total optimization steps = %d", t_total)
            logger.info("  Logging steps = %d", self.args.logging_steps)
            df_args = pd.DataFrame.from_dict([{
                        'initial lr': self.args.learning_rate,
                        "Num examples": len(train_dataloader),
                        'num_data(movie/sports)': f'{len(train_dataset)} ({len(train_dataset.movie_data)}/{len(train_dataset.sports_data)})',
                        'Max_Epochs': self.args.num_train_epochs,
                        'train_batch_size': self.args.train_batch_size,
                        'Total_optimization_steps': t_total,
                        'logging_steps': self.args.logging_steps,
                        'scheduler' : 'get_linear_schedule_with_warmup',
                        'warmup_steps': self.args.warmup_steps,
                        'num_cycles': 3
                    }])
            if not os.path.exists(self.args.first_domain_classifier_output):
                os.makedirs(self.args.first_domain_classifier_output)
            df_args.to_csv(f'{self.args.first_domain_classifier_output}/args.csv', sep='\t')
            self.model.zero_grad()
            best_valid_loss = float('inf')
            global_step = 0
            early_cnt = 0
            training_stats = []
            try:
                for epoch_idx in range(int(self.args.num_train_epochs)):
                    logger.info(f"========== {epoch_idx + 1} : {self.args.num_train_epochs} ==========")
                    
                    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                    epoch_train_loss, epoch_valid_loss = 0, 0
                    epoch_valid_accuracy, valid_cnt = 0, 0

                    for step, batch in enumerate(epoch_iterator):
                        
                        self.model.train()
                        optimizer.zero_grad()

                        batch_sentiment = tuple(t.to(self.device) for t in batch)  
                        inputs = {'input_ids': batch_sentiment[0],
                                'attention_mask': batch_sentiment[1],
                                'output_hidden_states': True
                                }
                        labels = batch_sentiment[2]

                        source_outputs = self.model(**inputs)

                        pooled_output = source_outputs[1]
                        # pooled_output = self.dropout(pooled_output)
                        logits = self.classifier(pooled_output)
                        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

                        if self.args.gradient_accumulation_steps > 1:
                            loss = loss / self.args.gradient_accumulation_steps

                        loss.backward()

                        epoch_train_loss += loss.item()
                        
                        if (step + 1) % self.args.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            optimizer.step()
                            scheduler.step() 
                            self.model.zero_grad()
                            global_step += 1 
                            
                            if self.args.warmup_steps > 0 and global_step % self.args.warmup_steps == 0:
                                logger.info(f'################ computed lr: {scheduler.get_last_lr()} | steps: {global_step} ')
                            if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                                logger.info(f'################ computed lr: {scheduler.get_last_lr()[0]} | steps: {global_step} ')
                                valid_loss, valid_accuracy = self.evaluate(valid_dataloader, "valid")
                                epoch_valid_loss += valid_loss
                                epoch_valid_accuracy += valid_accuracy
                                valid_cnt += 1
                                training_stats.append(
                                    {'epoch': epoch_idx + 1,
                                    'training_loss': epoch_train_loss / (step + 1) ,
                                    'valid_loss': valid_loss,
                                    'steps': global_step,
                                    'valid_loss': valid_loss,
                                    'valid_accuracy': valid_accuracy,
                                    'lr': scheduler.get_last_lr()[0]
                                    }
                                )
                                
                            if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                                if valid_loss < best_valid_loss:
                                    best_valid_loss = valid_loss
                                    self.save_model(optimizer, best_valid_loss)
                                    early_cnt = 0
                                else:
                                    early_cnt += 1
                        
                        if early_cnt > self.args.early_cnt - 1:
                                logger.info('training session has been early stopped')
                                df_stats = pd.DataFrame(data=training_stats, )
                                df_stats = df_stats.set_index('epoch')
                                df_stats.to_csv(f'{self.args.first_domain_classifier_output}/stats.csv', sep='\t', index=True)
                                break
                        
                        if 0 < self.args.max_steps < global_step:
                                epoch_iterator.close()
                                break
                    if early_cnt > self.args.early_cnt - 1:
                        break
                    if 0 < self.args.max_steps < global_step:
                        df_stats = pd.DataFrame(data=training_stats, )
                        df_stats = df_stats.set_index('epoch')
                        df_stats.to_csv(f'{self.args.first_domain_classifier_output}/stats.csv', sep='\t', index=True)
                        break
            
                    # epoch_train_loss = epoch_train_loss / (step + 1)
                    # epoch_valid_loss = epoch_valid_loss / valid_cnt
                    # epoch_valid_accuracy = epoch_valid_accuracy / valid_cnt
                    # if epoch_valid_loss < best_valid_loss:
                    #     best_valid_loss = epoch_valid_loss    
                    #     # self.save_model(optimizer, best_valid_loss)
                    #     early_cnt = 0
                    # else:
                    #     early_cnt += 1
                    # logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'train_loss', epoch_train_loss)
                    # logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'valid_loss', epoch_valid_loss)
                    # logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'valid_accuracy', epoch_valid_accuracy)
                    
                    # if early_cnt > self.args.early_cnt - 1:
                    #     logger.info('training session has been early stopped')
                    #     df_stats = pd.DataFrame(data=training_stats, )
                    #     df_stats = df_stats.set_index('epoch')
                    #     df_stats.to_csv(f'{self.args.first_domain_classifier_output}/stats.csv', sep='\t', index=True)
                    #     break
            except KeyboardInterrupt as e:
                logger.info(e)
                df_stats = pd.DataFrame(data=training_stats)
                df_stats = df_stats.set_index('epoch')
                df_stats.to_csv(f'{self.args.first_domain_classifier_output}/stats.csv', sep='\t', index=True)
                return
            except Exception as e:
                logger.info(e)
                df_stats = pd.DataFrame(data=training_stats)
                df_stats = df_stats.set_index('epoch')
                df_stats.to_csv(f'{self.args.first_domain_classifier_output}/stats.csv', sep='\t', index=True)
                return

    def evaluate(self, eval_dataloader, mode):
        logger.info("  ***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        preds = None
        out_label_ids = None
        loss_fct = nn.CrossEntropyLoss()
        nb_eval_steps = 0

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1]
                        }
                labels = batch[2]
                outputs = self.model(**inputs)
                pooled_output = outputs[1]
                # pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                
                eval_loss += loss.item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)

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
        if not os.path.exists(self.args.first_domain_classifier_output):
            os.makedirs(self.args.first_domain_classifier_output)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.first_domain_classifier_output)

        torch.save(self.args, os.path.join(self.args.first_domain_classifier_output, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.first_domain_classifier_output)