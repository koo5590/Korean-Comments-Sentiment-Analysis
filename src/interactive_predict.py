import torch
import torch.nn as nn
from utils import clean, load_tokenizer
from LSTMModel import SentimentClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BertModel
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

def convert_example_to_features(sentence, tokenizer):
    max_seq_length = 128
    tokens_a = tokenizer.tokenize(clean(sentence))
    
    _truncate_seq_pair(tokens_a,"", max_seq_length - 3)

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

    # print("*** Example ***")
    # print("tokens: %s" % " ".join(
    #         [str(x) for x in tokens]))
    # print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    # print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    # print(
    #         "segment_ids: %s" % " ".join([str(x) for x in token_type_ids]))
    tensors = (torch.tensor(input_ids),
                torch.tensor(input_mask),
                torch.tensor(token_type_ids)
            )
    return tensors

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def interactive_predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("***** Loading... *****")
    start = time.time()
    tokenizer = load_tokenizer(args)
    if args.model_2:    
        model = SentimentClassifier(args).to(device)
        model.load_state_dict(torch.load(args.test_model_dir, map_location=device))
    elif args.model_4:
        model = BertModel.from_pretrained(args.test_model_dir).to(device)
        dropout = nn.Dropout(0.1).to(device)
        classifier = nn.Linear(768, 2).to(device)
    else:
        config = AutoConfig.from_pretrained(args.bert_type,
                                                        num_labels=2, 
                                                        finetuning_task='nsmc',
                                                        id2label={str(i): label for i, label in enumerate([0, 1])},
                                                        label2id={label: i for i, label in enumerate([0, 1])})
        model = AutoModelForSequenceClassification.from_pretrained(args.test_model_dir, config=config).to(device)
    end = time.time()
    logger.info(f"***** Model Loaded: It takes {end-start:.2f} sec *****")


    model.eval()
    while True:
        sentence = input('\n문장을 입력하세요: ')
        tensors = convert_example_to_features(sentence, tokenizer)
        batch = tuple(t.to(device) for t in tensors)
     
        with torch.no_grad():     
            inputs = {"input_ids": batch[0].unsqueeze(0),
                        "attention_mask": batch[1].unsqueeze(0),
                        "token_type_ids": batch[2].unsqueeze(0)
                    }
            outputs = model(**inputs)

            if args.model_4:
                pooled_output = outputs[1]
                logits = classifier(pooled_output)
            else:
                logits = outputs[0]
       
        prob = np.max(torch.sigmoid(logits).detach().cpu().numpy())
        logits = logits.detach().cpu().numpy()
        if np.argmax(logits) == 1:
            print(f'{prob*100:.0f}% 확률로 긍정 문장입니다.')
        else:
            print(f'{prob*100:.0f}% 확률로 부정 문장입니다.')

            
