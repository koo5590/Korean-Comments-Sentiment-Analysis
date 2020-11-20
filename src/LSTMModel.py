from transformers import AutoModel
import torch
from torch import nn


"""
    BI-LSTM model
"""
class LSTMModel(nn.Module):
    """
        Constructor
        param:
            input_d: dimension of input sequence
            hidden_d: dimension of hidden states
            out_d: dimension of output sequence
            num_layers: number of LSTM layer(s)
            dropout: dropout rate between LSTM layers
    """
    def __init__(self, input_d, hidden_d, out_d, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_d,
                          hidden_size=hidden_d,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=True)
        self.linear = nn.Linear(hidden_d*2, out_d)
        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.linear.weight)

    """
        Forward Propagation
        param:
            input: seq of hidden states of BERT last layer
        return:
            logtis
    """
    def forward(self, input):

        _, (hidden, _) = self.lstm(input)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.linear(hidden)

        return output


"""
    BI-LSTM Classification with BERT final layer hidden states
"""
class SentimentClassifier(nn.Module):
    # arguments: [model_name_or_path, hidden, layer_num, dropout, freeze, freeze_num]
    def __init__(self, args):
        super().__init__()

        # download pre trained bert model
        self.bert = AutoModel.from_pretrained(args.bert_type)

        # input_d, hidden_d, out_d, num_layers, dropout
        self.lstm = LSTMModel(768, 128, 2, 2, 0.2)

        # freeze layers if necessary
        if args.freeze:
            for i in range(10):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False


    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None):

        outputs = self.bert(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            output_attentions=output_attentions)

        logits = self.lstm(outputs[0])
        
        if labels == None:
            return (logits,)

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)

        return loss, logits
