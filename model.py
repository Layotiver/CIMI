import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig


class Bert_stack(nn.Module):
    def __init__(self, args):
        super(Bert_stack, self).__init__()
        self.args = args
        self.device = torch.device("cuda", args.device)
        self.config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = args.max_length

        self.fc_g1 = nn.LSTM(input_size=768 * 2, hidden_size=64, batch_first=True, num_layers=1)
        self.fc_g2 = nn.Linear(64, 16)
        self.fc_g3 = nn.Linear(16, 2)
        self.fc_e1 = nn.Linear(768, 64)
        if self.args.dataset == "hate":
            self.fc_e2 = nn.Linear(64, 3)
        else:
            self.fc_e2 = nn.Linear(64, 2)
        self.sigma1 = nn.Parameter(torch.Tensor([1]))
        self.sigma2 = nn.Parameter(torch.Tensor([1]))

        if args.train_stack:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.fc_e1.parameters():
                param.requires_grad = False
            for param in self.fc_e2.parameters():
                param.requires_grad = False