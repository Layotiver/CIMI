import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.args = args
        self.device = torch.device("cuda", args.device)
        self.config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.fc_e1 = nn.Linear(768, 64)
        self.fc_e2 = nn.Linear(64, 2)

    def tokenize(self, prompt):
        llm_input = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=self.args.max_length, truncation=True).to(self.device)
        return llm_input

    def forward_ids_bert(self, input_ids, attention_mask, **kwargs):
        llm_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return llm_output

    def forward_embeds_bert(self, embeds, llm_input):
        extended_attention_mask = self.bert.get_extended_attention_mask(llm_input["attention_mask"], llm_input["input_ids"].shape, self.device)
        llm_output = self.bert.encoder(
            embeds,
            attention_mask=extended_attention_mask,
            output_attentions=self.bert.config.output_attentions,
            output_hidden_states=self.bert.config.output_hidden_states,
        )
        return llm_output

    def forward_pooler(self, x):
        return self.bert.pooler(x)

    def forward_linear(self, x):
        x = F.elu(self.fc_e1(x))
        x = F.softmax(self.fc_e2(x), dim=-1)
        return x

    def forward(self, x):
        llm_input = self.tokenize(x)
        llm_output = self.forward_ids_bert(**llm_input)
        pooler_output = llm_output["pooler_output"]
        ret = self.forward_linear(pooler_output)
        return ret

class CIMI(nn.Module):
    def __init__(self):
        super(CIMI, self).__init__()
        self.fc_g1 = nn.LSTM(input_size=768 * 2, hidden_size=64, batch_first=True, num_layers=1)
        self.fc_g2 = nn.Linear(64, 16)
        self.fc_g3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc_g1(x)[0]
        x = F.leaky_relu(self.fc_g2(x))
        x = F.softmax(self.fc_g3(x), dim=-1)
        return x
