import torch
import torch.nn as nn
from transformers import BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

bert_model = BertModel.from_pretrained('bert-base-uncased')

class Classifier(nn.Module):
    def __init__(self, bert_model):
        super(Classifier, self).__init__()
        self.emb = bert_model # creating the embeddings (with emb dim == 768)
        self.fc = nn.Linear(768, 6)
    
    def forward(self, ids, mask, token_type_ids):
        output = self.emb(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output = output.pooler_output
        output = self.fc(output)
        return output

model = Classifier(bert_model)
