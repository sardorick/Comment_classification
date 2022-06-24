# from model_bert import Classifier
import torch
import numpy as np
import pandas as pd
# from transformers import BertModel, BertTokenizer
import argparse
from model_bert import model
from data_handler_new import tokenizer, train_loader

# bert_model = BertModel.from_pretrained('bert-base-uncased')
# model = Classifier(bert_model)

model.load_state_dict(torch.load('model.pth')) 

df = pd.read_csv('data/train.csv').drop(columns='id')
sample = df.sample(500).reset_index(drop=True)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocessing(comment, tokenizer):
    inputs = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=36,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]

    return {
            'ids': torch.tensor(ids, dtype=torch.long).unsqueeze(0),
            'mask': torch.tensor(mask, dtype=torch.long).unsqueeze(0),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
        }
target_labels = df.drop(columns='comment_text').columns.to_list()

# comment_number = 18
# example_text = sample.comment_text[comment_number]
# # print(example_text)
# sample.drop(columns='comment_text').iloc[comment_number]

x_y = iter(train_loader).next()
y_true = x_y['targets']
logits = model(x_y['ids'], x_y['mask'], x_y["token_type_ids"])
y_true.shape, logits.shape
true_value = y_true.int().numpy()


def predict(model, comment, tokenizer, target_labels):
    
    # preprocessing
    print(comment)
    data = preprocessing(comment, tokenizer)
    ids = data['ids']
    mask = data['mask']
    token_type_ids = data['token_type_ids']

    # prediction
    model.eval()
    with torch.no_grad():
        logits = model(ids, mask, token_type_ids)
        true_false = torch.sigmoid(logits)

    # if true_false.sum() == 0:
    #     print('The message is not toxic')

    # else: 
    #     return target_labels[true_value]


    return true_false

# CLI
# parser = argparse.ArgumentParser(description='Toxic comment classifier')
# parser.add_argument('test_data', type=str, help='Input comment to classify')
# args = parser.parse_args()
# test_data = args.test_data

example = "I can't make any real suggestions on improvement - I wondered if the section statistics should be later on, or a subsection of """"types of accidents""""  -I think the references may need tidying so that they are all in the exact same format ie date format etc. I can do that later on, if no-one else does first - if you have any preferences for formatting style on references or want to do it yourself please let me know."
print(predict(model, example, tokenizer, target_labels))
    