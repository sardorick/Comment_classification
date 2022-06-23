import pandas as pd
import numpy as np
import torch
import spacy
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText
np.random.seed(0)

nlp = spacy.load('en_core_web_sm')
fasttext = FastText("simple")
# load the dataset

def load_data(pth):
    df = pd.read_csv(pth)
    df.drop(columns=['id'], inplace=True)
    return df

# split it to train and test

def train_test_split(df, train_size=0.8):
    df_idx = [i for i in range(len(df))]
    np.random.shuffle(df_idx)
    len_train = int(len(df) * train_size)
    df_train = df.iloc[:len_train].reset_index(drop=True)
    df_test = df.iloc[len_train:].reset_index(drop=True)
    return df_train, df_test

# preprocess

def preprocessing(sentence):
    """
    Clean texts, get rid of symbols, lower case the words, spaces
    return token list
    """
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space and not token.is_punct and not token.is_bracket and not 
    token.like_email and not token.is_currency and not token.is_digit and not token.like_url and token.is_ascii]
    return tokens

def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]

def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes)) * [padding_index]
    return output[:max_seq_len]

###################

df = load_data('data/train.csv')
# print(df.head())
train_df, test_df = train_test_split(df)
# print(len(train_df), len(test_df))
