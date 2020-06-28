#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import torch
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from transformers import *
import torch.utils.data as Data
import collections
import os
import random
import tarfile
from torch import nn
import torchtext.vocab as Vocab
import pickle as pk
import torch.nn.functional as F
from IPython.display import display,HTML
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn import CrossEntropyLoss, MSELoss
from torchcrf import CRF
from sklearn import metrics
import joblib
import math
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
import glob
import unicodedata
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook as tqdm
import torch.utils.data as Data
import jieba
import jieba.posseg as pseg
import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold

# In[2]:


def convert_text_to_ids(tokenizer, text, max_len=100):
    if isinstance(text,str):
        output=tokenizer.encode_plus(text,max_length=max_len,pad_to_max_length=True,return_tensors="pt")
        input_ids=output["input_ids"].squeeze(0)
        token_type_ids=output["token_type_ids"].squeeze(0)
        attention_mask=output["attention_mask"].squeeze(0)
    elif isinstance(text,list):
        input_ids,token_type_ids,attention_mask=[],[],[]
        for e in text:
            output=tokenizer.encode_plus(e,max_length=max_len,pad_to_max_length=True,return_tensors="pt")
            input_ids.append(output["input_ids"].squeeze(0))
            token_type_ids.append(output["token_type_ids"].squeeze(0))
            attention_mask.append(output["attention_mask"].squeeze(0))
    else:
        raise Exception('type error')
    return torch.stack(input_ids).long(),torch.stack(token_type_ids).long(),torch.stack(attention_mask).long()        
class RelDataset(Data.Dataset):
    def __init__(self,examples):
        self.input_ids=torch.stack([e['input_ids'] for e in examples]).long()
        self.token_type_ids=torch.stack([e['token_type_ids'] for e in examples]).long()
        self.attention_mask=torch.stack([e['attention_mask'] for e in examples]).long()
        self.rel_label=torch.stack([e['rel_label'] for e in examples]).long()
        self.postag=torch.stack([e['postag'] for e in examples]).long()
        self.feature=torch.stack([e['feature'] for e in examples]).float()
        self.token_vec=np.stack([np.array(e['token_vec']) for e in examples])
        self.word_vec=np.stack([np.array(e['word_vec']) for e in examples])
        self.word_mask=np.stack([np.array(e['word_mask']) for e in examples])
        self.token2docs=[e["token2doc"] for e in examples]
    def __len__(self):
        return self.input_ids.shape[0]
    def __getitem__(self,idx):
        return self.input_ids[idx],self.attention_mask[idx],self.token_type_ids[idx],               self.rel_label[idx],self.postag[idx],self.feature[idx],self.token_vec[idx],self.word_vec[idx],               self.word_mask[idx],self.token2docs[idx] 
class NerDataset(Data.Dataset):
    def __init__(self,examples):
        self.input_ids=torch.stack([e['input_ids'] for e in examples]).long()
        self.token_type_ids=torch.stack([e['token_type_ids'] for e in examples]).long()
        self.attention_mask=torch.stack([e['attention_mask'] for e in examples]).long()
        self.rel_label=torch.stack([e['rel_label'] for e in examples]).long()
        self.labels=torch.stack([e['labels'] for e in examples]).long()
        self.postag=torch.stack([e['postag'] for e in examples]).long()
        self.feature=torch.stack([e['feature'] for e in examples]).float()
        self.token2docs=[e["token2doc"] for e in examples]
    def __len__(self):
        return self.input_ids.shape[0]
    def __getitem__(self,idx):
        return self.input_ids[idx],self.attention_mask[idx],self.token_type_ids[idx],               self.rel_label[idx],self.labels[idx],self.postag[idx],self.feature[idx],self.token2docs[idx]  
import unicodedata
def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False
def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False
def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def build_tfidf_svd_matrix(texts,n_output,tfidf_vec=None,svd=None):
    corpus=[]
    for text in tqdm(texts):
#         print(text)
        words=word_segment(str(text['text']))
#         print(words)
        use_words=[]
        for word in words:
            if word in stop_words:
                continue
            use_words.append(word)
#         print(use_words)
        corpus.append(" ".join(use_words))
    print(len(corpus))
    print(corpus[0])
    if tfidf_vec is None:
        tfidf_vec=TfidfVectorizer()
        tfidf_matrix=tfidf_vec.fit_transform(corpus)
    else:
        tfidf_matrix=tfidf_vec.transform(corpus)
    if svd is None:
        svd=TruncatedSVD(n_components=n_output,n_iter=7,random_state=42)
        tf_idf_svd=svd.fit_transform(tfidf_matrix)
    else:
        tf_idf_svd=svd.transform(tfidf_matrix)
    return tf_idf_svd,tfidf_vec,svd
def word_segment(sentence):
    words=jieba.cut(sentence)
    return ",".join(words).split(",")
stop_words=set()
def load_stopwords():
    with open("./middle_data/stopwords.txt","r",encoding="UTF-8") as r:
        for line in r.readlines():
            stop_words.add(line.strip())
load_stopwords()
def remove_stopwords(word_list):
    res=[]
    for word in word_lists:
        if word not in stop_words:
            res.append(word)
    return ' '.join(res)
def clean_text(string):
    return string.replace(' ', '').replace('\n', '').replace('\u3000', '')
def _convert_example_to_record(example,
                               tokenizer):
    if example.__contains__('spo_list'):
        spo_list = example['spo_list']
    else:
        spo_list = []
    text_raw = example['text']
    sub_text = []
    buff = ""
    for char in text_raw:
        if is_chinese_char(ord(char)) or is_punctuation(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)
    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
        else:
            continue
        break
    #  find all entities and tag them with corresponding "B"/"I" labels
    labels_list=[[0
              for i in range(len(tokens))] for i in range(len(id2rels))]
    rel_labels=[]
    for spo in spo_list:
        for spo_object in spo['object'].keys():
            if not spo['predicate'] in special_rels:
                rel_label=rels2id[spo["predicate"]]
            else:
                rel_label=rels2id[spo["predicate"]+"_"+spo_object] 
            labels = labels_list[rel_label] #复杂类的不同part还是不会被归到一个，以后再讲
            label_subject = label2ids['B-SUB']
            label_object = label2ids['B-OBJ']
            subject_sub_tokens = tokenizer.tokenize(spo['subject'])
            object_sub_tokens = tokenizer.tokenize(spo['object'][
                spo_object])
            forbidden_index = None
            if len(subject_sub_tokens) > len(object_sub_tokens):
                for index in range(
                        len(tokens) - len(subject_sub_tokens) + 1):
                    if tokens[index:index + len(
                            subject_sub_tokens)] == subject_sub_tokens:
                        labels[index]=label_subject
                        for i in range(len(subject_sub_tokens) - 1):
                            labels[index + i + 1]=label_subject+1
                        forbidden_index = index
                        break

                for index in range(
                        len(tokens) - len(object_sub_tokens) + 1):
                    if tokens[index:index + len(
                            object_sub_tokens)] == object_sub_tokens:
                        if forbidden_index is None:
                            labels[index]=label_object
                            for i in range(len(object_sub_tokens) - 1):
                                labels[index + i + 1]=label_object+1
                            break
                        # check if labeled already
                        elif index < forbidden_index or index >= forbidden_index + len(
                                subject_sub_tokens):
                            labels[index]=label_object
                            for i in range(len(object_sub_tokens) - 1):
                                labels[index + i + 1]=label_object+1
                            break

            else:
                for index in range(
                        len(tokens) - len(object_sub_tokens) + 1):
                    if tokens[index:index + len(
                            object_sub_tokens)] == object_sub_tokens:
                        labels[index]=label_object
                        for i in range(len(object_sub_tokens) - 1):
                            labels[index + i + 1]=label_object+1
                        forbidden_index = index
                        break

                for index in range(
                        len(tokens) - len(subject_sub_tokens) + 1):
                    if tokens[index:index + len(
                            subject_sub_tokens)] == subject_sub_tokens:
                        if forbidden_index is None:
                            labels[index]=label_subject
                            for i in range(len(subject_sub_tokens) - 1):
                                labels[index + i + 1]=label_subject+1
                            break
                        elif index < forbidden_index or index >= forbidden_index + len(
                                object_sub_tokens):
                            labels[index]=label_subject
                            for i in range(len(subject_sub_tokens) - 1):
                                labels[index + i + 1]=label_subject+1
                            break
            labels_list[rel_label]=labels
            if rel_label not in rel_labels:
                rel_labels.append(rel_label)

    return tok_to_orig_start_index,tok_to_orig_end_index,orig_to_tok_index,tokens,labels_list,rel_labels
def _convert_example_to_cls_record(example,
                               tokenizer):
    if example.__contains__('spo_list'):
        spo_list = example['spo_list']
    else:
        spo_list = []
    text_raw = example['text']
    sub_text = []
    buff = ""
    for char in text_raw:
        if is_chinese_char(ord(char)) or is_punctuation(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)
    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
        else:
            continue
        break
    #  find all entities and tag them with corresponding "B"/"I" labels
    labels_list=[]
    rel_labels=[]
    for spo in spo_list:
        for spo_object in spo['object'].keys():
            labels = [0
                  for i in range(len(tokens))]  # initialize tag
            if not spo['predicate'] in special_rels:
                rel_label=rels2id[spo["predicate"]]
            else:
                rel_label=rels2id[spo["predicate"]+"_"+spo_object] 
            rel_labels.append(rel_label)

    return tok_to_orig_start_index,tok_to_orig_end_index,orig_to_tok_index,tokens,rel_labels
def create_cls_example(tokens,rel,token2doc,postag,tfidf_svd,word_vec,token_vec,tokenizer):
        word_vec=copy.deepcopy(word_vec)
        token_vec=copy.deepcopy(token_vec)
        if len(tokens)>max_len-2:
                tokens=tokens[:(max_len-2)]
                token2doc=[e[:(max_len-2)] if ind<2 else e for ind,e in enumerate(token2doc)]
        tag=[postag[token2doc[0][idx]] for idx in range(len(tokens))]
        tokenvec=[token_vec[token2doc[0][idx]] for idx in range(len(tokens))]
        tag=[pos2id_BIO[tokenizer.cls_token]]+tag+[pos2id_BIO[tokenizer.sep_token]]
        tokenvec=[word2id[tokenizer.cls_token]]+tokenvec+[word2id[tokenizer.sep_token]]
        full_tokens=[tokenizer.cls_token]+tokens+[tokenizer.sep_token]
        token_type_ids=[0]*len(full_tokens)
        attention_mask=[1]*len(token_type_ids)
        cur_len=len(full_tokens)
        if cur_len<max_len:
            full_tokens+=[tokenizer.pad_token]*(max_len-cur_len)
            attention_mask+=[0]*(max_len-cur_len)
            token_type_ids+=[0]*(max_len-cur_len)
            tag+=[pos2id_BIO[tokenizer.pad_token]]*(max_len-cur_len)
            tokenvec+=[word2id[tokenizer.pad_token]]*(max_len-cur_len)
        if len(word_vec)>max_len//2:
            word_vec=word_vec[:max_len//2]
        word_mask=[1]*len(word_vec)
        if len(word_vec)<max_len//2:
            word_mask+=[0]*(max_len//2-len(word_vec))
            word_vec+=[word2id[tokenizer.pad_token]]*(max_len//2-len(word_vec))

        full_ids=tokenizer.convert_tokens_to_ids(full_tokens)
        if len(rel)>0:
            example={"input_ids":torch.tensor(full_ids,dtype=torch.long),"token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),                    "attention_mask":torch.tensor(attention_mask,dtype=torch.long),
                    "rel_label":(F.one_hot(torch.tensor(rel),len(id2rels)).sum(dim=0)!=0).long(),
                     "postag":torch.tensor(tag).long(),"feature":torch.tensor(tfidf_svd).float(),\
                     "token_vec":tokenvec,"word_vec":word_vec,"word_mask":word_mask,"token2doc":token2doc}
        else:
            example={"input_ids":torch.tensor(full_ids,dtype=torch.long),"token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),                    "attention_mask":torch.tensor(attention_mask,dtype=torch.long),
                     "rel_label":torch.zeros(len(id2rels)).long(),
                     "postag":torch.tensor(tag).long(),"feature":torch.tensor(tfidf_svd).float(),\
                     "token_vec":tokenvec,"word_vec":word_vec,"word_mask":word_mask,"token2doc":token2doc}  
        return example
def create_example(tokens,rel,labels,token2doc,tokenizer,rel_text,tfidf_svd,postag,rel_postag,rel_token2doc):
        tag=[postag[token2doc[0][idx]] for idx in range(len(tokens))]
        tag=[pos2id_BIO[tokenizer.cls_token]]+tag+[pos2id_BIO[tokenizer.sep_token]]
        rel_tag=[rel_postag[rel_token2doc[idx]] for idx in range(len(rel_text))]
        rel_tag=rel_tag+[pos2id_BIO[tokenizer.sep_token]]
        tag=tag+rel_tag
        second_token=rel_text
        full_tokens=[[tokenizer.cls_token]+tokens+[tokenizer.sep_token],second_token+[tokenizer.sep_token]]
        full_labels=[[label2ids[tokenizer.cls_token]]+labels+[label2ids[tokenizer.sep_token]],[label2ids["[category]"]]*len(second_token)+[label2ids[tokenizer.sep_token]]]
        token_type_ids=[0]*len(full_tokens[0])+[1]*len(full_tokens[1])
        attention_mask=[1]*len(token_type_ids)
        full_tokens=full_tokens[0]+full_tokens[1]
        full_labels=full_labels[0]+full_labels[1]
        cur_len=len(full_labels)
        if cur_len<max_len:
            full_tokens+=[tokenizer.pad_token]*(max_len-cur_len)
            full_labels+=[label2ids[tokenizer.pad_token]]*(max_len-cur_len)
            attention_mask+=[0]*(max_len-cur_len)
            token_type_ids+=[0]*(max_len-cur_len)
            tag+=[pos2id_BIO[tokenizer.pad_token]]*(max_len-cur_len)
        full_ids=tokenizer.convert_tokens_to_ids(full_tokens)
        example={"input_ids":torch.tensor(full_ids,dtype=torch.long),"token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),                "attention_mask":torch.tensor(attention_mask,dtype=torch.long),"labels":torch.tensor(full_labels,dtype=torch.long),
                "rel_label":F.one_hot(torch.tensor(rel),num_classes=len(id2rels)),"postag":torch.tensor(tag).long(),"feature":torch.tensor(tfidf_svd).float(),\
                 "token2doc":token2doc}
        return example
def index_token(text_raw):
    sub_text = []
    buff = ""
    for char in text_raw:
        if is_chinese_char(ord(char)) or is_punctuation(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)
    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
        else:
            continue
        break
    return tok_to_orig_start_index,tok_to_orig_end_index,orig_to_tok_index
def find_subs_obs(token_label_pred,text):
    subs=[]
    sub_offsets=[]
    obs=[]
    ob_offsets=[]
    index=0
    while index<token_label_pred.shape[0]:
        if token_label_pred[index]==3:
            subs.append(text[index])
            index+=1
            while(index<token_label_pred.shape[0] and (token_label_pred[index]==4 or token_label_pred[index]==3)):
                subs[-1]+=(text[index])
                index+=1
            sub_offsets.append([index-len(subs[-1]),index])
            index-=1
        if index<token_label_pred.shape[0] and token_label_pred[index]==1:
            obs.append(text[index])
            index+=1
            while(index<token_label_pred.shape[0] and (token_label_pred[index]==2 or token_label_pred[index]==1)):
                obs[-1]+=(text[index])
                index+=1
            ob_offsets.append([index-len(obs[-1]),index])
            index-=1
        index+=1
    if len(subs)==0:
        subs.append('')
        sub_offsets.append([0,0])
    if len(obs)==0:
        obs.append('')
        ob_offsets.append([0,0])
    return subs,obs,sub_offsets,ob_offsets


# In[3]:


do_lower_case=True
max_len=256
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# bert_dir="./bert-pytorch-chinese/"
# vocab="bert-base-chinese-vocab.txt"
# config_file="bert_config.json"
bert_dir="./roberta-zh-wwm-pytorch/"
vocab="vocab.txt"
config_file="bert_config.json"
tokenizer=BertTokenizer.from_pretrained(os.path.join(bert_dir,vocab),do_lower_case=do_lower_case)


# In[4]:


text_data=[]
with open("./dataset/train_data/train_data.json","r") as r:
    raw_data=r.readlines()
    for d in raw_data:
        text_data.append(json.loads(d))
print(len(text_data))
with open("./dataset/dev_data/dev_data.json","r") as r:
    raw_data=r.readlines()
    for d in raw_data:
        text_data.append(json.loads(d))
print(len(text_data))
test_text_data=[]
with open("./dataset/test1_data/test1_data.json","r") as r:
    raw_data=r.readlines()
    for d in raw_data:
        test_text_data.append(json.loads(d))
schema=[]
with open("./dataset/schema.json","r") as r:
    raw_schema=r.readlines()
    for d in raw_schema:
        schema.append(json.loads(d))
rels=set()
special_rels=set()
for e in schema:
        if len(e['object_type'].keys())==1:
            rels.add(e["predicate"])
        else:
            special_rels.add(e["predicate"])
            for key in e['object_type'].keys():
                rels.add(e['predicate']+"_"+key)
if not os.path.exists("./dataset/dict.pk"):
    special_rels=list(special_rels)
    id2rels=list(rels)
    rels2id=dict([(rel,idx) for idx,rel in enumerate(id2rels)])
    id2labels=["O","B-OBJ","I-OBJ","B-SUB","I-SUB","[category]","[SEP]","[CLS]","[PAD]"]
    label2ids=dict([ (label,idx) for idx,label in enumerate(id2labels)])
    pk.dump([special_rels,id2rels,rels2id,id2labels,label2ids],open("./dataset/dict.pk","wb"))
else:
    print("loading dict...")
    special_rels,id2rels,rels2id,id2labels,label2ids=pk.load(open("./dataset/dict.pk","rb"))
id2reltype=[[] for i in range(len(id2rels))]
for e in schema:
    if len(e['object_type'].keys())==1:
        rel=e["predicate"]
        ids=rels2id[rel]
        id2reltype[ids].append(e)
    else:
        for key in e['object_type'].keys():
            rel=e['predicate']+"_"+key
            ids=rels2id[rel]
            temp_e=copy.deepcopy(e)
            poped_keys=[]
            for k in temp_e['object_type'].keys():
                if k!=key:
                    poped_keys.append(k)
            for k in poped_keys:
                 temp_e['object_type'].pop(k)
            id2reltype[ids].append(temp_e)
id2schema=[e[0] for e in id2reltype]
id2rel_text=[[] for i in range(len(id2rels))]
id2rel_rawtext=[[] for i in range(len(id2rels))]
id2rel_token2text=[[] for i in range(len(id2rels))]
for rel in range(len(id2rels)):
    if id2rels[rel].split("_")[0] not in special_rels:
        cls_text=id2schema[rel]['subject_type']+","+id2schema[rel]['predicate']+","+id2schema[rel]['object_type']['@value']
    else:
        cls_text=id2schema[rel]['subject_type']+","+id2schema[rel]['predicate']+","+id2schema[rel]['object_type'][id2rels[rel].split("_")[1]]
    id2rel_text[rel]=tokenizer.tokenize(cls_text)
    id2rel_rawtext[rel]=cls_text
    id2rel_token2text[rel]=index_token(cls_text)[0]
    assert len(id2rel_token2text[rel])==len(id2rel_text[rel])
if not os.path.exists("./middle_data/rel_data_postag.pk"):
    jieba.enable_paddle() 
    jieba.enable_parallel(8)
    rel_cut_words=[]
    rel_cut_tags=[]
    for idx in tqdm(range(len(id2rel_rawtext))):
        words = pseg.lcut(id2rel_rawtext[idx],use_paddle=True) #jieba默认模式
        new_words=[w for w,t in words]
        new_tags=[t for w,t in words]
        rel_cut_words.append([idx,new_words])
        rel_cut_tags.append([idx,new_tags])
    rel_cut_words=[e[1] for e in sorted(rel_cut_words,key=lambda x:x[0])]
    rel_cut_tags=[e[1] for e in sorted(rel_cut_tags,key=lambda x:x[0])]
    rel_data_postag=[]
    for idx in tqdm(range(len(id2rel_rawtext))):
        assert len(id2rel_rawtext[idx].strip())==len("".join(rel_cut_words[idx]))
        indexs=[]
        cur_length=0
        for e in rel_cut_words[idx]:
            indexs.append([cur_length,cur_length+len(e)-1])
            cur_length+=len(e)
        pos_label=np.zeros(len(id2rel_rawtext[idx])).astype(np.int8)
        for i,(b,e) in enumerate(indexs):
            assert (id2rel_rawtext[idx][b]==rel_cut_words[idx][i][0] or _is_whitespace(id2rel_rawtext[idx].strip()[b])                   or _is_whitespace(rel_cut_words[idx][i][0]))                     and (id2rel_rawtext[idx].strip()[e]==rel_cut_words[idx][i][-1]                          or _is_whitespace(id2rel_rawtext[idx].strip()[e])  or _is_whitespace(rel_cut_words[idx][i][-1]))
            pos_label[b+1:e+1]=pos2id_BIO['I-'+rel_cut_tags[idx][i]]
            pos_label[b]=pos2id_BIO['B-'+rel_cut_tags[idx][i]]
        rel_data_postag.append(pos_label)
    pk.dump(rel_data_postag,open("./middle_data/rel_data_postag.pk","wb"))
else:
    print("loading...")
    rel_data_postag=pk.load(open("./middle_data/rel_data_postag.pk","rb"))
special_major_idx=[2,4,22,32,54]
special_affilate_idx=[[] for i in range(len(id2rels))]
special_affilate_idx[2]=[5]
special_affilate_idx[4]=[0]
special_affilate_idx[22]=[51]
special_affilate_idx[32]=[8]
special_affilate_idx[54]=[6,11,14]
special_pass_idx=[0,5,6,8,11,14,51]
[(id2rels[e],[id2rels[t] for t in special_affilate_idx[e]]) for e in special_major_idx],[id2rels[e] for e in special_pass_idx]


# In[5]:


k_split=KFold(n_splits=8,shuffle=True,random_state=20)
num=0
for train_index,test_index in k_split.split(text_data):
    print(num)
    num+=1
    train_text_data=[]
    for i in train_index:
        train_text_data.append(text_data[i])
    test_text_data=[]
    for i in test_index:
        test_text_data.append(text_data[i])
    spo_corpus=[]
    for idx,e in tqdm(enumerate(train_text_data)):
        for spo in e['spo_list']:
            if spo['predicate'] in special_rels:
                for key,value in spo['object'].items():
                    rel=spo['predicate']+"_"+key
                    ob=value
                    sub=spo['subject']
                    sample={'predicate':rel,'object':ob,'subject':sub}
                    spo_corpus.append(sample)
            else:
                for key,value in spo['object'].items():
                    rel=spo['predicate']
                    ob=value
                    sub=spo['subject']
                    sample={'predicate':rel,'object':ob,'subject':sub}
                    spo_corpus.append(sample)
    for idx,e in tqdm(enumerate(test_text_data)):
        self_corpus=[]
        for spo in spo_corpus:
            if spo['subject'] in e['text']  and spo['object'] in e['text'] :
                if (spo['subject']!='' or spo['object']!='') and spo not in self_corpus:
                    self_corpus.append(spo)
        e['plan_spo_list']=e.get('plan_spo_list',[])+self_corpus


# In[16]:


with open("./dataset/train_data/new_train_data.json","w") as w:
    for d in text_data:
        w.write(json.dumps(d, ensure_ascii=False)+"\n")


if not os.path.exists("./dataset/test1_data/new_test1_data.json"):
    spo_corpus=[]
    for idx,e in tqdm(enumerate(text_data)):
        for spo in e['spo_list']:
            if spo['predicate'] in special_rels:
                for key,value in spo['object'].items():
                    rel=spo['predicate']+"_"+key
                    ob=value
                    sub=spo['subject']
                    sample={'predicate':rel,'object':ob,'subject':sub}
                    spo_corpus.append(sample)
            else:
                for key,value in spo['object'].items():
                    rel=spo['predicate']
                    ob=value
                    sub=spo['subject']
                    sample={'predicate':rel,'object':ob,'subject':sub}
                    spo_corpus.append(sample)
    for idx,e in tqdm(enumerate(test_text_data)):
        self_corpus=[]
        for spo in spo_corpus:
            if spo['subject'] in e['text']  and spo['object'] in e['text'] :
                if (spo['subject']!='' or spo['object']!='') and spo not in self_corpus:
                    self_corpus.append(spo)
        e['plan_spo_list']=e.get('plan_spo_list',[])+self_corpus
    with open("./dataset/test1_data/new_test1_data.json","w") as w:
        for e in test_text_data:
            w.write(json.dumps(e, ensure_ascii=False)+'\n')
else:
    print("loading...")
    new_test_text_data=[]
    with open("./dataset/test1_data/new_test1_data.json","r") as r:
        raw_data=r.readlines()
        for d in raw_data:
            new_test_text_data.append(json.loads(d))

if not os.path.exists("./dataset/test2_data/new_test2_data.json"):
    spo_corpus=[]
    for idx,e in tqdm(enumerate(text_data)):
        for spo in e['spo_list']:
            if spo['predicate'] in special_rels:
                for key,value in spo['object'].items():
                    rel=spo['predicate']+"_"+key
                    ob=value
                    sub=spo['subject']
                    sample={'predicate':rel,'object':ob,'subject':sub}
                    spo_corpus.append(sample)
            else:
                for key,value in spo['object'].items():
                    rel=spo['predicate']
                    ob=value
                    sub=spo['subject']
                    sample={'predicate':rel,'object':ob,'subject':sub}
                    spo_corpus.append(sample)
    for idx,e in tqdm(enumerate(test2_text_data)):
        self_corpus=[]
        for spo in spo_corpus:
            if spo['subject'] in e['text']  and spo['object'] in e['text'] :
                if (spo['subject']!='' or spo['object']!='') and spo not in self_corpus:
                    self_corpus.append(spo)
        e['plan_spo_list']=e.get('plan_spo_list',[])+self_corpus
    with open("./dataset/test2_data/new_test2_data.json","w") as w:
        for e in test2_text_data:
            w.write(json.dumps(e, ensure_ascii=False)+'\n')
else:
    print("loading...")
    new_test2_text_data=[]
    with open("./dataset/test2_data/new_test2_data.json","r") as r:
        raw_data=r.readlines()
        for d in raw_data:
            new_test2_text_data.append(json.loads(d))

