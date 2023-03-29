#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
import sys

#DATA_HOME = '/ahc/work3/kosuke-t/data/'
DATA_HOME = sys.argv[-1]

DA_HOME = os.path.join(DATA_HOME, 'WMT/DAseg-wmt-newstest2017/ensembled')
HUME_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/manual-evaluation/hume-testset-round-2.tsv')
# DARR_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/manual-evaluation/RR-seglevel.csv')
SRC_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/sources')
REF_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/references')
HYP_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/system-outputs/newstest2017')

# SRC_himl_HOME = os.path
# REF_himl_HOME = 
# HYP_himl_a_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/himltest17/txt/system-outputs/himltest2017a')
# HYP_himl_b_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/himltest17/txt/system-outputshimltest2017b')
# SAVE_PATH_DARR = os.path.join(DATA_HOME, 'WMT/wmt17_darr.pkl')

SAVE_PATH_DA_GOOD_REDUP = os.path.join(DATA_HOME, 'WMT/wmt17_da_good_redup.pkl')
SAVE_PATH_DA_SEG = os.path.join(DATA_HOME, 'WMT/wmt17_da_seg.pkl')

DIR_NAME = 'WMT15_17_DA_plus_ru_zh'
# DIR_NAME = 'WMT15_17_DA_HUME'
SAVE_SRC_TRAIN = os.path.join(DATA_HOME,'SRHDA/{}/train.src'.format(DIR_NAME))
SAVE_REF_TRAIN = os.path.join(DATA_HOME,'SRHDA/{}/train.ref'.format(DIR_NAME))
SAVE_HYP_TRAIN = os.path.join(DATA_HOME,'SRHDA/{}/train.hyp'.format(DIR_NAME))
SAVE_LABEL_TRAIN = os.path.join(DATA_HOME,'SRHDA/{}/train.label'.format(DIR_NAME))
SAVE_SRC_VALID = os.path.join(DATA_HOME,'SRHDA/{}/valid.src'.format(DIR_NAME))
SAVE_REF_VALID = os.path.join(DATA_HOME,'SRHDA/{}/valid.ref'.format(DIR_NAME))
SAVE_HYP_VALID = os.path.join(DATA_HOME,'SRHDA/{}/valid.hyp'.format(DIR_NAME))
SAVE_LABEL_VALID = os.path.join(DATA_HOME,'SRHDA/{}/valid.label'.format(DIR_NAME))
SAVE_SRC_TEST = os.path.join(DATA_HOME,'SRHDA/{}/test.src'.format(DIR_NAME))
SAVE_REF_TEST = os.path.join(DATA_HOME,'SRHDA/{}/test.ref'.format(DIR_NAME))
SAVE_HYP_TEST = os.path.join(DATA_HOME,'SRHDA/{}/test.hyp'.format(DIR_NAME))
SAVE_LABEL_TEST = os.path.join(DATA_HOME,'SRHDA/{}/test.label'.format(DIR_NAME))

langs_news = ['cs-en', 'de-en', 'en-cs', 'en-de', 'en-fi', 'en-lv', 'en-ru', 
              'en-tr', 'en-zh', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en']


# systems = {'cs-en':['CUNI-Transformer.5560', 
#                     'online-A.0', 
#                     'online-B.0', 
#                     'online-G.0', 
#                     'uedin.5561'], 
#            'de-en':[], 
#            'et-en':[], 
#            'fi-en':[], 
#            'ru-en':[], 
#            'tr-en':[], 
#            'zh-en':[], 
#            'en-cs':[], 
#            'en-de':[], 
#            'en-et':[], 
#            'en-fi':[], 
#            'en-ru':[], 
#            'en-tr':[], 
#            'en-zh':[]}

import csv
import pickle
import re
import csv
from pprint import pprint
import pandas as pd
import numpy as np
import copy
from  tqdm import tqdm
import random


# In[15]:


def load_file(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data

SRC_files = {lang:load_file(os.path.join(SRC_HOME, 'newstest2017-{0}{1}-src.{0}'.format(lang.split('-')[0], lang.split('-')[1])))  for lang in langs_news}
REF_files = {lang:load_file(os.path.join(REF_HOME, 'newstest2017-{0}{1}-ref.{1}'.format(lang.split('-')[0], lang.split('-')[1]))) for lang in langs_news}
HYP_files = {lang:{} for lang in langs_news}

for lang in langs_news:
    for fname in os.listdir(os.path.join(HYP_HOME, lang)):
        if not fname.startswith('newstest2017'):
            continue
        # extract system id from fname
        system_id = copy.deepcopy(fname).replace('newstest2017.', '').replace('.{}'.format(lang), '')
        # add
        HYP_files[lang][system_id] = load_file(os.path.join(os.path.join(HYP_HOME, lang), fname))

        


# ↓DARR

# In[16]:


# DArr = load_file(DARR_HOME)
# corpus = []
# for idx, da_data in enumerate(DArr):
#     if idx == 0:
#         continue
#     lang = da_data.split(' ')[0]
#     sid = int(da_data.split(' ')[2])
#     better_sys = da_data.split(' ')[3]
#     worse_sys = da_data.split(' ')[4]
#     corpus.append({'lang': lang, 
#                    'sid':sid,
#                    'src': SRC_files[lang][sid-1], 
#                    'ref': REF_files[lang][sid-1], 
#                    'hyp1': HYP_files[lang][better_sys][sid-1], 
#                    'hyp2': HYP_files[lang][worse_sys][sid-1], 
#                    'better':'hyp1'})
# print('saving {}'.format(SAVE_PATH_DARR))
# with open(SAVE_PATH_DARR, mode='wb') as w:
#     pickle.dump(corpus, w)


# DA for train

# In[17]:


filename_good_redup = {lang:'' for lang in langs_news}
DA_data_good_redup = {lang:[] for lang in langs_news}
for lang in langs_news:
    file_path = os.path.join(DA_HOME, 'ad-{}-good-stnd.csv'.format(lang.replace('-', '')))
    if os.path.isfile(file_path):
        filename_good_redup[lang] = file_path
        DA_data_good_redup[lang] = load_file(file_path) 
    else:
#         print('{} does not exist'.format(file_path))
        pass
filename_seg_scores = os.path.join(DA_HOME, 'ad-seg-scores-ensembled.csv')
DA_data_seg_scores = load_file(filename_seg_scores)

def make_corpus_good_stnd_redup(langs, DA_data):
    corpus = []
    type_set = set()
    for lang in langs:
        for idx, row in enumerate(DA_data[lang]):
            if idx == 0:
                continue

            type_id = row.split('\t')[8]
            score = float(row.split('\t')[10])
            sid = int(row.split('\t')[9])
            system_id = row.split('\t')[6]

            type_set.add(type_id)

            if type_id != 'SYSTEM':
                continue
            
            if system_id in HYP_files[lang]:
                corpus.append({'lang':lang,
                               'sid':sid,
                               'year':17,
                               'src':SRC_files[lang][sid-1],
                               'ref':REF_files[lang][sid-1],
                               'hyp':HYP_files[lang][system_id][sid-1],
                               'label':score})               
    return corpus, type_set


def make_corpus_seg_scores(DA_data):
    corpus = []
    sys_dic = {}
    for idx, row in enumerate(DA_data):
        if idx == 0:
            continue
        if re.search('SRC TRG HIT N.raw N.z SID SYS RAW.SCR Z.SCR', row):
#             print(idx)
            continue
        lang = '{}-{}'.format(row.split(' ')[0], row.split(' ')[1]) 
        system_id = row.split(' ')[6]
        sid = int(row.split(' ')[5])
        score = float(row.split(' ')[8])
        n = int(row.split(' ')[3])
        key = (lang, system_id)
        if re.search('\+', system_id):
            system_id = system_id.split('+')[0]
            key = (lang, system_id)
            if key not in sys_dic:
                sys_dic[key] = 1
            else:
                sys_dic[key] += 1
            
        corpus.append({'lang':lang,
                       'sid':sid,
                       'year':17,
                       'src':SRC_files[lang][sid-1],
                       'ref':REF_files[lang][sid-1],
                       'hyp':HYP_files[lang][system_id][sid-1],
                       'label':score})
    return corpus, sys_dic

corpus_good_redup, type_set = make_corpus_good_stnd_redup(langs_news, DA_data_good_redup)
corpus_seg_scores, sys_dic = make_corpus_seg_scores(DA_data_seg_scores)


# In[18]:


print('good redup')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs_news}
for corpus in corpus_good_redup:
    lang = corpus['lang']
    lang_count[lang] += 1
for lang in langs_news:
    print('{} has {} instances'.format(lang, lang_count[lang]))
print()

print('seg scores')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs_news}
for corpus in corpus_seg_scores:
    lang = corpus['lang']
    lang_count[lang] += 1
for lang in langs_news:
    print('{} has {} instances'.format(lang, lang_count[lang]))
print()


# In[19]:


print('saving {}'.format(SAVE_PATH_DA_GOOD_REDUP))
with open(SAVE_PATH_DA_GOOD_REDUP, mode='wb') as w:
    pickle.dump(corpus_good_redup, w)
    
print('saving {}'.format(SAVE_PATH_DA_SEG))
with open(SAVE_PATH_DA_SEG, mode='wb') as w:
    pickle.dump(corpus_seg_scores, w)


# In[20]:


def load_pickle(filename):
    if not os.path.isfile(filename):
        print('{} does not exist'.format(filename))
        exit(-2)
    data = None
    with open(filename, mode='rb') as r:
        data = pickle.load(r)
    return data

# return True when duplicated
def dup_check(train_data, valid_data):
    flag = False
    duplicate_dic = {}
    dup_index = []
    for i, val in enumerate(valid_data):
        key = (val['lang'], val['year'], val['sid'])
        if key not in duplicate_dic:
            duplicate_dic[key] = [i]
        else:
            duplicate_dic[key].append(i)
    for i, trn in enumerate(train_data):
        key = (trn['lang'], trn['year'], trn['sid'])
        if key in duplicate_dic:
            flag = True
            dup_index.append({'train':i, 'valid':duplicate_dic[key]})
    return flag, dup_index
            

def split_data(Alldata, ratio, exception_index, duplication=False):
    
    all_index = [i for i in range(len(Alldata))]
    valid_index = random.sample(list(set(all_index)-set(exception_index)), int((len(Alldata)-len(exception_index))*ratio))
    train_index = list(set(all_index)-set(valid_index))

    train_data = []
    valid_data = []
    for idx in all_index:
        if idx in train_index:
            train_data.append(copy.deepcopy(Alldata[idx]))
        else:
            valid_data.append(copy.deepcopy(Alldata[idx]))
    
    return train_data, valid_data

def get_dup_index(Alldata):
    exception_index = []
    dup_set = {}
    for idx, data in enumerate(Alldata):
        key = (data['lang'], data['year'], data['sid'])
        if key not in dup_set:
            dup_set[key] = [idx]
        else:
            exception_index.extend(dup_set[key])
            exception_index.append(idx)
    exception_index = sorted(list(set(exception_index)))
    return exception_index, dup_set


# In[24]:


valid_ratio = 0.1

SAVE_HOME = os.path.join(DATA_HOME, 'WMT')

Alldata15_16 = []
Alldata15_16.extend(load_pickle(os.path.join(SAVE_HOME, 'wmt15_da.pkl')))
Alldata15_16.extend(load_pickle(os.path.join(SAVE_HOME, 'wmt16_da.pkl')))
Alldata_langs = {}
for data in Alldata15_16:
    lang = data['lang']
    if lang not in Alldata_langs:
        Alldata_langs[lang] = []
    Alldata_langs[lang].append(data)

train_data_langs = {}
valid_data_langs = {}
for lang in Alldata_langs.keys():
#     print('splitting {} data'.format(lang))
    exception_index, dup_set = get_dup_index(Alldata_langs[lang])
    train_data, valid_data = split_data(Alldata_langs[lang], valid_ratio, exception_index, duplication=False)
    train_data_langs[lang] = train_data
    valid_data_langs[lang] = valid_data

# Da = load_pickle(os.path.join(SAVE_HOME, 'wmt17_da_seg.pkl'))
Da = corpus_seg_scores
test_data_langs = {}
for data in Da:
    lang = data['lang']
    if lang in ['en-ru', 'en-zh']:
        if lang not in train_data_langs:
            train_data_langs[lang] = []
        train_data_langs[lang].append(data)
        continue
    if lang not in test_data_langs:
        test_data_langs[lang] = []
    
    test_data_langs[lang].append(data)


# In[26]:


src_train = []
ref_train = []
hyp_train = []
label_train = []

src_valid = []
ref_valid = []
hyp_valid = []
label_valid = []

src_test = []
ref_test = []
hyp_test = []
label_test = []

for lang in Alldata_langs.keys():
    for tdata in train_data_langs[lang]:
        src_train.append('{}\t{}'.format(tdata['src'], lang))
        ref_train.append('{}\t{}'.format(tdata['ref'], lang))
        hyp_train.append('{}\t{}'.format(tdata['hyp'], lang))
        label_train.append('{}\t{}'.format(tdata['label'], lang))
    for vdata in valid_data_langs[lang]:
        src_valid.append('{}\t{}'.format(vdata['src'], lang))
        ref_valid.append('{}\t{}'.format(vdata['ref'], lang))
        hyp_valid.append('{}\t{}'.format(vdata['hyp'], lang))
        label_valid.append('{}\t{}'.format(vdata['label'], lang))   
for lang in test_data_langs.keys():    
    for tsdata in test_data_langs[lang]:
        src_test.append('{}\t{}'.format(tsdata['src'], lang))
        ref_test.append('{}\t{}'.format(tsdata['ref'], lang))
        hyp_test.append('{}\t{}'.format(tsdata['hyp'], lang))
        label_test.append('{}\t{}'.format(tsdata['label'], lang))  


# In[27]:


def writeout(filename, obj):
    with open(filename, mode='w', encoding='utf-8') as w:
        for d in obj:
            w.write(d+os.linesep)


# In[30]:


writeout(SAVE_SRC_TRAIN, src_train)
writeout(SAVE_REF_TRAIN, ref_train)
writeout(SAVE_HYP_TRAIN, hyp_train)
writeout(SAVE_LABEL_TRAIN, label_train)

writeout(SAVE_SRC_VALID, src_valid)
writeout(SAVE_REF_VALID, ref_valid)
writeout(SAVE_HYP_VALID, hyp_valid)
writeout(SAVE_LABEL_VALID, label_valid)

writeout(SAVE_SRC_TEST, src_test)
writeout(SAVE_REF_TEST, ref_test)
writeout(SAVE_HYP_TEST, hyp_test)
writeout(SAVE_LABEL_TEST, label_test)


# In[ ]:




