#!/usr/bin/env python
# coding: utf-8

# In[205]:


import os
import sys

#DATA_HOME = '/ahc/work3/kosuke-t/data/'
DATA_HOME = sys.argv[-1]

DA_HOME = os.path.join(DATA_HOME, 'WMT/newstest2018-humaneval/analysis')
DARR_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/manual-evaluation/RR-seglevel.csv')
SRC_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/source-system-outputs/wmt18-submitted-data/txt/sources')
REF_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/source-system-outputs/wmt18-submitted-data/txt/references')
HYP_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/source-system-outputs/wmt18-submitted-data/txt/system-outputs/newstest2018')
SAVE_PATH_DARR = os.path.join(DATA_HOME, 'WMT/wmt18_darr.pkl')
SAVE_PATH_DA_GOOD_REDUP = os.path.join(DATA_HOME, 'WMT/wmt18_da_good_redup.pkl')
SAVE_PATH_DA_SEG = os.path.join(DATA_HOME, 'WMT/wmt18_da_seg.pkl')

SAVE_SRC_TRAIN = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/train.src')
SAVE_REF_TRAIN = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/train.ref')
SAVE_HYP_TRAIN = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/train.hyp')
SAVE_LABEL_TRAIN = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/train.label')
SAVE_SRC_VALID = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/valid.src')
SAVE_REF_VALID = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/valid.ref')
SAVE_HYP_VALID = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/valid.hyp')
SAVE_LABEL_VALID = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/valid.label')
SAVE_SRC_TEST = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/test.src')
SAVE_REF_TEST = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/test.ref')
SAVE_HYP_TEST = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/test.hyp')
SAVE_LABEL_TEST = os.path.join(DATA_HOME,'SRHDA/WMT15_18_DA/test.label')

langs = ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 
         'en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']

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


# In[206]:


def load_file(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data

SRC_files = {lang:load_file(os.path.join(SRC_HOME, 'newstest2018-{0}{1}-src.{0}'.format(lang.split('-')[0], lang.split('-')[1])))  for lang in langs}
REF_files = {lang:load_file(os.path.join(REF_HOME, 'newstest2018-{0}{1}-ref.{1}'.format(lang.split('-')[0], lang.split('-')[1]))) for lang in langs}
HYP_files = {lang:{} for lang in langs}

for lang in langs:
    for fname in os.listdir(os.path.join(HYP_HOME, lang)):
        if not fname.startswith('newstest2018'):
            continue
        # extract system id from fname
        system_id = copy.deepcopy(fname).replace('newstest2018.', '').replace('.{}'.format(lang), '')
        # add
        HYP_files[lang][system_id] = load_file(os.path.join(os.path.join(HYP_HOME, lang), fname))

        


# ↓DARR

# In[207]:


DArr = load_file(DARR_HOME)
corpus = []
for idx, da_data in enumerate(DArr):
    if idx == 0:
        continue
    lang = da_data.split(' ')[0]
    sid = int(da_data.split(' ')[2])
    better_sys = da_data.split(' ')[3]
    worse_sys = da_data.split(' ')[4]
    corpus.append({'lang': lang, 
                   'sid':sid,
                   'year':18,
                   'src': SRC_files[lang][sid-1], 
                   'ref': REF_files[lang][sid-1], 
                   'hyp1': HYP_files[lang][better_sys][sid-1], 
                   'hyp2': HYP_files[lang][worse_sys][sid-1], 
                   'better':'hyp1'})
print('saving {}'.format(SAVE_PATH_DARR))
with open(SAVE_PATH_DARR, mode='wb') as w:
    pickle.dump(corpus, w)


# DA for train

# In[208]:


filename_good_redup = {lang: os.path.join(DA_HOME, 'ad-{}-good-stnd-redup.csv'.format(lang.replace('-', ''))) for lang in langs}
#filename_good_redup = {lang: os.path.join(DA_HOME, 'ad-{}-good-stnd.csv'.format(lang.replace('-', ''))) for lang in langs}
filename_seg_scores = {lang: os.path.join(DA_HOME, 'ad-seg-scores-{}.csv'.format(lang)) for lang in langs}

DA_data_good_redup = {lang: load_file(f) for lang, f in filename_good_redup.items()}
DA_data_seg_scores = {lang: load_file(f) for lang, f in filename_seg_scores.items()}

def make_corpus_good_stnd_redup(langs, DA_data):
    corpus = []
    type_set = set()
    for lang in langs:
        for idx, row in enumerate(DA_data[lang]):
            if idx == 0:
                continue

            type_id = row.split('\t')[8]
            score = float(row.split('\t')[-2])
            sid = int(row.split('\t')[9])
            system_id = row.split('\t')[6]

            type_set.add(type_id)

            if type_id != 'SYSTEM':
                continue
            if '+' in system_id:
                continue
            corpus.append({'lang':lang,
                           'sid':sid,
                           'year':18,
                           'src':SRC_files[lang][sid-1],
                           'ref':REF_files[lang][sid-1],
                           'hyp':HYP_files[lang][system_id][sid-1],
                           'label':score})
    return corpus


def make_corpus_seg_scores(langs, DA_data):
    corpus = []
    for lang in langs:
        for idx, row in enumerate(DA_data[lang]):
            if idx == 0:
                continue
            system_id = row.split(' ')[0]
            sid = int(row.split(' ')[1])
            score = float(row.split(' ')[3])
            n = int(row.split(' ')[4])
            if system_id == 'HUMAN':
                continue
            
            corpus.append({'lang':lang,
                           'sid':sid,
                           'year':18,
                           'n':n,
                           'src':SRC_files[lang][sid-1],
                           'ref':REF_files[lang][sid-1],
                           'hyp':HYP_files[lang][system_id][sid-1],
                           'label':score})
    return corpus

corpus_good_redup = make_corpus_good_stnd_redup(langs, DA_data_good_redup)
corpus_seg_scores = make_corpus_seg_scores(langs, DA_data_seg_scores)


# In[209]:


times_langs = {}
for corpus in corpus_seg_scores:
    lang = corpus['lang']
    n = corpus['n']
    if lang not in times_langs:
        times_langs[lang] = {}
        times_langs[lang][n] = 1
    else:
        if n not in times_langs[lang]:
            times_langs[lang][n] = 1
        else:
            times_langs[lang][n] += 1


# In[210]:


times_langs


# In[211]:


# def remove_outlier(corpus):
#     human_scores = {}
#     mean = {}
#     MAP = {}
#     z = {}
#     for c in corpus:
#         lang = c['lang']
#         if lang not in human_scores:
#             human_scores[lang] = [c['label']]
#         else:
#             human_scores[lang].append(c['label'])
    
#     for lang in human_scores.keys():
#         mean[lang] = np.mean(human_scores[lang]) 
#         MAP[lang] = 1.483 * np.mean([s - mean[lang] for s in human_scores[lang]])
#         z[lang] = ([(s - mean[lang])/MAP[lang] for s in human_scores[lang]])
        
#     return human_scores, mean, MAP, z

# human_scores, mean, MAP, z = remove_outlier(corpus_seg_scores)


# In[212]:


print('good redup')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs}
for corpus in corpus_good_redup:
    lang = corpus['lang']
    lang_count[lang] += 1
total = 0
toen_total = 0
for lang in langs:
    total += lang_count[lang]
    if lang.endswith('en'):
        toen_total += lang_count[lang]
    print('{} has {} instances'.format(lang, lang_count[lang]))
print('{} : {}'.format('Total', total))
print('{} : {}'.format('To En Total', toen_total))
print()

print('seg scores')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs}
for corpus in corpus_seg_scores:
    lang = corpus['lang']
    lang_count[lang] += 1
total = 0
toen_total = 0
for lang in langs:
    total += lang_count[lang]
    if lang.endswith('en'):
        toen_total += lang_count[lang]
    print('{} has {} instances'.format(lang, lang_count[lang]))
print('{} : {}'.format('Total', total))
print('{} : {}'.format('To En Total', toen_total))
print()

print('seg scores over n==2')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs}
for corpus in corpus_seg_scores:
    lang = corpus['lang']
    if corpus['n'] == 1:
        continue
    lang_count[lang] += 1
total = 0
toen_total = 0
for lang in langs:
    total += lang_count[lang]
    if lang.endswith('en'):
        toen_total += lang_count[lang]
    print('{} has {} instances'.format(lang, lang_count[lang]))
print('{} : {}'.format('Total', total))
print('{} : {}'.format('To En Total', toen_total))
print()


# In[213]:


print('saving {}'.format(SAVE_PATH_DA_GOOD_REDUP))
with open(SAVE_PATH_DA_GOOD_REDUP, mode='wb') as w:
    pickle.dump(corpus_good_redup, w)
    
print('saving {}'.format(SAVE_PATH_DA_SEG))
with open(SAVE_PATH_DA_SEG, mode='wb') as w:
    pickle.dump(corpus_seg_scores, w)


# In[163]:


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
    
#     if duplication:
    return train_data, valid_data
    
#     count = 1
#     duplication_flag, dup_index = dup_check(train_data, valid_data)
#     if not duplication_flag:
#         while(duplication_flag):
#             count += 1
#             all_index = [i for i in range(len(Alldata))]
#             valid_index = random.sample(all_index, int(len(Alldata)*ratio))
#             train_index = list(set(all_index)-set(valid_index))

#             train_data = []
#             valid_data = []
#             for idx in all_index:
#                 if idx in train_index:
#                     train_data.append(copy.deepcopy(Alldata[idx]))
#                 else:
#                     valid_data.append(copy.deepcopy(Alldata[idx]))
#             duplication_flag, dup_index = dup_check(train_data, valid_data)
        
# #         if count % 100 == 0:
# #             print('trial No.{}'.format(count))
    
# #     print('Total trial : {}'.format(count))
    
#     return train_data, valid_data

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


# Split data into train, valid, test
# Train:Valid = 9:1 in wmt18

# In[166]:


valid_ratio = 0.1

SAVE_HOME = os.path.join(DATA_HOME, 'WMT')

Alldata15_17 = []
Alldata15_17.extend(load_pickle(os.path.join(SAVE_HOME, 'wmt15_da.pkl')))
Alldata15_17.extend(load_pickle(os.path.join(SAVE_HOME, 'wmt16_da.pkl')))
Alldata15_17.extend(load_pickle(os.path.join(SAVE_HOME, 'wmt17_da_seg.pkl')))
Alldata_langs = {}
for data in Alldata15_17:
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

Darr = load_pickle(SAVE_PATH_DARR)


# In[167]:


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

# 'lang': lang, 
# 'sid':sid,
# 'year':18,
# 'src': SRC_files[lang][sid-1], 
# 'ref': REF_files[lang][sid-1], 
# 'hyp1': HYP_files[lang][better_sys][sid-1], 
# 'hyp2': HYP_files[lang][worse_sys][sid-1], 
# 'better':'hyp1'
sid = 0
for idx, test_data in enumerate(Darr):
    sid += 1
    src_test.append('{}\t{}'.format(test_data['src'], test_data['lang']))
    ref_test.append('{}\t{}'.format(test_data['ref'], test_data['lang']))
    hyp_test.append('{}\t{}'.format(test_data['hyp1'], test_data['lang']))
    sid += 1
    src_test.append('{}\t{}'.format(test_data['src'], test_data['lang']))
    ref_test.append('{}\t{}'.format(test_data['ref'], test_data['lang']))
    hyp_test.append('{}\t{}'.format(test_data['hyp2'], test_data['lang']))
    label_test.append('{}>{}\t{}'.format(sid-1, sid, test_data['lang']))


# In[170]:


def writeout(filename, obj):
    with open(filename, mode='w', encoding='utf-8') as w:
        for d in obj:
            w.write(d+os.linesep)


# In[173]:


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


# In[175]:





# In[ ]:




