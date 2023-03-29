#!/usr/bin/env python
# coding: utf-8

# In[5]:




file_dir = '/ahc/work3/kosuke-t/data/WMT/newstest2017-system-level-human/anon-proc-hits-sys-combined/analysis'

import os
import re
import pickle

langs = ['cs-en','de-en','en-cs','en-de','en-fi','en-lv','en-ru',
         'en-tr','en-zh','fi-en','lv-en','ru-en','tr-en','zh-en']


# In[6]:


def load_file(filename):
    if not os.path.isfile(filename):
        print('{} does not exist'.format(filename))
        exit(-2)
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data
        


# In[16]:


da_files = {}
sys_dic = {}
corpus = {lang:[] for lang in langs}
times_dic = {lang:{} for lang in langs}
for lang in langs:
    da_files[lang] = load_file(os.path.join(file_dir, 'ad-seg-scores-{}.csv'.format(lang)))
    for idx, row in enumerate(da_files[lang]):
        if idx == 0:
            continue
        sys_id = row.split(' ')[0]
        sid = int(row.split(' ')[1])
        score = float(row.split(' ')[3])
        n = int(row.split(' ')[4])
        key = (lang, sys_id)
        if key not in sys_dic:
            sys_dic[key] = 1
        else:
            sys_dic[key] += 1
        
        corpus[lang].append({'lang':lang, 'sys_id':sys_id, 'sid':sid, 'label':score})
        
        if n not in times_dic[lang]:
            times_dic[lang][n] = 1
        else:
            times_dic[lang][n] += 1
        


# In[26]:


filepath = '/ahc/work3/kosuke-t/data/WMT/DAseg-wmt-newstest2017/anon-proc-hits-seg-en/analysis/ad-seg-scores.csv'

langs = ['cs-en','de-en','fi-en','lv-en','ru-en','tr-en','zh-en']
SYSTEMS17 = {'cs-en':['online-A.0', 'online-B.0', 'PJATK.4760', 'uedin-nmt.4955'], 
             'de-en':['C-3MA.4958', 'online-A.0', 'online-G.0', 'TALP-UPC.4830', 
                      'KIT.4951', 'online-B.0', 'RWTH-nmt-ensemble.4920', 'uedin-nmt.4723',
                      'LIUM-NMT.4733', 'online-F.0', 'SYSTRAN.4846'], 
             'fi-en':['apertium-unconstrained.4793', 'online-A.0', 'online-G.0',
                      'Hunter-MT.4925', 'online-B.0', 'TALP-UPC.4937'], 
             'lv-en':['C-3MA.5067', 'online-A.0', 'tilde-c-nmt-smt-hybrid.5051',
                      'Hunter-MT.5092', 'online-B.0', 'tilde-nc-nmt-smt-hybrid.5050',
                      'jhu-pbmt.4980', 'PJATK.4740', 'uedin-nmt.5017'], 
             'ru-en':['afrl-mitll-opennmt.4896', 'jhu-pbmt.4978', 'online-A.0', 
                      'online-F.0', 'uedin-nmt.4890', 'afrl-mitll-syscomb.4905',
                      'NRC.4855', 'online-B.0', 'online-G.0'], 
             'tr-en':['afrl-mitll-m2w-nr1.4901', 'JAIST.4859', 'LIUM-NMT.4888',
                      'online-B.0', 'PROMT-SMT.4737', 'afrl-mitll-syscomb.4902',
                      'jhu-pbmt.4972', 'online-A.0', 'online-G.0', 'uedin-nmt.4931'], 
             'zh-en':['afrl-mitll-opennmt.5109', 'NRC.5172', 'online-G.0',
                      'SogouKnowing-nmt.5171', 'CASICT-DCU-NMT.5144', 'online-A.0',
                      'Oregon-State-University-S.5173', 'uedin-nmt.5112', 'jhu-nmt.5151', 
                      'online-B.0', 'PROMT-SMT.5125', 'UU-HNMT.5162',
                      'NMT-Model-Average-Multi-Cards.5099', 'online-F.0', 'ROCMT.5183', 'xmunmt.5160']
            }
data = load_file(filepath)
sys_dic = {}
corpus = {lang:[] for lang in langs}
times_dic = {lang:{} for lang in langs}
for idx, row in enumerate(data):
    if idx == 0:
        continue
    lang = '{}-{}'.format(row.split(' ')[0], row.split(' ')[1])
    hit = int(row.split(' ')[2])
    n = int(row.split(' ')[4])
    sid = int(row.split(' ')[5])
    sys_id = row.split(' ')[6]
    score = row.split(' ')[8]
    key = (lang, sys_id)
    
    if key not in sys_dic:
        sys_dic[key] = 1
    else:
        sys_dic[key] += 1
        
    if sys_id not in SYSTEMS17[lang]:
        continue

    corpus[lang].append({'lang':lang, 'sys_id':sys_id, 'sid':sid, 'label':score})

    if n not in times_dic[lang]:
        times_dic[lang][n] = 1
    else:
        times_dic[lang][n] += 1


# In[28]:


for lang in langs:
    print('{}:{}'.format(lang, len(corpus[lang])))


# In[ ]:




