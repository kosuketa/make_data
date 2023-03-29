#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import sys

#DATA_HOME = '/ahc/work3/kosuke-t/data/'
DATA_HOME = sys.argv[-1]

SAVE_PATH_DA = os.path.join(DATA_HOME, 'WMT/wmt16_da.pkl')
_DATA_HOME = os.path.join(DATA_HOME, 'WMT/DAseg-wmt-newstest2016')


langs = ['cs-en', 'de-en', 'en-ru', 'fi-en', 'ro-en', 'ru-en', 'tr-en']

import csv
import pickle
import re
import csv
from pprint import pprint
import pandas as pd
import numpy as np
import copy
from  tqdm import tqdm


# In[7]:


def load_file(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data

SRC_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2016.source.{}'.format(lang)))  for lang in langs}
REF_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2016.reference.{}'.format(lang))) for lang in langs}
HYP_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2016.mt-system.{}'.format(lang))) for lang in langs}
DA_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2016.human.{}'.format(lang))) for lang in langs}


# In[8]:


corpus = []
for lang in langs:
    for i, (src, ref, hyp, label) in enumerate(zip(SRC_files[lang], REF_files[lang], HYP_files[lang], DA_files[lang])):
        corpus.append({'lang':lang,
                       'year':16,
                       'sid':int(i)+1,
                       'src':src.rstrip(),
                       'ref':ref.rstrip(),
                       'hyp':hyp.rstrip(),
                       'label':label.rstrip()})
        
        


# In[9]:


print('saving {}'.format(SAVE_PATH_DA))
with open(SAVE_PATH_DA, mode='wb') as w:
    pickle.dump(corpus, w)


# In[ ]:





# In[ ]:




