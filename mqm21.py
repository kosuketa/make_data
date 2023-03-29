#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import re
import numpy as np
import json
import pickle
import copy
import pandas as pd


# In[5]:


premade_file = '/home/ksudoh/kosuke-t/data_link/WMT/wmt21-newstest2021.json'
r = open(premade_file, "r")
df = pd.read_json(r, lines=True)
r.close()


# In[10]:


lang_dic = {}
for lang in list(df['lang']):
    if lang not in lang_dic:
        lang_dic[lang] = 1
    else:
        lang_dic[lang] += 1
print(lang_dic)


# In[ ]:




