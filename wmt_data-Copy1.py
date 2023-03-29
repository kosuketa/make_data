#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle

filename = '/tmp/wmt_data15-17.pkl'


# In[3]:


f = open(filename, mode='rb')
data = pickle.load(f)
f.close()


# In[ ]:




