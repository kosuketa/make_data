#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pprint import pprint
import re
import pickle

DATAHOME = '/home/is/kosuke-t/project_disc/data/SRHDA/15-18/'

langs = {'15':['cs-en', 'de-en',          'fi-en',                   'ru-en'],  # 4
         '16':['cs-en', 'de-en',          'fi-en',          'ro-en', 'ru-en', 'tr-en'], # 6
         '17':['cs-en', 'de-en',          'fi-en', 'lv-en',          'ru-en', 'tr-en', 'zh-en'], # 7
         '18':['cs-en', 'de-en', 'et-en', 'fi-en',                   'ru-en', 'tr-en', 'zh-en'],  # 7
         '15-17': ['cs-en', 'de-en', 'fi-en', 'ru-en'],
         '15-18': ['cs-en', 'de-en', 'fi-en', 'ru-en']
        }


# In[2]:


def unienc(text):
    if not text:
        return ''
    return (text.encode('utf-8','ignore')).decode()


# In[3]:


def load17(SRCF, REFF, HYPF, DAF, lang=None):
    srcList = []
    refList = []
    hypList = []
    DAList = []
    SIDList = []
    
    DA_sid = []
    DA_sys = []
    DA_score = [] 
    srcdata = []
    refdata = []
    hypdata = {} #key:system, value:list of sentences
    
    #load SRC
    with open(SRCF, mode='r', encoding='utf-8') as r:
        srcdata = list(map(unienc, r.read().split('\n')))
    if srcdata[-1] == '':
        srcdata.pop(-1)
    
    #load REF
    with open(REFF, mode='r', encoding='utf-8') as r:
        refdata = list(map(unienc, r.read().split('\n')))
    if refdata[-1] == '':
        refdata.pop(-1)
        
    #load HYP
    for system in HYPF:
        with open(HYPF[system], mode='r', encoding='utf-8') as r:
            hypdata[system] = list(map(unienc, r.read().split('\n')))
            if hypdata[system][-1] == '':
                hypdata[system].pop(-1)
        if re.search('CASICT', system) and lang=='zh-en':
            hypdata['CASICT-cons.5144'] = hypdata[system]
        elif re.search('ROCMT', system) and lang=='zh-en':
            hypdata['ROCMT.5167'] = hypdata[system]
    #load DA file
    f = open(DAF, mode='r')
    
    # 'SRC', 'TRG', 'HIT', 'N.raw', 'N.z', 'SID', 'SYS', 'RAW.SCR', 'Z.SCR'
    flag = 0
    for line in f:
        if flag == 0:
            flag = 1
        else:
            if line.split()[0] == lang.split('-')[0] and line.split()[1] == lang.split('-')[-1]:
                DA_sid.append(int(line.split()[5]))
                DA_sys.append(line.split()[6])
                if '+' in DA_sys[-1]:
                    DA_sys[-1] = DA_sys[-1].split('+')[0]
                # if '+' in sys:
                #     sys = sys.split('+')
                #     print(sysout_list_dict[sys[0]][sent_ID - 1])
                #     print(sysout_list_dict[sys[1]][sent_ID - 1])
                DA_score.append(line.split()[-1])
    f.close()
    
    for sid, sys, score in zip(DA_sid, DA_sys, DA_score):
        if sys in hypdata:
            hypList.append(hypdata[sys][sid-1])
            srcList.append(srcdata[sid-1])
            refList.append(refdata[sid-1])
            DAList.append(float(score))
            SIDList.append(sid)
#             elif DAdic['sys_id'] == 'REFERENCE':
#                 hypList.append(refdata[idx])
#                 srcList.append(srcdata[idx])
#                 refList.append(refdata[idx])
#                 DAList.append(mean(DAdic[str(idx)][sys]))
        else:
            #print('\"{}\" not found in DA file'.format(sys))
            continue
        
    return [srcList, refList, hypList, DAList, SIDList]

def make17():
    srcF = {}
    refF = {}
    hypF = {}
    DAF = {}
    data = {}
    
    SYSTEMS17 = {'cs-en':['online-A.0', 'online-B.0', 'PJATK.4760', 'uedin-nmt.4955'], 
                 'de-en':['C-3MA.4958', 'online-A.0', 'online-G.0', 'TALP-UPC.4830', 'KIT.4951', 'online-B.0', 'RWTH-nmt-ensemble.4920', 'uedin-nmt.4723', 'LIUM-NMT.4733', 'online-F.0', 'SYSTRAN.4846'], 
                 'fi-en':['apertium-unconstrained.4793', 'online-A.0', 'online-G.0', 'Hunter-MT.4925', 'online-B.0', 'TALP-UPC.4937'], 
                 'lv-en':['C-3MA.5067', 'online-A.0', 'tilde-c-nmt-smt-hybrid.5051', 'Hunter-MT.5092', 'online-B.0', 'tilde-nc-nmt-smt-hybrid.5050', 'jhu-pbmt.4980', 'PJATK.4740', 'uedin-nmt.5017'], 
                 'ru-en':['afrl-mitll-opennmt.4896', 'jhu-pbmt.4978', 'online-A.0', 'online-F.0', 'uedin-nmt.4890', 'afrl-mitll-syscomb.4905', 'NRC.4855', 'online-B.0', 'online-G.0'], 
                 'tr-en':['afrl-mitll-m2w-nr1.4901', 'JAIST.4859', 'LIUM-NMT.4888', 'online-B.0', 'PROMT-SMT.4737', 'afrl-mitll-syscomb.4902', 'jhu-pbmt.4972', 'online-A.0', 'online-G.0', 'uedin-nmt.4931'], 
                 'zh-en':['afrl-mitll-opennmt.5109', 'NRC.5172', 'online-G.0', 'SogouKnowing-nmt.5171', 'CASICT-DCU-NMT.5144', 'online-A.0', 'Oregon-State-University-S.5173', 'uedin-nmt.5112', 'jhu-nmt.5151', 'online-B.0', 'PROMT-SMT.5125', 'UU-HNMT.5162', 'NMT-Model-Average-Multi-Cards.5099', 'online-F.0', 'ROCMT.5183', 'xmunmt.5160']
                }
    DATAHOME = '/project/nakamura-lab08/Work/kosuke-t/data'
    for lang in langs['17']:
        lang2 = re.sub('-', '', lang)
        langSRC = lang[:2]
        langREF = lang[-2:]
        srcF[lang] = os.path.join(DATAHOME, 'wmt17-submitted-data/txt/sources/newstest2017-{}-src.{}'.format(lang2, langSRC))
        refF[lang] = os.path.join(DATAHOME, 'wmt17-submitted-data/txt/references/newstest2017-{}-ref.{}'.format(lang2, langREF))
        hypF[lang] = {}
        for system in SYSTEMS17[lang]:
            hypF[lang][system] = os.path.join(DATAHOME, 'wmt17-submitted-data/txt/system-outputs/newstest2017/{0}/newstest2017.{1}.{0}'.format(lang, system)) 
        DAF[lang] =  os.path.join(DATAHOME, 'DAseg-wmt-newstest2017/anon-proc-hits-seg-{}/analysis/ad-seg-scores.csv'.format(langREF))
        
        src, ref, hyp, da, sid = load17(srcF[lang], refF[lang], hypF[lang], DAF[lang], lang=lang)
        data[lang] = {}
        data[lang]['SRC'] = src
        data[lang]['REF'] = ref
        data[lang]['HYP'] = hyp
        data[lang]['DA'] = da
        data[lang]['SID'] = sid
        
# #         writeout('17', lang, data[lang])
#         print('wmt17 {} : {} sentences'.format(lang, str(len(data[lang]['DA']))))
        
    return data


# In[5]:


with open(os.path.join(DATAHOME, 'data15.pkl'), mode='rb') as r:
    data15 = pickle.load(r)
with open(os.path.join(DATAHOME, 'data16.pkl'), mode='rb') as r:
    data16 = pickle.load(r)
# with open(os.path.join(DATAHOME, 'data17.pkl'), mode='rb') as r:
#     data17 = pickle.load(r)
data17 = make17()
with open(os.path.join(DATAHOME, 'data17.pkl'), mode='wb') as w:
    pickle.dump(data17, w)


# In[8]:


D = data17
for lang in D.keys():
    if not (len(D[lang]['REF']) == len(D[lang]['SRC']) == len(D[lang]['HYP']) == len(D[lang]['DA'])):
        print('{} number of sentences does not match'.format(lang))
    else:
        print('{} : {}'.format(lang, len(D[lang]['REF'])))


# In[73]:


# data = []
# REFs = []
# DUPs = []
# for lang in data15.keys():
#     for i in range(len(data15[lang]['REF'])):
#         refsent = data15[lang]['REF'][i]
#         if refsent not in REFs:
#             REFs.append(refsent)
#             data.append({'reference':refsent, 
#                          'source':data15[lang]['SRC'][i],
#                          'language':lang, 
#                          'year':15, 
#                          'sent_num':i+1}
#                         )
#         else:
#             DUPs.append({'reference':refsent, 'lang':lang, 'year':15, 'sid':i+1})
# for lang in data16.keys():
#     for i in range(len(data16[lang]['REF'])):
#         refsent = data16[lang]['REF'][i]
#         if refsent not in REFs:
#             REFs.append(refsent)
#             data.append({'reference':refsent, 
#                          'source':data16[lang]['SRC'][i],
#                          'language':lang, 
#                          'year':16, 
#                          'sent_num':i+1}
#                         )
#         else:
#             DUPs.append({'reference':refsent, 'lang':lang, 'year':16, 'sid':i+1})
# for lang in data17.keys():
#     for i in range(len(data17[lang]['REF'])):
#         refsent = data17[lang]['REF'][i]
#         if refsent not in REFs:
#             REFs.append(refsent)
#             data.append({'reference':refsent, 
#                          'source':data17[lang]['SRC'][i],
#                          'language':lang, 
#                          'year':17, 
#                          'sent_num':data17[lang]['SID'][i]}
#                         )
#         else:
#             DUPs.append({'reference':refsent, 'lang':lang, 'year':17, 'sid':i+1})


# In[26]:


data = []
REFs = []
DUPs = []
d_tmp = []
for lang in data15.keys():
    for i in range(len(data15[lang]['REF'])):
        refsent = data15[lang]['REF'][i]
#         if refsent not in REFs:
#             REFs.append(refsent)
        d_tmp.append({'language':str(lang), 
                     'year':int(15), 
                     'sent_num':int(i+1),
                     'reference':str(refsent), 
                     'source':str(data15[lang]['SRC'][i]),
                     'hypothesis':str(data15[lang]['HYP'][i]), 
                     'z-DAscore':float(data15[lang]['DA'][i])}
                    )
data = sorted(d_tmp, key=lambda k: k['source'])
#         else:
#             DUPs.append({'reference':refsent, 'lang':lang, 'year':15, 'sid':i+1})
d_tmp = []
for lang in data16.keys():
    for i in range(len(data16[lang]['REF'])):
        refsent = data16[lang]['REF'][i]
#         if refsent not in REFs:
#             REFs.append(refsent)
        d_tmp.append({'language':str(lang), 
                     'year':int(16), 
                     'sent_num':int(i+1),
                     'reference':str(refsent), 
                     'source':str(data16[lang]['SRC'][i]),
                     'hypothesis':str(data16[lang]['HYP'][i]), 
                     'z-DAscore':float(data16[lang]['DA'][i])}
                    )
#         else:
#             DUPs.append({'reference':refsent, 'lang':lang, 'year':16, 'sid':i+1})
d_tmp = sorted(d_tmp, key=lambda k: k['source'])
data.extend(d_tmp)
d_tmp = []
for lang in data17.keys():
    for i in range(len(data17[lang]['REF'])):
        refsent = data17[lang]['REF'][i]
#         if refsent not in REFs:
#             REFs.append(refsent)
        d_tmp.append({'language':str(lang), 
                     'year':int(17), 
                     'sent_num':int(data17[lang]['SID'][i]),
                     'reference':str(refsent), 
                     'source':str(data17[lang]['SRC'][i]),
                     'hypothesis':str(data17[lang]['HYP'][i]), 
                     'z-DAscore':float(data17[lang]['DA'][i])}
                   )
#         else:
#             DUPs.append({'reference':refsent, 'lang':lang, 'year':17, 'sid':i+1})
d_tmp = sorted(d_tmp, key=lambda k: k['source'])
data.extend(d_tmp)


# In[27]:


with open(os.path.join(DATAHOME, 'wmt_data15-17_all_dup.pkl'), mode='wb') as w:
    pickle.dump(data, w)


# In[13]:


import csv
import os


# In[19]:


get_ipython().system('cp $SAVEDIR /ahc/ahcshare/Data/WMT/WMT_Metrics_Task/wmt_15_17_all_data.tsv')


# In[30]:


SAVEDIR = os.path.join('/project/nakamura-lab08/Work/kosuke-t/data/utils', 'wmt_15_17_all_data_dup.tsv')

with open(SAVEDIR, "w", newline="") as f:

    # 「delimiter」に区切り文字、「quotechar」に囲い文字を指定します
    # quotingにはクォーティング方針を指定します（後述）
    writer = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # writerowに行列を指定することで1行分を出力できます
    writer.writerow(["year", "language", "sent_num", "source", "reference", "hypothesis", "z-DAscore"])
    keys = ["year", "language", "sent_num", "source", "reference", "hypothesis", "z-DAscore"]
    for d in data:
        writer.writerow([d[k] for k in keys])


# In[74]:


DUPs[:3]


# In[70]:


REFs.index(DUPs[0]['reference'])


# In[77]:


len(data)


# In[75]:





# In[ ]:




