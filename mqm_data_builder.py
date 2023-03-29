#!/usr/bin/env python
# coding: utf-8

# In[3]:


import abc
import glob
import itertools
import json
import os
import re
import shutil
import tarfile
import urllib
import urllib.request
import subprocess
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score
import numpy as np
import tempfile
import sys
from logging import getLogger
import pandas as pd
import six
import copy
import random
import math
# import tensorflow.compat.v1 as tf
from distutils.dir_util import copy_tree
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False
from transformers import AutoTokenizer
from polyleven import levenshtein
from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import difflib
from pprint import pprint


# In[4]:


"""
MQMデータ取り扱い時の注意点
・srcファイルとMQMファイル内のsrc間でのpunctuationの違い
・同じsystemの出力で同じhypなはずなのに、 MQMファイル内でraterごと、またはrater内でも異なる
・
"""
pass


# In[6]:


DATA_HOME = '/ahc/work3/kosuke-t/WMT'

MQM_1hot_only_severity_to_vec = {"no-error":np.asarray([1, 0, 0]), 
                                 "No-error":np.asarray([1, 0, 0]),
                                 "Neutral":np.asarray([0, 1, 0]),
                                 "Minor":np.asarray([0, 1, 0]),
                                 "Major":np.asarray([0, 0, 1])}
MQM_tag_list = ["No-error", 
                "Neutral",
                "Minor",
                "Major"]


class MQM_importer20():
    def __init__(self, save_dir,
                 emb_label=False, emb_only_sev=True, 
                 no_error_score=np.asarray([1, 0, 0]),
                 tokenizer_name='xlm-roberta-large',
                 google_score=True, split_dev=False,
                 dev_ratio=0.1, remove_strange=False,
                 agreement='low'):
        self.save_dir = save_dir
        self.save_tmp = save_dir + '_tmp'
        self.year = '20'
        self.emb_label = emb_label
        self.emb_only_sev = emb_only_sev
        self.no_error_score = no_error_score
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.google_score = google_score
        self.split_dev = split_dev
        self.dev_ratio = dev_ratio
        self.remove_strange = remove_strange
        self.agreement = agreement

        self.src_files = os.path.join(DATA_HOME,
                                      'WMT20_data/txt/sources/newstest2020-{}-src.{}.txt'.format('LANGWO_LETTERS', 'LANG1_LETTERS'))
        self.ref_files = os.path.join(DATA_HOME,
                                      'WMT20_data/txt/references/newstest2020-{}-ref.{}.txt'.format('LANGWO_LETTERS', 'LANG2_LETTERS'))
        self.hyp_files = os.path.join(DATA_HOME,
                                      'WMT20_data/txt/system-outputs/{}/newstest2020.{}.{}.txt'.format('LANG', 'LANG', 'SYSTEM'))
        self.MQM_avg_files = os.path.join(DATA_HOME,
                                          'wmt-mqm-human-evaluation/newstest2020/{}/mqm_newstest2020_{}.avg_seg_scores.tsv'.format('LANGWO_LETTERS', 'LANGWO_LETTERS'))
        self.MQM_tag_files = os.path.join(DATA_HOME,
                                          'wmt-mqm-human-evaluation/newstest2020/{}/mqm_newstest2020_{}.tsv'.format('LANGWO_LETTERS', 'LANGWO_LETTERS'))
        self.langs = ['en-de', 'zh-en']
        self.systems = {'en-de': ['eTranslation.737',
                                  'Human-A.0',
                                  'Human-B.0',
                                  'Human-P.0',
                                  'Huoshan_Translate.832',
                                  'Online-A.1574',
                                  'Online-B.1590',
                                  'OPPO.1535',
                                  'Tencent_Translation.1520',
                                  'Tohoku-AIP-NTT.890'],
                        'zh-en': ['DeepMind.381', 
                                  'DiDi_NLP.401',
                                  'Human-A.0',
                                  'Human-B.0',
                                  'Huoshan_Translate.919',
                                  'Online-B.1605',
                                  'OPPO.1422',
                                  'Tencent_Translation.1249',
                                  'THUNLP.1498',
                                  'WeChat_AI.1525']}
        
    def get_MQM_fname(self, fname, lang):
        lang1 = lang.split('-')[0]
        lang2 = lang.split('-')[1]
        langwo = lang.replace('-', '')
        return fname.replace('LANGWO_LETTERS', langwo, 2)
    
    def load_file(self, fname):
        with open(fname, mode='r', encoding='utf-8') as r:
            return r.readlines()
        
    def get_langs(self):
        return self.langs
    
    def get_srcs(self, lang):
        langwo = lang.replace('-', '')
        lang1 = lang[:2]
        fname = self.src_files.replace('LANGWO_LETTERS', langwo).replace('LANG1_LETTERS', lang1)
        src_data = self.load_file(fname)
        src_data = [s.rstrip() for s in src_data]
        return src_data
    
    def get_refs(self, lang):
        langwo = lang.replace('-', '')
        lang2 = lang[-2:]
        fname = self.ref_files.replace('LANGWO_LETTERS', langwo).replace('LANG2_LETTERS', lang2)
        ref_data = self.load_file(fname)
        ref_data = [r.rstrip() for r in ref_data]
        return ref_data
    
    def get_hyps(self, lang):
        hyp_data = {}
        for sys in self.systems[lang]:
            fname = self.hyp_files.replace('LANG', lang, 2).replace('SYSTEM', sys)
            hyp_data[sys] = self.load_file(fname)
            hyp_data[sys] = [h.rstrip() for h in hyp_data[sys]]
        return hyp_data
        
    def get_score(self, category, severity):
        if self.google_score:
            scores = [0.0, 0.1, 1.0, 5.0, 25.0]
        else:
            scores = [0.0, 1.0, 2.0, 3.0, 4.0]
        if severity == 'Major':
            if category == 'Non-translation!' or category == 'Non-translation':
                return scores[4]
            else:
                return scores[3]
        elif severity == 'Minor':
            if category == 'Fluency/Punctuation':
                return scores[1]
            else:
                return scores[2]
        elif severity in ['no-error', 'No-error', 'Neutral']:
            return scores[0]
        else:
            print(severity, category)
            raise NotImplementedError
    
    def get_untagged_hyp(self, tagged_hyp):
        return tagged_hyp.replace('<v>', '').replace('</v>', '')
    
    def get_untagged_token(self, tagged_hyp):
        untagged_hyp = self.get_untagged_hyp(tagged_hyp)
        token_untagged_hyp = self.tokenizer.encode(untagged_hyp)
        return token_untagged_hyp
    
    def get_src_ref_token(self, sent):
        return self.tokenizer.encode(sent, truncation=True)
    
    def get_pos_start_tag(self, tagged_hyp):
        token_untagged_hyp = self.get_untagged_token(tagged_hyp) 
        token_tagged_hyp = self.tokenizer.encode(tagged_hyp)
        for idx, t_tagged in enumerate(token_tagged_hyp):
            if t_tagged == token_untagged_hyp[idx]:
                continue
            else:
                return idx
        
    def get_pos_end_tag(self, tagged_hyp):
        token_untagged_hyp = self.get_untagged_token(tagged_hyp)[::-1]
        token_tagged_hyp = self.tokenizer.encode(tagged_hyp)[::-1]
        for idx, t_tagged in enumerate(token_tagged_hyp):
            if t_tagged == token_untagged_hyp[idx]:
                continue
            else:
                return len(token_untagged_hyp)-idx

    def get_score_token(self, target, category, severity):
        score = self.get_score(category, severity)
        tagged_hyp = target
        untagged_hyp = tagged_hyp.replace('<v>', '').replace('</v>', '')
        untagged_token = self.get_untagged_token(tagged_hyp)
        untagged_score = self.no_error_score if self.emb_label else 0.0
        if severity in ['no-error', 'No-error', 'Neutral']:
            scored_token = [untagged_score for _ in untagged_token]
            return np.expand_dims(np.asarray(scored_token), axis=0)
        else: 
            start_idx = self.get_pos_start_tag(tagged_hyp)
            end_idx = self.get_pos_end_tag(tagged_hyp)
            scored_token = []    
            for i in range(len(untagged_token)):
                try:
                    if i < start_idx:
                        scored_token.append(untagged_score)
                    elif i <= end_idx:
                        scored_token.append(score)
                    else:
                        scored_token.append(untagged_score)
                except:
                    from IPython.core.debugger import Pdb; Pdb().set_trace()
            return np.expand_dims(np.asarray(scored_token), axis=0)
    
    def get_clean_keys(self, lines):
        MQM_data = {}
        clean_keys = []
        bad_keys = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            line = line.rstrip()
            split_line = line.split('\t')
            system = split_line[0]
            seg_id = int(split_line[3])
            target = split_line[6]
            key = (seg_id, system)
            if key not in MQM_data:
                MQM_data[key] = []
            MQM_data[key].append(target)
        for key, md in MQM_data.items():
            if all([self.get_untagged_hyp(md[0]) == self.get_untagged_hyp(md[i]) for i in range(len(md))]):
                clean_keys.append(key)
            else:
                bad_keys.append(key)
        # print(bad_keys)
        # return 
        if self.remove_strange:
            print('removing {} (seg_id, system) pairs, {} sentences'.format(len(bad_keys), sum([len(MQM_data[k]) for k in bad_keys])))
        return clean_keys
    
    def get_maj_avg(self, token_score_np):
        n_rater = token_score_np.shape[0]
        seq_len = token_score_np.shape[1]
        n_maj = math.ceil(n_rater/2)
        count_up_ls = [0]*seq_len
        return_score_token = [0]*seq_len
        for token_score in token_score_np:
            for i, score in enumerate(token_score):
                if score > 0.0:
                    count_up_ls[i] += 1
                return_score_token[i] += score
        for i, count in enumerate(count_up_ls):
            if count < n_maj:
                return_score_token[i] = 0.0
            return_score_token[i] /= n_rater
        return return_score_token
    
    def get_heavy_avg(self, token_score_np):
        # n_rater = token_score_np.shape[0]
        # seq_len = token_score_np.shape[1]
        # n_maj = math.ceil(n_rater/2)
        # return_score_token = [0]*seq_len
        pass
        # return return_score_token
    
    def get_MQM_keyed_data(self, lang):
        MQM_data = {}
        MQM_dic = {}
        fname = self.get_MQM_fname(self.MQM_tag_files, lang)
        rlines = self.load_file(fname)
        clean_keys = self.get_clean_keys(rlines)
        for i, line in enumerate(rlines):
            if i == 0:
                continue
            line = line.rstrip()
            split_line = line.split('\t')
            system = split_line[0]
            doc = split_line[1]
            doc_id = split_line[2]
            seg_id = int(split_line[3])
            rater = split_line[4]
            source = split_line[5]
            target = split_line[6]
            category = split_line[7]
            severity = split_line[8]
            score = self.get_score(category, severity)
            # score_token, target_token = self.get_score_token(target, category, severity)
            key = (lang, system, seg_id)
            if (seg_id, system) not in clean_keys and self.remove_strange:
                continue
            if key in MQM_dic:
                flag = False
                for d in MQM_dic[key]:
                    if d['rater'] == rater:
                        d['severity'].append(severity)
                        d['category'].append(category)
                        d['score'].append(score)
                        d['target'].append(target)
                        flag = True
                if flag == False:
                    MQM_dic[key].append({'system':system, 'seg_id':seg_id, 
                                         'lang': lang,
                                         'rater':rater, 'source':source,
                                         'target':[target],
                                         'category':[category],
                                         'severity':[severity],
                                         'score':[score]})
            else:
                MQM_dic[key] = [{'system':system, 'seg_id':seg_id,
                                 'lang': lang,
                                 'rater':rater, 'source':source,
                                 'target':[target],
                                 'category':[category],
                                 'severity':[severity],
                                 'score':[score]}]
        ret_dic = {}     
        for key in MQM_dic.keys():
            lang = key[0]
            system = key[1]
            seg_id = key[2]
            score = np.mean([sum(a['score']) for a in MQM_dic[key]])
            source = MQM_dic[key][0]['source']
            rater = []
            target = []
            category = []
            severity = []
            for md in MQM_dic[key]:
                for tgt, cat, sev in zip(md['target'], md['category'], md['severity']):
                    untagged_tgt = tgt.replace('<v>', '').replace('</v>', '')
                    if untagged_tgt == tgt and sev not in ['no-error','No-error','Neutral']:
                        continue
                    tgt_token = self.get_untagged_token(tgt)
                    score_token = self.get_score_token(tgt,cat,sev)
                    tmp_key = (key[0], key[1], key[2], untagged_tgt)
                    if tmp_key not in ret_dic:
                        ret_dic[tmp_key] = {'lang':lang,
                                            'system':system,
                                            'seg_id':int(seg_id),
                                            'src':source,
                                            'hyp':untagged_tgt,
                                            'avg_score':score,
                                            'target_token':tgt_token,
                                            'rater':[md['rater']],
                                            'tagged_hyp':[tgt],
                                            'severity':[sev],
                                            'category':[cat],
                                            'score_token':score_token}
                    else:
                        ret_dic[tmp_key]['rater'].append(md['rater'])
                        ret_dic[tmp_key]['tagged_hyp'].append(tgt)
                        ret_dic[tmp_key]['severity'].append(sev)
                        ret_dic[tmp_key]['category'].append(cat)
                        try:
                            ret_dic[tmp_key]['score_token'] = np.append(ret_dic[tmp_key]['score_token'],
                                                                        score_token,
                                                                        axis=0)
                        except:
                            from IPython.core.debugger import Pdb; Pdb().set_trace()
                        
        for key in ret_dic.keys():
            rater_num_dic = {}
            for r in ret_dic[tmp_key]['rater']:
                if r not in rater_num_dic:
                    rater_num_dic[r] = 0
                rater_num_dic[r] += 1
            rater_num = len(list(rater_num_dic.keys()))
            try:
                ret_dic[key]['avg_score_token'] = (np.sum(np.asarray(ret_dic[key]['score_token']),
                                                          axis=0)/rater_num).tolist()
                ret_dic[key]['maj_avg_score_token'] = self.get_maj_avg(ret_dic[key]['score_token'])
                ret_dic[key]['heavy_avg_score_token'] = self.get_heavy_avg(ret_dic[key]['score_token'])
                ret_dic[key]['score_token'] = ret_dic[key]['score_token'].tolist()
            except:
                from IPython.core.debugger import Pdb; Pdb().set_trace()
        return ret_dic
    
    def get_MQM_data(self, data_idx=0):
        langs = self.get_langs()
        all_data = []
        for lang in langs:
            MQM_data = self.get_MQM_keyed_data(lang)
            srcs = self.get_srcs(lang)
            refs = self.get_refs(lang)
            for key in MQM_data.keys():
                sys = key[1]
                seg_id = key[2]
                hyp = key[3]
                src = srcs[seg_id - 1]
                ref = refs[seg_id - 1]
                src_token = self.get_src_ref_token(src)
                ref_token = self.get_src_ref_token(ref)
                hyp_token = MQM_data[key]['target_token']
                avg_score = MQM_data[key]['avg_score']
                rater = MQM_data[key]['rater']
                tagged_hyp = MQM_data[key]['tagged_hyp']
                severity = MQM_data[key]['severity']
                category = MQM_data[key]['category']
                score_token = MQM_data[key]['score_token']
                if self.agreement == 'low':
                    avg_score_token = MQM_data[key]['avg_score_token']
                elif self.agreement == 'high':
                    avg_score_token = MQM_data[key]['maj_avg_score_token']
                elif self.agreement == 'heavy':
                    avg_score_token = MQM_data[key]['heavy_avg_score_token']
                all_data.append(self.to_example(data_idx,
                                                lang,
                                                sys,
                                                seg_id,
                                                src,
                                                ref,
                                                "ref-A",
                                                hyp,
                                                tagged_hyp,
                                                src_token,
                                                ref_token,
                                                hyp_token,
                                                score_token,
                                                avg_score_token,
                                                avg_score,
                                                rater,
                                                severity,
                                                category))
                data_idx += 1
        return all_data
    
    def get_google_MQM_data(self):
        google_MQM_data = []
        for lang in self.get_langs():
            fname = self.get_MQM_fname(self.MQM_avg_files, lang)
            rlines = self.load_file(fname)
            avg_data = []
            for i, line in enumerate(rlines):
                if i == 0:
                    continue
                line = line.rstrip()
                split_line = line.split(' ')
                system = split_line[0]
                score = split_line[1]
                seg_id = split_line[2]
                avg_data.append({'lang':lang,
                                 'system':system,
                                 'score':float(score)*(-1.0),
                                 'seg_id':int(seg_id)})
            google_MQM_data.extend(avg_data)
        return google_MQM_data
    
    def to_json(self, data_idx, lang, sys_name, seg_id, src, ref, ref_id,
                hyp, tagged_hyp, src_token, ref_token, hyp_token, token_score, mean_token_score,
                mean_score,
                rater, severity, category):
        json_dict = {"data_idx":data_idx,
                     "year": "20", "lang": lang, "system": sys_name,
                     "seg_id": seg_id, "src": src, 
                     "ref": ref, "ref_id":ref_id,
                     "hyp": hyp, "tagged_hyp":tagged_hyp,
                     "src_token" : src_token,
                     "ref_token" : ref_token, 
                     "hyp_token": hyp_token,
                     "token_score": token_score,
                     "mean_token_score" : mean_token_score,
                     "mean_score": mean_score,
                     "rater":rater,
                     'severity':severity, 'category':category}
        return json.dumps(json_dict)
    
    def to_example(self, data_idx, lang, sys_name, seg_id, src, ref, ref_id,
                hyp, tagged_hyp, src_token, ref_token, hyp_token, token_score, mean_token_score, mean_score,
                rater, severity, category):
        return {"data_idx":data_idx, "lang": lang, "system": sys_name,
                "seg_id": seg_id, "src": src, 
                "ref": ref, "ref_id":ref_id,
                "hyp": hyp, "tagged_hyp":tagged_hyp,
                "src_token": src_token,
                "ref_token" : ref_token,
                "hyp_token": hyp_token,
                "token_score": token_score,
                "mean_token_score": mean_token_score,
                "mean_score": mean_score,
                "rater":rater,
                'severity':severity, 'category':category}
    
    def split_example(self, all_example):
        all_dic = {}
        train_example = []
        dev_example = []
        train_size = int(len(all_example)*(1-self.dev_ratio))
        
        for example in all_example:
            key = (example['lang'], example['seg_id'])
            if key not in all_dic:
                all_dic[key] = []
            all_dic[key].append(example)
        key_list = list(all_dic.keys())
        random.shuffle(key_list)
        for key in key_list:
            if len(train_example) < train_size:
                train_example.extend(all_dic[key])
            else:
                dev_example.extend(all_dic[key])
        return train_example, dev_example
    
    def write_file(self, examples, fname):
        with open(fname, mode='w', encoding='utf-8') as w:
            for ex in examples:
                json_ex = self.to_json(ex['data_idx'], 
                                       ex['lang'],ex['system'],
                                       ex['seg_id'],ex['src'],
                                       ex['ref'],ex['ref_id'],
                                       ex['hyp'], ex['tagged_hyp'], 
                                       ex['src_token'],
                                       ex['ref_token'],
                                       ex['hyp_token'],
                                       ex['token_score'],
                                       ex['mean_token_score'],
                                       ex['mean_score'], 
                                       ex['rater'],
                                       ex['severity'], ex['category'])
                w.write(json_ex)
                w.write('\n')
        print('{} example for {}'.format(len(examples), fname))
                
    def make_MQM_corpus(self):
        all_example = self.get_MQM_data()
        self.write_file(all_example, self.save_dir.replace('.json', '_full.json'))
        
        if self.split_dev:
            train_example, dev_example = self.split_example(all_example)
            self.write_file(train_example, self.save_dir.replace('.json', '_train.json'))
            self.write_file(dev_example, self.save_dir.replace('.json', '_dev.json'))
        
    
    def check_no_tag(self):
        langs = self.get_langs()
        no_tags = {}
        for lang in langs:
            no_tags[lang] = []
            fname = self.get_MQM_fname(self.MQM_tag_20_files, lang)
            rlines = self.load_file(fname)
            for i, line in enumerate(rlines):
                if i == 0:
                    continue
                line = line.rstrip()
                split_line = line.split('\t')
                system = split_line[0]
                doc = split_line[1]
                doc_id = split_line[2]
                seg_id = int(split_line[3])
                rater = split_line[4]
                source = split_line[5]
                target = split_line[6]
                category = split_line[7]
                severity = split_line[8]
                untagged_hyp = target.replace('<v>', '').replace('</v>', '')
                if untagged_hyp == target and severity not in ['no-error', 'No-error', 'Neutral']:
                    no_tags[lang].append({'hyp':target, 'severity':severity,
                                          'category':category, 'rater':rater})
        return no_tags

importer = MQM_importer20('/ahc/work3/kosuke-t/WMT/wmt20_mqm_clean_xlm_low_agreement.json',
                          emb_label=False, 
                          emb_only_sev=False,
                          tokenizer_name='xlm-roberta-large',
                          google_score=True, 
                          split_dev=True,
                          remove_strange=True,
                          agreement='low')
importer.make_MQM_corpus()
# importer = MQM_importer20('/ahc/work3/kosuke-t/WMT/wmt20_mqm_clean_xlm_high_agreement.json',
#                           emb_label=False, 
#                           emb_only_sev=False,
#                           tokenizer_name='xlm-roberta-large',
#                           google_score=True, 
#                           split_dev=True,
#                           remove_strange=True,
#                           agreement='high')
# importer.make_MQM_corpus()
# importer = MQM_importer20('/ahc/work3/kosuke-t/WMT/wmt20_mqm_clean_xlm_high_agreement.json',
#                           emb_label=False, 
#                           emb_only_sev=False,
#                           tokenizer_name='xlm-roberta-large',
#                           google_score=True, 
#                           split_dev=True,
#                           remove_strange=True,
#                           agreement='heavy')
# importer.make_MQM_corpus()


# In[ ]:


# importer = MQM_importer20('/ahc/work3/kosuke-t/WMT/wmt20_mqm.json',emb_label=False, 
#                           emb_only_sev=False, google_score=True, split_dev=True)
# MQM_data = importer.get_MQM_data()


# In[ ]:





# In[ ]:





# In[ ]:


DATA_HOME = '/ahc/work3/kosuke-t/WMT'

MQM_1hot_only_severity_to_vec = {"no-error":np.asarray([1, 0, 0]), 
                                 "No-error":np.asarray([1, 0, 0]),
                                 "Neutral":np.asarray([0, 1, 0]),
                                 "Minor":np.asarray([0, 1, 0]),
                                 "Major":np.asarray([0, 0, 1])}
MQM_tag_list = ["No-error", 
                "Neutral",
                "Minor",
                "Major"]


class MQM_importer21(MQM_importer20):
    def __init__(self, save_dir, 
                 emb_label=False, emb_only_sev=True, 
                 no_error_score=np.asarray([1, 0, 0]),
                 tokenizer_name='xlm-roberta-large',
                 google_score=True, split_dev=False,
                 dev_ratio=0.1, remove_strange=False,
                 agreement='low'):
        super(MQM_importer21, self).__init__(save_dir, 
                                             emb_label,
                                             emb_only_sev,
                                             no_error_score,
                                             tokenizer_name=tokenizer_name,
                                             google_score=google_score,
                                             split_dev=split_dev,
                                             dev_ratio=dev_ratio, 
                                             remove_strange=remove_strange,
                                             agreement=agreement)
        self.year = '21'
        self.src_files = os.path.join(DATA_HOME,
                                      'WMT21-data/sources/newstest2021.{}.src.{}'.format('LANG_LETTERS', 'LANG1_LETTERS'))
        self.ref_files = os.path.join(DATA_HOME,
                                      'WMT21-data/references/newstest2021.{}.ref.ref-A.{}'.format('LANG_LETTERS', 'LANG2_LETTERS'))
        self.MQM_avg_files = os.path.join(DATA_HOME,
                                          'wmt-mqm-human-evaluation/newstest2021/{}/mqm_newstest2021_{}.avg_seg_scores.tsv'.format('LANGWO_LETTERS', 'LANGWO_LETTERS'))
        self.MQM_tag_files = os.path.join(DATA_HOME,
                                          'wmt-mqm-human-evaluation/newstest2021/{}/mqm_newstest2021_{}.tsv'.format('LANGWO_LETTERS', 'LANGWO_LETTERS'))
        self.langs = ['en-de', 'zh-en']
        self.systems = {'en-de': ['hyp.Facebook-AI',
                                  'hyp.HuaweiTSC',
                                  'hyp.Nemo',
                                  'hyp.Online-W',
                                  'hyp.UEdin',
                                  'hyp.VolcTrans-AT',
                                  'hyp.VolcTrans-GLAT',
                                  'hyp.eTranslation',
                                  'hyp.metricsystem1',
                                  'hyp.metricsystem2',
                                  'hyp.metricsystem3',
                                  'hyp.metricsystem4',
                                  'hyp.metricsystem5',
                                  'ref.A',
                                  'ref.B',
                                  'ref.C',
                                  'ref.D'],
                        'zh-en': ['hyp.Borderline',
                                  'hyp.DIDI-NLP',
                                  'hyp.Facebook-AI',
                                  'hyp.IIE-MT',
                                  'hyp.MiSS',
                                  'hyp.NiuTrans',
                                  'hyp.Online-W',
                                  'hyp.SMU',
                                  'hyp.metricsystem1',
                                  'hyp.metricsystem2',
                                  'hyp.metricsystem3',
                                  'hyp.metricsystem4',
                                  'hyp.metricsystem5',
                                  'ref.A',
                                  'ref.B']}
    def get_srcs(self, lang):
        lang1 = lang[:2]
        fname = self.src_files.replace('LANG_LETTERS', lang).replace('LANG1_LETTERS', lang1)
        src_data = self.load_file(fname)
        src_data = [s.rstrip() for s in src_data]
        return src_data
    
    def get_refs(self, lang):
        lang2 = lang[-2:]
        fname = self.ref_files.replace('LANG_LETTERS', lang).replace('LANG2_LETTERS', lang2)
        ref_data = self.load_file(fname)
        ref_data = [r.rstrip() for r in ref_data]
        return ref_data
    
    # def get_hyps(self, lang):
    #     hyp_data = {}
    #     for sys in self.systems[lang]:
    #         fname = self.hyp_files.replace('LANG', lang, 2).replace('SYSTEM', sys)
    #         hyp_data[sys] = self.load_file(fname)
    #         hyp_data[sys] = [h.rstrip() for h in hyp_data[sys]]
    #     return hyp_data
                
importer = MQM_importer21('/ahc/work3/kosuke-t/WMT/wmt21_mqm_clean_xlm_high_agreement.json',
                          emb_label=False, 
                          emb_only_sev=False,
                          tokenizer_name='xlm-roberta-large',
                          google_score=True, 
                          split_dev=True,
                          remove_strange=True,
                          agreement='high')
importer.make_MQM_corpus()
# no_tags = importer.check_no_tag()


# In[5]:


class TED_importer(MQM_importer21):
    def __init__(self, save_dir,
                 emb_label=False, emb_only_sev=True, 
                 no_error_score=np.asarray([1, 0, 0]),
                 tokenizer_name='xlm-roberta-large',
                 google_score=True, split_dev=False, 
                 combine_20=True, dev_ratio=0.1,
                 remove_strange=False, agreement='low'):
        super(TED_importer, self).__init__(save_dir, 
                                           emb_label,
                                           emb_only_sev,
                                           no_error_score,
                                           tokenizer_name=tokenizer_name,
                                           google_score=google_score,
                                           split_dev=split_dev,
                                           dev_ratio=dev_ratio,
                                           remove_strange=remove_strange,
                                           agreement=agreement)
        self.importer20 = MQM_importer20(save_dir, 
                                           emb_label,
                                             emb_only_sev,
                                             no_error_score,
                                             tokenizer_name=tokenizer_name,
                                             google_score=google_score,
                                             split_dev=split_dev,
                                             dev_ratio=dev_ratio, 
                                         remove_strange=remove_strange,
                                         agreement=agreement)
        self.combine_20 = combine_20
        self.year = 'TED21'
        self.src_files = os.path.join(DATA_HOME,
                                      'WMT21-data/sources/tedtalks.{}.src.{}'.format('LANG_LETTERS', 'LANG1_LETTERS'))
        self.ref_files = os.path.join(DATA_HOME,
                                      'WMT21-data/references/tedtalks.{}.ref.ref-A.{}'.format('LANG_LETTERS', 'LANG2_LETTERS'))
        self.MQM_avg_files = os.path.join(DATA_HOME,
                                          'wmt-mqm-human-evaluation/ted/{}/mqm_ted_{}.avg_seg_scores.tsv'.format('LANGWO_LETTERS', 'LANGWO_LETTERS'))
        self.MQM_tag_files = os.path.join(DATA_HOME,
                                          'wmt-mqm-human-evaluation/ted/{}/mqm_ted_{}.tsv'.format('LANGWO_LETTERS', 'LANGWO_LETTERS'))
        self.langs = ['en-de', 'zh-en']
        self.systems = {'ende': ['Facebook-AI',
                                 'HuaweiTSC',
                                 'Nemo',
                                 'Online-W',
                                 'UEdin',
                                 'VolcTrans-AT',
                                 'VolcTrans-GLAT',
                                 'eTranslation',
                                 'metricsystem1',
                                 'metricsystem2',
                                 'metricsystem3',
                                 'metricsystem4',
                                 'metricsystem5',
                                 'ref'],
                        'zhen': ['Borderline',
                                 'DIDI-NLP',
                                 'Facebook-AI',
                                 'IIE-MT',
                                 'MiSS',
                                 'NiuTrans',
                                 'Online-W',
                                 'SMU',
                                 'metricsystem1',
                                 'metricsystem2',
                                 'metricsystem3',
                                 'metricsystem4',
                                 'metricsystem5',
                                 'ref',
                                 'refB']}
        
    def make_MQM_corpus(self):
        all_example = self.get_MQM_data()
        if self.combine_20:
            example_20 = self.importer20.get_MQM_data(len(all_example))
            all_example.extend(example_20)
        self.write_file(all_example, self.save_dir.replace('.json', '_full.json'))
        
        if self.split_dev:
            train_example, dev_example = self.split_example(all_example)
            self.write_file(train_example, self.save_dir.replace('.json', '_train.json'))
            self.write_file(dev_example, self.save_dir.replace('.json', '_dev.json'))
        
        
importer = TED_importer('/ahc/work3/kosuke-t/WMT/WMT20_TED21_mqm_clean_xlm_high_agreement.json',
                        emb_label=False, 
                        emb_only_sev=False,
                        tokenizer_name='xlm-roberta-large',
                        google_score=True,
                        split_dev=True,
                        combine_20=True,
                        dev_ratio=0.1,
                        remove_strange=True,
                        agreement='high')
importer.make_MQM_corpus()


# In[19]:


import pandas as pd
import pickle 
import numpy as np
import torch
import json
import random

fname1 = '/ahc/work3/kosuke-t/WMT/WMT20_TED21_mqm_clean_full.json'
fname2 = '/ahc/work3/kosuke-t/WMT/wmt21_mqm_clean_full.json'

def to_json(data_idx, lang, sys_name, seg_id, src, ref, ref_id,
                hyp, tagged_hyp, src_token, ref_token, hyp_token, token_score, mean_token_score,
                mean_score,
                rater, severity, category):
        json_dict = {"data_idx":data_idx,
                     "year": "20", "lang": lang, "system": sys_name,
                     "seg_id": seg_id, "src": src, 
                     "ref": ref, "ref_id":ref_id,
                     "hyp": hyp, "tagged_hyp":tagged_hyp,
                     "src_token" : src_token,
                     "ref_token" : ref_token, 
                     "hyp_token": hyp_token,
                     "token_score": token_score,
                     "mean_token_score" : mean_token_score,
                     "mean_score": mean_score,
                     "rater":rater,
                     'severity':severity, 'category':category}
        return json.dumps(json_dict)

def write_file(examples, fname):
    with open(fname, mode='w', encoding='utf-8') as w:
        for ex in examples:
            json_ex = to_json(int(ex['data_idx']), 
                                   ex['lang'],ex['system'],
                                   int(ex['seg_id']),ex['src'],
                                   ex['ref'],ex['ref_id'],
                                   ex['hyp'], ex['tagged_hyp'], 
                                   ex['src_token'],
                                   ex['ref_token'],
                                   ex['hyp_token'],
                                   ex['token_score'],
                                   ex['mean_token_score'],
                                   ex['mean_score'], 
                                   ex['rater'],
                                   ex['severity'], ex['category'])
            w.write(json_ex)
            w.write('\n')
    print('{} example for {}'.format(len(examples), fname))

def df_to_example(df):
    all_example = []
    for i in range(len(df)):
        all_example.append(df.iloc[i])
    return all_example
    
def split_example(all_example, dev_ratio=0.1):
    all_dic = {}
    train_example = []
    dev_example = []
    train_size = int(len(all_example)*(1-dev_ratio))

    for example in all_example:
        key = (example['lang'], example['seg_id'])
        if key not in all_dic:
            all_dic[key] = []
        all_dic[key].append(example)
    key_list = list(all_dic.keys())
    random.shuffle(key_list)
    for key in key_list:
        if len(train_example) < train_size:
            train_example.extend(all_dic[key])
        else:
            dev_example.extend(all_dic[key])
    return train_example, dev_example
    
def load_data(fname):
    with open(fname, "r") as f:
        df = pd.read_json(f, lines=True)
    return df

df1 = load_data(fname1)
df2 = load_data(fname2)
df = pd.concat([df1, df2])
all_example = df_to_example(df)
train_example, dev_example = split_example(all_example)

write_file(all_example, '/ahc/work3/kosuke-t/WMT/wmt20-21_mqm_clean_full.json')
write_file(train_example, '/ahc/work3/kosuke-t/WMT/wmt20-21_mqm_clean_train.json')
write_file(dev_example, '/ahc/work3/kosuke-t/WMT/wmt20-21_mqm_clean_dev.json')


# In[2]:


import csv
import pprint
fname = '/ahc/work3/kosuke-t/WMT/WMT21-data/wmt-enru-newstest2021.csv'

def get_tmp_dic(src, mt, ref, score, sys, testset, lang):
    return {'src':src, 'hyp':mt, 'ref':ref, 'label':float(score),
            'system':sys, 'testset':testset, 'lang':lang}

def load_csv_MQM(fname):
    data = []
    with open(fname, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)    
        for i, row in enumerate(reader):
            if i == 0:
                print(row)
                continue
            data.append(get_tmp_dic(row[0], row[1], row[2], row[3],
                                    row[4], row[5], row[6]))
    return data

data = load_csv_MQM(fname)
with open('')


# In[14]:


sorted_data = sorted(data, key=lambda x: x['label'])
count = {k:0 for k in range(-400, 101, 50)}
for d in sorted_data:
    score = d['label']
    for k in count.keys():
        if score <= k:
            count[k] += 1
            break
count


# In[ ]:




