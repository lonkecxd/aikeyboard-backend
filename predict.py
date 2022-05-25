# -*- coding: utf-8 -*-
"""
Created on May 2022

@author: 陈想东
"""

import os
import re
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from networks import NetworkAlbertTextCNN
from classifier_utils import get_feature_test,id2label
from hyperparameters import Hyperparamters as hp
import jieba
import jieba.analyse
import pandas as pd

reconmmendSymbols = []
class ModelAlbertTextCNN(object,):
    """
    Load NetworkAlbert TextCNN model
    """
    def __init__(self):
        self.albert, self.sess = self.load_model()
    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert =  NetworkAlbertTextCNN(is_training=False)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd,hp.file_load_model))
                print (checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert,sess

def get_prediction(sentence):
    """
    Prediction of the sentence's label.
    """
    MODEL = ModelAlbertTextCNN()
    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids:[feature[2]],
          }
    prediction = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)[0]
    return [id2label(l) for l in np.where(prediction==1)[0] if l!=0]

def get_knowledge_points(s):
    jieba.load_userdict('./data/1.discrete_dict.txt')
    keywords = []
    with open('./data/1.discrete_dict.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            keyword = line.strip('\n').split(' ')[0]
            keywords.append(keyword)
    jieba.analyse.set_idf_path("./data/2.idf_dict.txt")
    tags = jieba.analyse.extract_tags(s, topK=10)
    tags = list(filter(lambda x:x in keywords, tags))
    return tags[:4]
def findRelativeConcepts(concept):
    df = pd.read_csv('./data/3.知识关联图谱.csv', encoding='utf-8', header=0, delimiter='#')
    map = {}
    Max = 0
    noFather = True
    for row in df.iterrows():
        fatherConcept = row[1]['父概念']
        childrenConcepts = list(row[1]['子概念'].split(','))
        map.setdefault(fatherConcept,0)
        if concept in childrenConcepts or concept==fatherConcept:
            return childrenConcepts
    return []
def findFatherConcept(concept):
    df = pd.read_csv('./data/3.知识关联图谱.csv', encoding='utf-8', header=0, delimiter='#')
    map = {}
    Max = 0
    noFather = True
    for row in df.iterrows():
        fatherConcept = row[1]['父概念']
        childrenConcepts = list(row[1]['子概念'].split(','))
        map.setdefault(fatherConcept,0)
        if concept in childrenConcepts or concept==fatherConcept:
            return fatherConcept
    return ''
def get_symbols_from_concepts(concepts):
    scenes = {}
    with open('./data/4.符号-场景.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            scene = line.strip('\n').split(' ')[0]
            concept = scene.split(':')[0]
            symbols = scene.split(':')[1].split(',')
            scenes.setdefault(concept,symbols)
    map = {}
    for concept in concepts:
        if concept in scenes:
            symbols = scenes[concept]
            for symbol in symbols:
                map.setdefault(symbol,0)
                map[symbol]+=1
    map = dict(sorted(map.items(), key=lambda item: item[1], reverse=True))
    return list(map.keys())
def get_symbols_from_concept(concept):
    scenes = {}
    with open('./data/4.符号-场景.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            scene = line.strip('\n').split(' ')[0]
            c = scene.split(':')[0]
            symbols = scene.split(':')[1].split(',')
            scenes.setdefault(c,symbols)
    fatherConcept = findFatherConcept(concept)
    if fatherConcept==concept:
        childrenConcepts = findRelativeConcepts(concept)
        return get_symbols_from_concepts(childrenConcepts)
    else:
        return get_symbols_from_concepts([concept])
def get_l1_symbols(question):
    # 符号来源：题目中的符号，题目中关键词提取的比如“并”，TextCNN打标签并经过知识图谱扩展的知识点。最后采用投票法排序。
    # 1. 题目中的符号
    symbols = []
    with open('./data/symbols.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            keyword = line.strip('\n').split(' ')[0]
            symbols.append(keyword)
    s_from_question = list(filter(lambda x:x in symbols, list(question)))
    # 2. 关键词提取的符号
    jieba.load_userdict('./data/1.discrete_dict.txt')
    keywords = []
    with open('./data/1.discrete_dict.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            keyword = line.strip('\n').split(' ')[0]
            keywords.append(keyword)
    jieba.analyse.set_idf_path("./data/2.idf_dict.txt")
    tags = jieba.analyse.extract_tags(question, topK=20)
    keywords = list(filter(lambda x:x in keywords, tags))
    s_from_keywords = get_symbols_from_concepts(keywords)
    # 3. 从知识图谱扩展得到符号
    symbols_from_relative_concepts = []
    relative_concepts = []
    for keyword in keywords:
        result = findRelativeConcepts(keyword)
        relative_concepts.extend(result)
    relative_concepts = list(set(relative_concepts))
    for relative_concept in relative_concepts:
        symbols_from_relative_concepts.extend(get_symbols_from_concepts(relative_concept))
    all_symbols = list(set(s_from_question+s_from_keywords+symbols_from_relative_concepts))
    all_symbols = list(re.sub('[a-zA-XZ=!]+', '', ''.join(all_symbols)).replace("−", "")) #去掉字母（除了Y）,=,-
    print('l1_data' + str(all_symbols))
    reconmmendSymbols = all_symbols
    return all_symbols

def get_l2_symbols(s):
    jieba.load_userdict('./data/1.discrete_dict.txt')
    keywords = []
    with open('./data/1.discrete_dict.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            keyword = line.strip('\n').split(' ')[0]
            keywords.append(keyword)
    jieba.analyse.set_idf_path("./data/2.idf_dict.txt")
    tags = jieba.analyse.extract_tags(s, topK=10)
    tags = list(filter(lambda x:x in keywords, tags))
    m = {}
    for tag in tags:
        symbols = get_symbols_from_concept(tag)
        m.setdefault(tag,symbols)
    print('l2_data'+str(m))
    return m

def get_l3_symbols():
    symbols = []
    with open('./data/symbols.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            symbol = line.strip('\n').split(' ')[0]
            symbols.append(symbol)
    l3_symbols = list(filter(lambda x:x not in reconmmendSymbols,symbols))
    print('l3_data'+str(l3_symbols))
    return l3_symbols

def get_l4_symbols():
    l4_symbols = []
    with open('./data/common_used_symbols.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            symbol = line.strip('\n').split(' ')[0]
            l4_symbols.append(symbol)
    print('l4_data'+str(l4_symbols))
    return l4_symbols