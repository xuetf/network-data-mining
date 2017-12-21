# -*- coding: utf-8 -*-
import jieba
import io
from util import *
from constant import *
import datetime
import numpy as np
from feature_extraction import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def cut_messages(messages, name=None, is_load_from_file=True):
    if is_load_from_file and is_exist_file(dir_path, name):
        return load_from_pickle(dir_path, name)
    cut_messages_list = []
    stop = [line.strip() for line in io.open(stop_path, 'r', encoding='utf-8').readlines()]  # 停用词
    for i in range(50): stop.append("x" * i) # 脱敏词x去掉
    for message in messages:
        s = message.split('\n')
        fenci = jieba.cut(s[0], cut_all=False)  # False默认值：精准模式
        valid_words = list(set(fenci) - set(stop))
        cut_messages_list.append(valid_words)

    if is_load_from_file: dump_to_pickle(dir_path, name, cut_messages_list)
    return cut_messages_list


def word2vec(messages, selected_features, tag = None, is_test_mode=False, tol=1):
    #print 'begin word2vec...'
    #starttime = datetime.datetime.now()
    vec = []
    features = selected_features.keys()
    for message in messages:
        a = {}
        for word in message:
            if word in features:
                a[word] = 'True'
        if is_test_mode: # 测试模式，不允许贴标签
            vec.append(a)
        elif len(a) >= tol: #训练模式，贴标签。如果该条短信能够提取到有用特征大于阈值tol，则加入训练
            vec.append([a, tag])

    #endtime = datetime.datetime.now()
    #print (endtime - starttime).seconds
    #print 'finish word2vec.....'
    return vec

def construct_vsm_train_features(pos_messages, neg_messages, selected_features, name=None):
    '''构建训练数据的向量空间模型'''
    if name is not None and is_exist_file(dir_path, name): return load_from_pickle(dir_path, name)
    pos_features = word2vec(pos_messages, selected_features, pos)
    neg_features = word2vec(neg_messages, selected_features, neg)
    train_features = np.concatenate([pos_features, neg_features])
    if name is not None: dump_to_pickle(dir_path, name, train_features)
    return train_features


def fit_preprocess(train_data, n, is_need_cut, is_load_from_file, train_feature_file_name):
    '''训练前预处理，包括特征提取和特征表示'''
    # cut or not
    if is_need_cut: train_data[message_name] = cut_messages(train_data[message_name], all_word_cut_name, is_load_from_file)
    pos_messages = train_data[train_data[label_name] == pos][message_name].values
    neg_messages = train_data[train_data[label_name] == neg][message_name].values
    # Feature Selection by Chi-square Test Method
    selected_features = chi_features(n, pos_messages, neg_messages)
    # Construct Train features
    train_features = construct_vsm_train_features(pos_messages, neg_messages, selected_features, name=train_feature_file_name)
    return selected_features, train_features

def transform_features(data_features):
    X, y = list(zip(*data_features))
    X = DictVectorizer(sparse=True, dtype=float).fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    return X, y






