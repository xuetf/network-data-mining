# -*- coding: utf-8 -*-
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")
from visualizer import *
from util import *
from feature_processer import *
from constant import *
from feature_extraction import *
from score import *

class Message_Classcifier(object):
    def __init__(self):
        pass
    def load_model(self, model_file_name):
        self.classifier = load_from_pickle(dir_path, model_file_name)
        self.selected_features = load_from_pickle(dir_path, chi_feature_name)
        return self

    def fit(self, train_data, classifier, n=250, is_load_from_file=False,
            model_file_name='final_model', select_feature_file_name=chi_feature_name, is_need_cut=True, train_feature_file_name=None):
        '''
        :param train_data:  训练数据，包含标签
        :param classifier:  使用的分类模型
        :param n: 抽取的特征数量
        :param is_load_from_file: 是否从文件中加载模型
        :param model_file_name: 若is_load_from_file=True,model_file_name是保存的模型的文件名
        :param select_feature_file_name: 若is_load_from_file=True，select_feature_file_name是保存的抽取的特征的文件名，用于下述构建短信-特征矩阵
        :param is_need_cut: 是否需要进行分词处理
        :param train_feature_file_name: 使用布尔模型或VSM模型构建的短信-特征矩阵所保存的文件名，非None说明从文件中加载
        :return: 模型本身
        '''
        # load model from file if the model and feature file exists
        if is_load_from_file and is_exist_file(dir_path, model_file_name) and is_exist_file(dir_path, select_feature_file_name):
            return self.load_model(model_file_name)

        # 训练前预处理，获取训练特征
        selected_features, train_features = fit_preprocess(train_data=train_data, n=n, is_need_cut=is_need_cut,
                                             is_load_from_file=is_load_from_file, train_feature_file_name=train_feature_file_name)
        self.selected_features = selected_features

        # training model through scikit-learn interface
        print 'begin training, %d instances......' % (len(train_features))
        self.classifier = SklearnClassifier(classifier)  # nltk with scikit-learn interface inside
        self.classifier.train(train_features)  # train_features include text feature and labels

        # save model
        if is_load_from_file: dump_to_pickle(dir_path, model_file_name, self.classifier)
        return self


    def predict(self, test_x, is_need_cut=True):
        '''
        预测
        :param test_x: 预测数据
        :param is_need_cut: 是否需要分词处理
        :return: 分类结果
        '''
        if isinstance(test_x, basestring): test_x = [test_x]
        test_messages =  cut_messages(test_x, is_load_from_file=False) if is_need_cut else test_x
        test_features = word2vec(test_messages, self.selected_features, is_test_mode=True) # word to vec
        print 'test on %d instances' % (len(test_features))
        return self.classifier.classify_many(test_features)



def cross_validate_score(data, k_fold=5, model=LogisticRegression(), n=1000):
    # load data and cut word
    data_x, data_y = data[message_name], data[label_name]
    # split data into k fold
    clf = Message_Classcifier()
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1)#固定种子
    result = []
    # split data according to label balances
    for i, (train_index, validate_index) in enumerate(kf.split(data_x, data_y)):
        print 'fold %d start......' % i
        train_data = data.ix[train_index] # 包含标签
        validate_x = data_x.ix[validate_index] # 不包含标签
        validate_y = data_y.ix[validate_index]
        clf.fit(train_data, model, n=n, is_load_from_file=False, is_need_cut=False, train_feature_file_name='train_fold_%d_features_%d'%(i,n))
        pred_y = clf.predict(validate_x, is_need_cut=False)
        result.append(acc_precision_recall_score(validate_y, pred_y))
    print 'average score over %d fold cross data' % k_fold
    print 'accuracy | pos: precision, recall, f1_score | neg: precision, recall, f1_score'
    result = np.mean(result, axis=0)
    print result
    return result[-1] # f1_score of neg class

def learning_curve(data, model, classifier_name='LogisticRegression', n=1000, is_need_cut=False, is_load_from_file=False, train_feature_file_name=all_train_features_name,
                   cv=5, train_sizes=np.linspace(.1, 1.0, 10), ylim=(0.8, 1.1), baseline=0.9):
    '''绘制学习曲线'''
    _, data_features = fit_preprocess(train_data=data, is_need_cut=is_need_cut, n=n,
                                        is_load_from_file=is_load_from_file, train_feature_file_name=train_feature_file_name)
    X, y = transform_features(data_features)
    plot_learning_curve(model, classifier_name, X=X, y=y, ylim=ylim, cv=cv,
                        train_sizes=train_sizes, baseline=baseline)

def adjust_parameter_validate_curve(data, model, model_file_name,
                                    n=1000, is_need_cut=False, is_load_from_file=False, train_feature_file_name=all_train_features_name, cv=5,):
    '''调参'''
    _, data_features = fit_preprocess(train_data=data, is_need_cut=is_need_cut, n=n,
                                      is_load_from_file=is_load_from_file, train_feature_file_name=train_feature_file_name)
    X, y = transform_features(data_features)
    param_name = 'class_weight'
    param_plot_range = np.arange(1, 2.1, 0.1)
    param_range = [{pos:1.0, neg:neg_class_weight} for neg_class_weight in param_plot_range]
    return plot_validation_curve(model, model_file_name, X=X, y=y, param_name=param_name, param_range=param_range, param_plot_range=param_plot_range,
                          cv=cv)


def train_all_and_predict_no_label_data(data, model, n=1000):
    '''在所有数据上进行训练'''
    clf = Message_Classcifier()
    clf.fit(data, model, n=n, is_load_from_file=False,
            is_need_cut=False, train_feature_file_name= all_train_features_name)
    to_predict_data = pd.read_csv(no_label_short_message_path, names=[message_name], sep='\t')
    print "to predict data length:", len(to_predict_data)
    pred_y = clf.predict(to_predict_data[message_name])
    to_predict_data[pred_label_name] = pred_y
    to_predict_data.to_csv(no_label_short_message_pred_result_path, sep='\t', index=False, header=None)


def precision_recall_curve(data, model,
                                    n=1000, is_need_cut=False, is_load_from_file=False, train_feature_file_name=all_train_features_name, cv=5):
    _, data_features = fit_preprocess(train_data=data, is_need_cut=is_need_cut, n=n,
                                      is_load_from_file=is_load_from_file, train_feature_file_name=train_feature_file_name)
    X, y = transform_features(data_features)
    plot_precision_recall_curve(model, X, y)


