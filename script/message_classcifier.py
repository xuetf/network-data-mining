# -*- coding: utf-8 -*-
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
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
    def load_model(self, model_name):
        self.classifier = load_from_pickle(dir_path, model_name)
        self.selected_features = load_from_pickle(dir_path, chi_feature_name)
        return self



    def fit(self, train_data, classifier, n=250, is_load_from_file=False, model_name=None, select_feature_name=chi_feature_name, is_need_cut=True, train_feature_name=None):
        # load model from file if the model and feature file exists
        if is_load_from_file and is_exist_file(dir_path, model_name) and is_exist_file(dir_path, select_feature_name):
            return self.load_model(model_name)

        # 训练前预处理，获取训练特征
        selected_features, train_features = fit_preprocess(train_data=train_data, n=n, is_need_cut=is_need_cut,
                                             is_load_from_file=is_load_from_file, train_feature_name=train_feature_name)
        self.selected_features = selected_features

        # training model through scikit-learn interface
        print 'begin training, %d instances......' % (len(train_features))
        self.classifier = SklearnClassifier(classifier)  # nltk with scikit-learn interface inside
        self.classifier.train(train_features)  # train_features include text feature and labels
        print "iteration: %d" % self.classifier._clf.n_iter_

        # save model
        if is_load_from_file: dump_to_pickle(dir_path, model_name, self.classifier)
        return self


    def predict(self, test_x, is_need_cut=True):
        if isinstance(test_x, basestring): test_x = [test_x]
        test_messages =  cut_messages(test_x, is_load_from_file=False) if is_need_cut else test_x
        test_features = word2vec(test_messages, self.selected_features, is_test_mode=True) # word to vec
        print 'test on %d instances' % (len(test_features))
        return self.classifier.classify_many(test_features)



def cross_validate_score(data, k_fold=5, model=LogisticRegression()):
    # load data and cut word
    data_x, data_y = data[message_name], data[label_name]
    # split data into k fold
    clf = Message_Classcifier()
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1)#固定种子
    result = []
    for i, (train_index, validate_index) in enumerate(kf.split(data_x, data_y)):  # split data according to label balances
        print 'fold %d start......' % i
        train_data = data.ix[train_index] # 包含标签
        validate_x = data_x.ix[validate_index] # 不包含标签
        validate_y = data_y.ix[validate_index]
        clf.fit(train_data, model, n=1000, is_load_from_file=False, is_need_cut=False, train_feature_name='train_fold_%d_features'%i)
        pred_y = clf.predict(validate_x, is_need_cut=False)
        result.append(acc_precision_recall_score(validate_y, pred_y))
    print 'average score over %d fold cross data' % k_fold
    print 'accuracy | pos: precision, recall, f1_score | neg: precision, recall, f1_score'
    result = np.mean(result, axis=0)
    print result
    return result[-1] # f1_score of neg class

def learning_curve(data, model, classifier_name, n, is_need_cut=False, is_load_from_file=False, train_feature_name=all_train_features_name,
                   cv=5, train_sizes=np.linspace(.1, 1.0, 10), ylim=(0.8, 1.1), baseline=0.9):
    _, data_features = fit_preprocess(train_data=data, is_need_cut=is_need_cut, n=n,
                                        is_load_from_file=is_load_from_file, train_feature_name=train_feature_name)
    X, y = transform_features(data_features)
    plot_learning_curve(model, classifier_name, X=X, y=y, ylim=ylim, cv=cv,
                        train_sizes=train_sizes, baseline=baseline)

def adjust_parameter_validate_curve(data, model, model_name,
                                    n=1000, is_need_cut=False, is_load_from_file=False, train_feature_name=all_train_features_name, cv=5,):
    _, data_features = fit_preprocess(train_data=data, is_need_cut=is_need_cut, n=n,
                                      is_load_from_file=is_load_from_file, train_feature_name=train_feature_name)
    X, y = transform_features(data_features)

    param_name = 'class_weight'
    param_plot_range = np.arange(1, 2.1, 0.1)
    param_range = [{pos:1.0, neg:neg_class_weight} for neg_class_weight in param_plot_range]
    plot_validation_curve(model, model_name, X=X, y=y, param_name=param_name, param_range=param_range, param_plot_range=param_plot_range,
                          cv=cv)


def train_all_and_predict_no_label_data(data):
    print 'cut finished.........................'
    clf = Message_Classcifier()
    clf.fit(data, LogisticRegression(class_weight={pos: 0.5, neg: 0.8}), n=1000, is_load_from_file=False,
            is_need_cut=False, train_feature_name=all_train_features_name)
    to_predict_data = pd.read_csv(no_label_short_message_path, names=[message_name], sep='\t')
    print "to predict data length:", len(to_predict_data)
    pred_y = clf.predict(to_predict_data[message_name])
    to_predict_data[pred_label_name] = pred_y
    to_predict_data.to_csv(no_label_short_message_pred_result_path, sep='\t', index=False, header=None)


def precision_recall_curve(data, model,
                                    n=1000, is_need_cut=False, is_load_from_file=False, train_feature_name=all_train_features_name, cv=5):
    _, data_features = fit_preprocess(train_data=data, is_need_cut=is_need_cut, n=n,
                                      is_load_from_file=is_load_from_file, train_feature_name=train_feature_name)
    X, y = transform_features(data_features)
    plot_precision_recall_curve(model, X, y)


if __name__ == '__main__':
    data = pd.read_csv(short_message_path, names=[label_name, message_name], sep='\t')
    data[message_name] = cut_messages(data[message_name], is_load_from_file=True, name=all_word_cut_name)  # 统一切词

    # 交叉验证
    cross_validate_score(data, k_fold=5, model=LogisticRegression(class_weight={pos:1, neg:1.7}))


    # 交叉验证绘制学习曲线
    # learning_curve(data, LogisticRegression(class_weight={pos:1, neg:1.5}), 'LogisticRegression',n=1000,
    #                          train_sizes=np.linspace(.01, 1.0, 10))

    # 绘制precision_recall曲线
    # precision_recall_curve(data, LogisticRegression(class_weight={pos: 1, neg: 1.7}))

    # 最终在所有训练集上训练，并预测不带标签数据
    # train_all_and_predict_no_label_data(data)

    # 调参
    # adjust_parameter_validate_curve(data, LogisticRegression(), 'LogisticRegression Validation')


