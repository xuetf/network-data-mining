# -*- coding: utf-8 -*-
from message_classcifier import *


def compare_models(data):
    '''对比模型, 使用不同模型进行交叉验证并绘制学习曲线'''
    models= {"logisticRegression": LogisticRegression(),
            "Perceptron":Perceptron(),
            "DecisionTree":DecisionTreeClassifier(),
            "GBDT":GradientBoostingClassifier(),
             "SVC":svm.LinearSVC()
            }
    for name in models:
         print name, " begin..."
         cross_validate_score(data, k_fold=5, model=models[name], n=200)

    _, data_features = fit_preprocess(train_data=data, is_need_cut=False, n=200,
                                      is_load_from_file=False,
                                      train_feature_file_name=all_train_features_name)
    X, y = transform_features(data_features)
    plot_compare_learning_curve(models, X, y)


'''
main方法，
1.首先进行模型对比，得到最优分类器之一LogisticRegression
2.再进一步考察LR学习过程， 包括LR参数选择，学习曲线绘制，PR曲线绘制，交叉验证，最终训练模型并得到测试集的预测结果
这里注释了中间过程，运行哪个就把注释去掉
'''
if __name__ == '__main__':
    # 加载数据并进行分词处理
    data = pd.read_csv(short_message_path, names=[label_name, message_name], sep='\t')
    data[message_name] = cut_messages(data[message_name], is_load_from_file=True, name=all_word_cut_name)  # 统一切词

    # 对比实验：不同模型交叉验证/学习曲线
    # compare_models(data)

    # 调参
    # best_neg_class_weight = adjust_parameter_validate_curve(data, LogisticRegression(), 'LogisticRegression Validation')
    best_neg_class_weight = 1.4 # 最优参数结果

    # 交叉验证绘制学习曲线
    # learning_curve(data,LogisticRegression(class_weight={pos: 1, neg: best_neg_class_weight}))

    # 绘制precision_recall曲线
    # precision_recall_curve(data, LogisticRegression(class_weight={pos: 1, neg: best_neg_class_weight}))


    # 交叉验证，输出评价指标结果
    cross_validate_score(data, k_fold=5, model=LogisticRegression(class_weight={pos:1, neg:best_neg_class_weight}))

    # 最终在所有训练集上训练，并预测不带标签数据
    # train_all_and_predict_no_label_data(data, model=LogisticRegression(class_weight={pos:1, neg:best_neg_class_weight}))

