# -*- coding: utf-8 -*-
from message_classcifier import *

if __name__ == '__main__':
    data = pd.read_csv(short_message_path, names=[label_name, message_name], sep='\t')
    data[message_name] = cut_messages(data[message_name], is_load_from_file=True, name=all_word_cut_name)  # 统一切词
    # 调参
    # best_neg_class_weight = adjust_parameter_validate_curve(data, LogisticRegression(), 'LogisticRegression Validation')
    best_neg_class_weight = 1.4 # 最优参数

    # 交叉验证绘制学习曲线
    # learning_curve(data,LogisticRegression(class_weight={pos: 1, neg: best_neg_class_weight}))

    # 绘制precision_recall曲线
    # precision_recall_curve(data, LogisticRegression(class_weight={pos: 1, neg: best_neg_class_weight}))


    # 交叉验证，输出评价指标结果
    cross_validate_score(data, k_fold=5, model=LogisticRegression(class_weight={pos:1, neg:best_neg_class_weight}))

    # 最终在所有训练集上训练，并预测不带标签数据
    # train_all_and_predict_no_label_data(data, model=LogisticRegression(class_weight={pos:1, neg:best_neg_class_weight}))

    # 对比实验：不同模型交叉验证/学习曲线
    # compare_models(data)