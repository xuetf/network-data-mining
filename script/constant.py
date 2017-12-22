# -*- coding: utf-8 -*-
'''
constant definition
'''
data_root_path = '../data/'
#  save path
dir_path = data_root_path + "save"
all_word_cut_name = "all_messages_cut"
chi_feature_name = 'chi_features'
stop_path = data_root_path + 'stop.txt'

# label kind
pos, neg = 0, 1

# field name
label_name = 'label'
message_name = 'message'
pred_label_name = 'pred_label'
all_train_features_name = 'all_train_features_200'

# file name

short_message_path = data_root_path + 'short_message.txt' # train data path
no_label_short_message_path =  data_root_path + 'no_label_short_message.txt' # to predict data path
no_label_short_message_pred_result_path = data_root_path + 'no_label_short_message_pred_result.txt' # result of to predict data path
