类说明：
message_classcifier.py: 总的运行主类
util.py:保存中间数据的方法

data文件夹：
short_message.txt:老师给的带标签短信训练数据
no_label_short_message.txt:老师给的无标签短信测试集数据
no_label_short_message.pred_result.txt:模型对无标签短信的预测结果

save文件夹：保存中间过程数据。


运行结果：
D:\Anaconda2\python.exe F:/py-workspace/network-data-mining/message_classcifier.py
LogisticRegression:..........
train on 322880 instances
Building prefix dict from the default dictionary ...
Loading model from cache c:\users\xtf\appdata\local\temp\jieba.cache
Loading model cost 0.932 seconds.
Prefix dict has been built succesfully.
test on 188712 instances
metric from scikit learn................
accuracy: 0.991966594599
pos precision: 0.992849948468
pos recall: 0.998263091579
neg precision: 0.983559964333
neg recall: 0.935294117647
pos f1 score 0.995549161798
neg f1 score 0.958820014125

(array([ 0.99284995,  0.98355996]), array([ 0.99826309,  0.93529412]), array([ 0.99554916,  0.95882001]), array([169842,  18870], dtype=int64))
to predict data length: 197030
test on 197030 instances

Process finished with exit code 0
