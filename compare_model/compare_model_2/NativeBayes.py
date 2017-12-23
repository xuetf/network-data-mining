#coding:utf-8
from sklearn.externals import joblib                                #模型本地化保存
import jieba
import jieba.analyse
from scipy import sparse
import codecs

def cutWords(msg, stopWords):
    seg_list = jieba.cut(msg, cut_all=False)
    # key_list = jieba.analyse.extract_tags(msg,20) #get keywords
    leftWords = ""
    for i in seg_list:
        if (i not in stopWords):
            leftWords += i + " "
    return leftWords

def loadStopWords(file):
    stop = [line.strip().decode('utf-8') for line in open(file).readlines()]
    return stop

class MyClass:
    stop = loadStopWords("stopWord.txt")
    tv = joblib.load("TfidfVectorizer_model.m")
    clf = joblib.load("NaiveBayes_model.m")


_instance = MyClass()                          #_instance其实相当于全局变量  一直存放在静态常量区


def NaiveBayes_myTest(testData):
    """
    :param testData:  list格式
    :return: pred     list格式
    """
    # print type(testData)
    for i in range(len(testData)):
        testData[i] = cutWords(testData[i],_instance.stop)
        print testData[i].encode("utf-8")
    fea_test = _instance.tv.transform(testData);
    pred = _instance.clf.predict(fea_test)
    return pred.tolist()


if __name__ == "__main__":
    testData = []
    testData.append("我是正常短信吗？")
    testData.append(".x月xx日推出凭证式国债x年期x.xx.xx%，x年期x.xx%到期一次还本付息。真情邮政，为您竭诚服务！  咨询电话xxxx-xx")

    # #测试函数
    pred = NaiveBayes_myTest(testData)

    # print type(pred)," ",len(pred)
    print pred
