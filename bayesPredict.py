from jieba import cut
from sklearn.naive_bayes import MultinomialNB
import re
from collections import Counter
from itertools import chain
from os import listdir
from numpy import array
import matplotlib.pyplot as plt
import time

#加载全部邮件
def load_label_files(label_file):
    # 创建空字典存储邮件相关信息
    label_dict ={}
    for line in open(label_file).readlines():
        # 标签格式为"spam ../data/000/000"
        list1 = line.strip().split("..")
        # 字典的键(key)为路径，值(value)为属性
        label_dict[list1[1].strip()] = list1[0].strip()
    return label_dict

#获得停用词表
def getStopWords():
    stopList=[]
    for line in open("./中文停用词表.txt"):
        stopList.append(line[:len(line)-1])
    return stopList

#获得词典
def get_word_list(content,wordsList,stopList):
    #分词结果放入res_list
    res_list = list(cut(content))
    for i in res_list:
        if i not in stopList and i.strip()!='' and i!=None:
            if i not in wordsList:
                wordsList.append(i)

#获得所有邮件的单词words和属性labels
def label_dicts(label_dict):
    #所有文件的所有单词
    words = []
    #邮件属性，1表示垃圾邮件，0表示正常邮件
    labels = []
    for key, value in label_dict.items():
        #每个文件的所有单词
        wordsList = []
        #每次遍历一个key和一个value，即一个邮件，并对它赋予属性
        for line in open("./trec06c" + key, errors = 'ignore'):
            #过滤掉非中文字符
            rule=re.compile(r"[^\u4e00-\u9fa5]")
            line=rule.sub("",line)
            #获得每个邮件的词典wordsList
            get_word_list(line, wordsList, stopList)
        words.append(wordsList)
        if value == 'spam':
            labels.append(1)
        else:
            labels.append(0)
    #返回值为tuple类型
    return words, labels

#获取出现频率最高的前top个单词
def get_top_word(top):
    freq = Counter(chain(*words))
    return [i[0] for i in freq.most_common(top)]

#获得停用词表，用于对停用词过滤
stopList=getStopWords()
#训练集邮件路径
label_path=r"C:\Users\eyefree\Desktop\python大作业\trec06c\full\demoIndex"
#邮件路径字典
label_dict = load_label_files(label_path)
print('训练集邮件数量:', len(label_dict))
#初始化时间
print('Initial')
init = time.time()
#words为训练集邮件的总词典(一个邮件的词典占据列表words中
#的一个位置)，labels为每个邮件对应的属性的集合
temp = label_dicts(label_dict)
words = temp[0]
labels = temp[1]
#获得频率最高的六百个单词
topWords = get_top_word(600)
#特征向量
vector = []
#获取特征向量，前topWords个单词的每个单词在每个邮件中出现的频率
for word in words:
    temp = list(map(lambda x: word.count(x), topWords))
    vector.append(temp)

#将list转化为array
vector = array(vector)
labels = array(labels)
#初始化结束
final = time.time()
print('initialTime:', final - init, 'seconds')
#创建模型，使用“多项式分布的朴素贝叶斯”的方法利用已知训练集进行训练
model = MultinomialNB()
#训练时间
init = time.time()
model.fit(vector, labels)
final = time.time()
print('trainTime:', final - init, 'seconds')
#进行批量预测
def predict(label_test_dict):
    #测试时间
    init = time.time()
    #p为概率，t为预测正确，f为预测错误，x为横坐标，y为纵坐标
    p = -1
    t = 0
    f = 0
    x = list(range(len(label_test_dict)))
    y = []
    all = label_dicts(label_test_dict)
    testWords = all[0]
    testLabels = all[1]
    for i in range(len(testWords)):
        currentVector = array(tuple(map(lambda x: testWords[i].count(x), topWords)))
        #将测试向量currentVector转化成一行，并进行预测
        result = model.predict(currentVector.reshape(1, -1))
        #testLabels里为对应的1和0，1为垃圾邮件，0为正常邮件
        if result == 1:
            if testLabels[i] == 1:
                t = t + 1
                y.append(1)
            else:
                f = f + 1
                y.append(0)
        else:
            if testLabels[i] == 0:
                t = t + 1
                y.append(1)
            else:
                f = f + 1
                y.append(0)
    if t > 0:
        p = t / (t + f)
    else:
        p = -1
    final = time.time()
    print('testTime:', final - init, 'seconds')
    print('精确度:', p)
    plt.scatter(x, y, s = 0.01)
    plt.show()

label_test_path=r"C:\Users\eyefree\Desktop\BayesSpam-master\src\trec06c\full\testIndex"
label_test_dict = load_label_files(label_test_path)
print('测试邮件数量:', len(label_test_dict))
predict(label_test_dict)
