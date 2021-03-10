# -*- coding: utf-8 -*-
"""

svm模型的构建

针对usual和virus的六个情绪各生成6个模型
放置于allmodel与illmodel文件中

"""
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import jieba
import re

 
n_dim = 300 # 词向量长度
 
# 分词 
# in：可for循环的数据集，如['a','b',……]
# out：[[a,],[b,],……]
def fenci(batch):    
    fc=[]      # 建立列表，存储每条数据的分词结果
    for i in batch:
        i=str(i)    # 将数据转化为字符串格式
        i=re.sub(r'//@.*?:', '',i)   # 处理转发
        i=re.sub(r'http://(\w|\.)+(/\w+)*', '',i)   #处理超链接
        cut = jieba.cut(i)
        cut_list = [ i for i in cut ]
        cut=stop(cut_list)
        fc.append(cut)  
    return fc

# 引用停词文档
def stopwordslist():
    stopwords = [line.strip() for line in open('data/stopword1.txt',encoding='UTF-8').readlines()]
    return stopwords

# 过滤停词文档
def stop(each):    
    stopword=stopwordslist()    # 获取停词文档
    new=[]              # 建立列表，存储过滤后的分词结果
    for i in each:
        if i not in stopword:
            new.append(i)
    return new
 
 
# 加载数据，x与y的转化（需要6次），分词进行保存
def loadfile(i):
    print('开始加载文件')
    train = pd.read_excel('data/virus_train.xlsx', usecols = [1,2], encoding='utf-8')  # 读取excel文件
    print('加载文件完毕')   
    a=train.values.tolist()   # 转换为列表形式
    x=[]
    y=[]
    if i==0:       
        for n in a:
            x.append(n[0])
            if n[1]=='happy':
                y.append(1)
            elif n[1]=='sad':
                y.append(0)
            elif n[1]=='surprise':
                y.append(0)
            elif n[1]=='fear':
                y.append(0)
            elif n[1]=='angry':
                y.append(0)
            elif n[1]=='neural':
                y.append(0)               
    elif i==1:
        for n in a:
            x.append(n[0])
            if n[1]=='happy':
                y.append(0)
            elif n[1]=='sad':
                y.append(1)
            elif n[1]=='surprise':
                y.append(0)
            elif n[1]=='fear':
                y.append(0)
            elif n[1]=='angry':
                y.append(0)
            elif n[1]=='neural':
                y.append(0)
    elif i==2:
        for n in a:
            x.append(n[0])
            if n[1]=='happy':
                y.append(0)
            elif n[1]=='sad':
                y.append(0)
            elif n[1]=='surprise':
                y.append(1)
            elif n[1]=='fear':
                y.append(0)
            elif n[1]=='angry':
                y.append(0)
            elif n[1]=='neural':
                y.append(0)
    elif i==3:
        for n in a:
            x.append(n[0])
            if n[1]=='happy':
                y.append(0)
            elif n[1]=='sad':
                y.append(0)
            elif n[1]=='surprise':
                y.append(0)
            elif n[1]=='fear':
                y.append(1)
            elif n[1]=='angry':
                y.append(0)
            elif n[1]=='neural':
                y.append(0)
                
    elif i==4:
        for n in a:
            x.append(n[0])
            if n[1]=='happy':
                y.append(0)
            elif n[1]=='sad':
                y.append(0)
            elif n[1]=='surprise':
                y.append(0)
            elif n[1]=='fear':
                y.append(0)
            elif n[1]=='angry':
                y.append(1)
            elif n[1]=='neural':
                y.append(0)               
    elif i==5:
        for n in a:
            x.append(n[0])
            if n[1]=='happy':
                y.append(0)
            elif n[1]=='sad':
                y.append(0)
            elif n[1]=='surprise':
                y.append(0)
            elif n[1]=='fear':
                y.append(0)
            elif n[1]=='angry':
                y.append(0)
            elif n[1]=='neural':
                y.append(1)    
    print('x、y拆分完毕')        
    x=fenci(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)    
    np.save('data/y_train.npy',y_train)
    np.save('data/y_test.npy',y_test)
    return x, x_train, x_test, y_train, y_test
  
# 获取单个句子中词语的词向量，并获取平均后的词向量
def buildWordVector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count 
    return vec
 
 
# 获取word2vec的词向量词典，并将x中的数据加入其中进行训练，将训练与验证数据转化为词向量
def get_train_vecs(x, x_train, x_test):
    # Initialize model and build vocab
    imdb_w2v = Word2Vec.load('word2vec.bin')
    imdb_w2v.build_vocab(x, update=True)
    imdb_w2v.train(x, total_examples=imdb_w2v.corpus_count, epochs=20)
    imdb_w2v.save("word2vec.bin") 
    
    train_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_train]) 
    np.save('data/train_vecs.npy', train_vecs)
    print(train_vecs.shape)

    test_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_test])
    np.save('data/test_vecs.npy', test_vecs)
    print(test_vecs.shape)
    
    return train_vecs, test_vecs
 
 
# 训练SVC模型
def svm_train(train_vecs, y_train, test_vecs, y_test,i):
    clf = SVC(kernel='rbf', verbose=True, probability=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, 'model/svm/svm_model'+str(i)+'.pkl')
    print(clf.score(test_vecs, y_test))
 
 
# 对SVC模型进行预测
def svm_predict(test,i):
    clf = joblib.load('model/svm/svm_model'+str(i)+'.pkl')
    imdb_w2v = Word2Vec.load('word2vec.bin')
    
    str_sege = fenci(test)
    str_vecs = [buildWordVector(z, n_dim, imdb_w2v) for z in str_sege]
    pred_result = [clf.predict(i) for i in str_vecs]
    print(pred_result)
 
 
if __name__ == '__main__':
    for i in range(6):
        n=i+1
        print('第'+str(n)+'个模型.')           
        print("loading data ...")
        x, x_train, x_test, y_train, y_test = loadfile(i)
        print("train word2vec model and get the input of svm model")
        train_vecs, test_vecs = get_train_vecs(x, x_train, x_test)
        print("train svm model...")        
        svm_train(train_vecs, y_train, test_vecs, y_test,i)
        print('第'+str(n)+'个模型完成.')      
    print("use svm model to predict...")
