# -*- coding: utf-8 -*-
"""
使用svm和textcnn进行疫情微博加权预测
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import jieba
import re
from gensim.models import Word2Vec
from sklearn.externals import joblib
import json


batchsz = 32 # 批量大小
max_review_len = 30 # 每个句子词语量
embedding_len = 300 # 词向量长度
epochs = 20 # 训练次数


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

# in：分词列表
# out：分词转换为的数字列表
# 数据预处理，主要是转化为vector、等长处理，作为svm的输入
def w2v(each):
    ##!!!!!!!!!!!!!!!!!!!!!
    imdb_w2v = Word2Vec.load('other/word2vec.model')    
    vecs=[]    
    for word in each:
        vec = np.zeros(embedding_len).reshape((1, embedding_len))
        count = 0
        for i in word:
            try:
                vec += imdb_w2v[i].reshape((1, embedding_len))
                count += 1
            except KeyError:
                continue
            if count != 0:
                vec /= count 
        vecs.append(vec)
    return vecs

# 数据预处理，主要是分词、转化为数字、等长处理，作为cnn的输入 
def process(x):
    # 分词
    print('开始转化为数字ing')
    index_dict=np.load('other/w2n_dic.npy').item() 
    xx=[]
    n=1
    for i in x:
        xx.append([index_dict.get(each, 0) for each in i])
        print('第',n,'个转化完毕')
        n+=1
    print(xx)

    print('转化完毕')    
    # 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在后面填充
    print('等长处理ing')
    x_predict = keras.preprocessing.sequence.pad_sequences(xx, maxlen=max_review_len,padding='post')
    x_predict = tf.data.Dataset.from_tensor_slices(x_predict)
    x_predict = x_predict.batch(1, drop_remainder=True)
    print('处理完毕')

    return x_predict


# 预测cnn
def cnn_predict(x):
    model =tf.saved_model.load('model/new-cnn')
    cnn_pre=[model(i) for i in x]
    return cnn_pre

# 获取数据并分词
def load_fen():
    print('获取数据')
    train = pd.read_excel('data/test.xlsx', usecols = [0],encoding='utf-8')  # 读取excel文件
    a=train.values.tolist()   # 转换为列表形式
    x=[]
    
    for i in a:
        x.append(i[0])
        
    print(x)
    x=fenci(x)
    return x

if __name__ == '__main__':
    x=load_fen()
    cnn_x=process(x)
    
    cnn_pre = cnn_predict(cnn_x)
    cnn_pre=[i[0] for i in cnn_pre]


    cnn_pre = tf.argmax(cnn_pre,1)
    print(cnn_pre)

    y=[]

    for i in range(len(cnn_pre)):
        x={}
        if cnn_pre[i]==0:
            x['id']=i+1
            x['label']=5
            y.append(x)
        elif cnn_pre[i]==1:
            x['id']=i+1
            x['label']=4
            y.append(x)
        elif cnn_pre[i]==2:
            x['id']=i+1
            x['label']=3
            y.append(x)
        elif cnn_pre[i]==3:
            x['id']=i+1
            x['label']=2
            y.append(x)
        elif cnn_pre[i]==4:
            x['id']=i+1
            x['label']=1
            y.append(x)


    fp=open("result.txt", "w") 
    fp.write(json.dumps(y,indent=4))
    fp.close()
        
      
    


    





