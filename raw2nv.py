# -*- coding: utf-8 -*-
"""

@author: Ccccandy

"""

import pandas as pd
import jieba
import numpy as np
import gensim
from gensim.models import word2vec,KeyedVectors
import re
from gensim.corpora.dictionary import Dictionary
# 引用停词文档，添加了老师、地、得、的等，可自行添加
def stopwordslist():
    stopwords = [line.strip() for line in open('data/stopword1.txt',encoding='UTF-8').readlines()]
    return stopwords
#strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开

# 进行停用词的删减    
def stop(each):    
    stopword=stopwordslist()    # 获取停用词
    new=[]              # 重新创建一个list保存最后的
    for i in each:
        if i not in stopword:
            new.append(i)
    return new


data1 = pd.read_excel('data/test.xlsx', usecols = [0],encoding='utf-8')  # 读取excel文件
data2 = pd.read_excel('data/train.xlsx', usecols = [0],encoding='utf-8')  # 读取excel文件


a1=data1.values.tolist()                                    # 转化为list形式，便于后续分词
a2=data2.values.tolist()                                    # 转化为list形式，便于后续分词

 
batch1=[''.join(str(b)) for b in a1]                                # 把一个小列表都转化为字符串形式   
batch2=[''.join(str(b)) for b in a2]                                # 把一个小列表都转化为字符串形式  
batch=batch1+batch2

output = open('data/t-test.txt', 'w',encoding='utf-8')
for i in batch:
    i=re.sub(r'//@.*?:', '',i)   # 处理转发
    i=re.sub(r'http://(\w|\.)+(/\w+)*', '',i)   #处理超链接
    cut = jieba.cut(i)
    cut_list = [ i for i in cut ]
    cut=stop(cut_list)
    cut = ' '.join(cut)
    output.writelines(cut+'\n')        
output.close()

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.Text8Corpus("data/t-test.txt")

# 训练模型，词向量的长度设置为100， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = gensim.models.Word2Vec(sentences, size=300, sg=1, iter=10)
#sg=1表示skip-gram，sg=2表示CBOW，size表示词向量的维度，iter表示迭代次数
model.wv.save_word2vec_format("other/word2vec.bin", binary=True)#词向量 
model.save("other/word2vec.model")#模型
print('1 over.')


import sys
from gensim.models import Word2Vec
import numpy as np
#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model):
    num=1
    w2indx={}
    for i in model.index2word:
        w2indx[i]=num
        num+=1

    w2vec = {word: model[word] for word in model.index2word}
    return w2indx, w2vec
   
model = gensim.models.KeyedVectors.load_word2vec_format("other/word2vec.bin", binary=True)
index_dict, word_vectors= create_dictionaries(model=model)

np.save('other/w2n_dic.npy',index_dict)
np.save('other/w2v_dic.npy',word_vectors)
print('2 over.')
