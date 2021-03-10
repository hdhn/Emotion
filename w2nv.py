# -*- coding: utf-8 -*-
"""
@author: Ccccandy
"""

import gensim
import numpy as np
from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary

# 生成word2vec模型

data0 = open('data/wiki_corpus00', 'r',encoding='utf-8')
data1 = open('data/wiki_corpus01', 'r',encoding='utf-8')
data2 = open('data/wiki_corpus02', 'r',encoding='utf-8')

output = open('data/test.txt', 'w',encoding='utf-8')

for i in data0:
    output.writelines(i+'\n')     
data0.close()

for i in data1:
    output.writelines(i+'\n')     
data1.close()

for i in data2:
    output.writelines(i+'\n')     
data2.close()

output.close()

sentences = word2vec.Text8Corpus("data/test.txt")

model = gensim.models.Word2Vec(sentences, size=300, sg=1, iter=10)  
model.save("other/word2vec.model") 

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model):

    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
    w2indx = {v: k+1 for k, v in gensim_dict.items()}
    w2vec = {word: model[word] for word in w2indx.keys()}
    return w2indx, w2vec
   

model = gensim.models.Word2Vec.load('other/word2vec.model')
index_dict, word_vectors= create_dictionaries(model=model)

np.save('other/w2n_dic.npy',index_dict)
np.save('other/w2v_dic.npy',word_vectors)




