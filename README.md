基于SVM和CNN的多分类模型


代码文件介绍：

w2v.py：获取word2vec模型
输入：维基百科（wiki_corpus00，wiki_corpus01，wiki_corpus01）的分词txt，逐行读取写入test.txt
输出：word2vec.model，每个词组对应的300维词向量；word2vec.model.trainables.syn1neg.npy，word2vec.model.wv.vectors.npy，用于模型的调用
训练代码：
sentences = word2vec.Text8Corpus("test.txt")
model = gensim.models.Word2Vec(sentences, size=300, sg=1, iter=10)  

	
dic.py：生成word2vec词典和word2num词典
输入：model = Word2Vec.load('word2vec.model')
输出：w2n_dic.npy，词组转数字；n2v_dic.npy，词组转词向量
训练代码：
gensim_dict = Dictionary()
gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
w2indx = {v: k+1 for k, v in gensim_dict.items()}
w2vec = {word: model[word] for word in w2indx.keys()}


ill_svm.py和all_svm.py：生成通用和疫情各六个二分类模型
输入：原数据（all）--分词过滤（all，每句词长不一）--等长（all，30）--词向量（all，30，300）--平均后的词向量（all，300）
输出：类别是与否的概率
Key Codes：
1、word2vec更新：
imdb_w2v = Word2Vec.load('word2vec.model')
imdb_w2v.build_vocab(x, update=True)
imdb_w2v.train(x, total_examples=imdb_w2v.corpus_count, epochs=20)
imdb_w2v.save("word2vec.model") 
2、分词过滤：
i=re.sub(r'//@.*?:', '',i)   # 处理转发
i=re.sub(r'http://(\w|\.)+(/\w+)*', '',i)   #处理超链接
cut = jieba.cut(i)
cut_list = [ i for i in cut ]
cut=stop(cut_list)
3、词向量平均：
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
4、训练测试代码：
clf = SVC(kernel='rbf', verbose=True, probability=True)
clf.fit(train_vecs, y_train)
clf.score(test_vecs, y_test)
pred_result = [clf.predict(i) for i in str_vecs]


cnn.py：生成cnn模型
输入：原数据（all）--分词过滤（all，每句词长不一）--等长（all，30）--数字句子（all，30）
输出：六个类别的概率，最大数字的序列号即为对应情绪
Key Codes：
1、构建模型
def __init__(self):
    super(MyCNN, self).__init__()
    self.embedding = layers.Embedding(num_words,
                                          embedding_len,
                                          weights=[embedding_matrix],
                                          input_length=max_review_len,
                                          trainable=True)

    self.cnn1 = layers.Conv1D(128, 3, padding='same', strides=1, activation='relu')
    self.p1 = layers.MaxPooling1D(pool_size=28)
    self.cnn2 = layers.Conv1D(128, 4, padding='same', strides=1, activation='relu')
    self.p2 = layers.MaxPooling1D(pool_size=27)
    self.f = layers.Flatten()  # 打平层，方便全连接层处理
    self.d = layers.Dropout(0.3)
    self.outlayer = layers.Dense(6, activation='softmax')
2、模型装配、训练、验证
model.compile(optimizer = optimizers.Adam(0.001),
                  loss = losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
model.fit(db_train, epochs=epochs, validation_data=db_test)
model.evaluate(db_test)

ill_predict.py和all_predict.py：用于疫情和通用模型的预测
输入：svm模型以平均词向量输入，cnn模型以数字句子输入，操作方式与建模一致
输出：选取svm在1位置的六个概率进行softmax归一，再与cnn的六个概率相加，取最大数字的位置对应的情绪即为结果，以json方式存入txt中
训练代码：
x=load_fen()
svm_x=w2v(x)
cnn_x=process(x)
cnn_pre=cnn_predict(cnn_x)
cnn_pre=[i[0] for i in cnn_pre]
svm_pre=svm_predict(svm_x)
svm_pre=[np.exp(z)/sum(np.exp(z)) for z in svm_pre]
pre=svm_pre+cnn_pre
pre = tf.argmax(pre,1)

data文件：
stopword1.txt：停用词
wiki_corpus00、wiki_corpus01、wiki_corpus02：外部数据
all_train_data、ill_train_data：二八划分训练集文件
训练、验证、评测六个原数据集

model文件：
all_svm_model：内含六个通用微博svm模型
ill_svm_model：内含六个疫情微博svm模型
cnn_model：cnn模型大于1g，压缩文件为空，网盘链接为：链接：https://pan.baidu.com/s/16zp6SHQFYX28licdRAUHlg 提取码：t5ue

module版本：
tensorflow==2.0.0
numpy==1.16.2
scikit-learn==0.22.1
gast==0.2.2
gensim==3.8.3
jieba==0.39
pandas==1.0.2
xlrd==1.2.0
yaml==0.1.7
