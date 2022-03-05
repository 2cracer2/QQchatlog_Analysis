# -*- coding: utf-8 -*-
from io import StringIO
import math
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools
from pymongo import MongoClient
import pandas as pd
import re


def get_stopword_list():
    stop_word_path = './base/chinese_stopword.txt'
    stopword_list = [sw.replace('\n', '')
                     for sw in open(stop_word_path, encoding='utf-8').readlines()]
    return stopword_list


# 分词方法


def seg_to_list(sentence, pos):
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词，根据pos判断是否过滤除名词外的其他词性，再判断词是否在停用词表中，长度是否大于等于2等。


def word_filter(seg_list, pos):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据pos参数选择是否词性过滤
    # 不进行词性过滤，则将词性都标记为n,表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤高停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)
    return filter_list


# 数据加载


def load_data(pos=True, corpus_path='./base/chinese_stopword.txt'):
    doc_list = []
    for line in open(corpus_path, 'r', encoding='utf-8'):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)
    return doc_list


# idf值统计方法


def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
    tt_count = len(doc_list)
    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))
    # 对于没有在字典中的词，默认其尽在一个文档出现，得到默认idf值
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


# topK


def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# TF-IDF类


class TfIdf(object):
    # 训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    # 统计tf值

    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count
        return tf_dic

    # 按公式计算tf-idf

    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)
            tfidf = tf * idf
            tfidf_dic[word] = tfidf
        # 根据tf-idf排序，取排名前keyword_num的词作为关键词
        list_keywords = []
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            list_keywords.append(k)
            print(k + "/", end='')
        print()
        return list_keywords


# 主题模型


class TopicModel(object):
    #
    def __init__(self, doc_list, keyword_num, model="LSI", num_topics=4):
        # 使用gensim接口，将文本转为向量化表示
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算

        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()


def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    return tfidf_model.get_tfidf()


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword, end='/')
    return keywords
    # print()


def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


class key_words:
    def __init__(self):
        self.client = MongoClient()  # 默认连接 localhost 27017
        self.db = self.client.chatlog
        self.post = self.db.vczh

    def get_all_text(self):
        word_list = []
        for i in self.post.find({}, {'_id': 0, 'time': 1, 'text': 1, }):
            i['text'] = i['text'][0]
            word_list.append(i)
        df = pd.DataFrame(word_list)

        # df['time'] = df.time.apply(lambda x: re.findall(
        #     r"(\d{4}-\d{1,2}-\d{1,2})", x)[0])
        # df['time'] = pd.to_datetime(df['time'])  # 将时间列转换为时间格式
        df_string = df.to_csv()
        return df_string

    def get_key_words(self):
        keyword = key_words()
        text = keyword.get_all_text()
        pos = True
        seg_list = seg_to_list(text, pos)
        filter_list = word_filter(seg_list, pos)

        print('TF-IDF模型结果：')
        word_list1 = tfidf_extract(filter_list)
        print('TextRank模型结果：')
        word_list2 = textrank_extract(text)
        return [word_list1, word_list2]

    # def chart_key_words(self):
    #     list = self.get_key_words()
    #     print(list)

# if __name__ == '__main__':
#     keyword = key_words()
#     with open('./chatlog/chat.txt', 'w', encoding='utf-8') as f:
#         f.write(keyword.get_all_text())
#     text = keyword.get_all_text()
#     pos = True
#     seg_list = seg_to_list(text, pos)
#     filter_list = word_filter(seg_list, pos)

#     print('TF-IDF模型结果：')
#     tfidf_extract(filter_list)
#     print('TextRank模型结果：')
#     textrank_extract(text)

#     # print('LSI模型结果：')
#     # topic_extract(filter_list, 'LSI', pos)
#     # print('LDA模型结果：')
#     # topic_extract(filter_list, 'LDA', pos)