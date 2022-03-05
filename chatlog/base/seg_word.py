import re
from collections import Counter
import jieba
import numpy
import pandas as pd
from pymongo import MongoClient


class SegWord(object):
    def __init__(self):
        self.client = MongoClient()  # 默认连接 localhost 27017
        self.db = self.client.chatlog
        self.post = self.db.vczh

    def sed_all_words(self):
        stopword_list = []
        fp = open('./base/chinese_stopword.txt', 'r', encoding='utf-8')
        for line in fp.readlines():
            stopword_list.append(line.replace('\n', ''))
        fp.close()

        word_list = []
        self.post = self.db.vczh
        for doc in self.post.find({}, {'_id': 0, 'text': 1}):
            print(len(word_list))
            word_list.extend(jieba.lcut(doc['text'][0]))
        word_dict = Counter(word_list)
        self.post = self.db.word
        for key in word_dict.keys():
            if str(key) in stopword_list:
                print(key)
                continue
            self.post.insert({'word': key, 'item': word_dict[key]})

    def senstive_words_Detecter(self):

        # 记录分词后敏感词的发言者及时间
        senstive_words = []
        sw = open('./base/senstive_words.txt', 'r', encoding='utf-8')
        for line in sw.readlines():
            senstive_words.append(line.replace('\n', ''))
        sw.close()

        def is_senstive(listtxt):
            isb = numpy.NAN
            for i in listtxt:
                if str(i) in senstive_words:
                    isb = str(i)
                    break
            return isb

        frame =self.get_all_numbers()
        jieba.load_userdict('./base/senstive_words.txt')
        frame['seg_word'] = frame.text.apply(lambda x: jieba.lcut(x))
        frame['shit_word'] = frame.seg_word.apply(is_senstive)
        frame = frame[~pd.isna(frame.shit_word)]
        print(frame)
        # 将违禁词汇及对应发送者ID，时间写入数据库
        self.post = self.db.senstive_words
        data = frame.to_dict(orient='records')
        self.post.insert(data)


    def get_all_numbers(self):
        word_list = []
        self.post = self.db.vczh
        for i in self.post.find({}, {'_id': 0, 'time': 1, 'ID': 1, 'name': 1, 'text': 1, }):
            i['text'] = i['text'][0]
            word_list.append(i)
        df = pd.DataFrame(word_list)
    #    df['time'] = df.time.apply(lambda x: re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", x)[0])
        df['time'] = pd.to_datetime(df['time'])  # 将时间列转换为时间格式
        return df

    def close(self):
        self.client.close()

    def work(self):
        self.sed_all_words()
        self.senstive_words_Detecter()
        self.close()
