import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pymongo import MongoClient
from wordcloud import WordCloud, ImageColorGenerator

from chatlog.base.seg_word import SegWord


class WordImg(object):
    def __init__(self):
        self.client = MongoClient()  # 默认连接 localhost 27017
        self.db = self.client.chatlog
        self.post = self.db.word

    def close(self):
        self.client.close()

    def draw_wordcloud(self, word_dict, name):
        cat_mask = np.array(Image.open('./visualization/cat.png'))
        wc = WordCloud(font_path='./visualization/msyh.ttc',
                       width=800, height=400,
                       background_color="white",  # 背景颜色
                       mask=cat_mask,  # 设置背景图片
                       min_font_size=6
                       )
        wc.fit_words(word_dict)

        image_colors = ImageColorGenerator(cat_mask)
        # recolor wordcloud and show
        # we could also give color_func=image_colors directly in the constructor
        plt.imshow(wc)
        plt.axis("off")
        plt.savefig('./img/' + name + '.png', dpi=800)
        plt.close()

    def PL_wordcloud(self):
        word_dict = {'JAVA': ['java', 'jawa'], 'C++': ['c++', 'c艹'], 'C': ['c', 'c语言'],
                     'PHP': ['php'], 'Python': ['py', 'python'], 'C#': ['c#']}
        self.draw_wordcloud(self.word_fre(word_dict), sys._getframe().f_code.co_name)

    def all_wordcloud(self, word_len=0):
        word_dict = {}
        stop_word = ['图片', '表情', '说','[]','在','的','[',']','都']
        for doc in self.post.find({}):
            if len(doc['word']) > word_len and doc['word'] not in stop_word:
                word_dict[doc['word']] = doc['item']
        self.draw_wordcloud(word_dict, sys._getframe().f_code.co_name + str(word_len))

    def company_wordcloud(self):
        word_dict = {'Microsoft': ['微软', '巨硬', 'ms', 'microsoft'], 'Tencent': ['腾讯', 'tencent', '鹅厂'],
                     '360': ['360', '安全卫士', '奇虎'], 'Netease': ['netease', '网易', '猪场'],
                     'JD': ['jd', '京东', '某东', '狗东'], 'Taobao': ['淘宝', '天猫', 'taobao'],
                     'BaiDu': ['百度', '某度', 'baidu'], 'ZhiHu': ['zhihu', '知乎', '你乎', '某乎'],
                     'Sina': ['新浪', 'sina', '微博', 'weibo']}

        self.draw_wordcloud(self.word_fre(word_dict), sys._getframe().f_code.co_name)

    def word_fre(self, word_dict):
        word_fre = {}
        for key in word_dict.keys():
            word_fre[key] = 0

        res_dict = {}
        for doc in self.post.find({}):
            res_dict[doc['word']] = doc['item']

        for res_key in res_dict.keys():
            for word_key in word_dict.keys():
                if str(res_key).lower() in word_dict[word_key]:
                    word_fre[word_key] = word_fre[word_key] + res_dict[res_key]

        return word_fre

    def longest_formation_wordcloud(self):
        top_words = self.top_words()
        top_words = top_words[3:110]
        word_dict = {}
        # TODO 取出词频前10词汇以及频率
        for x in top_words:
            word_dict [x[0]] = int (x[1])

        cfc = np.array(Image.open('./visualization/cat2.png'))

        wc = WordCloud(font_path='./visualization/msyh.ttc',
                       width=1080, height=720,
                       mask=cfc,
                       background_color="white",  # 背景颜色
                       min_font_size=6
                       )
        wc.fit_words(word_dict)
        plt.imshow(wc)
        plt.axis("off")
        plt.savefig('./img/' + sys._getframe().f_code.co_name + '.png', dpi=800)
        plt.show()
        plt.close()

    def work(self):
        self.PL_wordcloud()
        self.company_wordcloud()
        self.all_wordcloud()
        self.longest_formation_wordcloud()
        self.close()

    # TODO 取出词频前10词汇以及频率
    def top_words(self):
        self.post = self.db.word
        top_list = []
        for doc in self.post.find({}, {'word': 1, 'item': 1, }):
            top_list.append((doc['word'], doc['item']))

        return sorted(top_list, key=lambda x: x[1], reverse=True)

