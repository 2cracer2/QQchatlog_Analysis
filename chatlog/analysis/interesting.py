"""
    因吹斯听 分析及统计

"""
import pandas as pd
from pymongo import MongoClient


class Interesting(object):
    def __init__(self):
        self.client = MongoClient()  # 默认连接 localhost 27017
        self.db = self.client.chatlog
        self.post = self.db.vczh

    def longest_name(self):
        """
        取出所有用户的name,并排序。
        ..note::由于聊天记录时间跨度大,有的聚聚改名频繁
        :return:top_list[('name',len(name)),(...,...),...] 按长度从大到小排序
        """
        res_list = []
        for doc in self.post.find({}, {'name': 1}):
            res_list.append(doc['name'])
        res_list = {}.fromkeys(res_list).keys()

        top_list = []
        for li in res_list:
            top_list.append((li, len(li)))
        return pd.DataFrame(data=sorted(top_list, key=lambda x: x[1], reverse=True), columns=['马甲名', '字符数']).head(10)

    def longest_formation(self):
        """
        所有记录中,跟队形最长的聊天记录。
        :return:top_list[('text',len(text)),(...,...),...] 按长度从大到小排序
        """
        res_list = []
        for doc in self.post.find({}, {'text': 1}):
            res_list.append(doc['text'])

        top_list = []
        # text 数据存储形式 [[sentences1],[sentences2],...] 队形大多只有一句 所以只考虑text长度为1的
        i = 0
        while i < len(res_list) - 1:
            if res_list[i][0] == '[图片]':
                i += 1
            elif res_list[i][0] == res_list[i + 1][0]:
                pos = i + 1
                while pos < len(res_list) - 1:
                    if res_list[pos][0] == res_list[pos + 1][0]:
                        pos += 1
                    else:
                        if pos - i + 1 > 2:
                            top_list.append((res_list[i][0], pos - i + 1))
                        i = pos + 1
                        break
            else:
                i += 1

        # 例如中间有人插话一句将队形打断的话,整合队形
        k = 0
        while k < len(top_list) - 1:
            if top_list[k][0] == top_list[k + 1][0]:
                top_list.append((top_list[k][0], top_list[k][1] + top_list[k + 1][1]))
                top_list.pop(k - 1)
                top_list.pop(k)
            else:
                k += 1
        return pd.DataFrame(data =sorted(top_list, key=lambda x: x[1], reverse=True),columns=['整齐的队形','次数'])

    # 返回违禁词发言者
    def get_senstive_words(self):
        self.post = self.db.senstive_words
        sw_list = list(self.post.find({}, {'_id': 0,'name': 1, 'time': 1, 'text':1}))

        return pd.DataFrame(sw_list).sort_values(by=['time'])



    def close(self):
        self.client.close()
