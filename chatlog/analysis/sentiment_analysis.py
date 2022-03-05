import pandas as pd
from pymongo import MongoClient
from snownlp import SnowNLP
import matplotlib.pyplot as plt
from chatlog.analysis.individual import Individual
import re


class Sentiment_Analysis:
    def __init__(self):
        self.client = MongoClient()  # 默认连接 localhost 27017
        self.db = self.client.chatlog
        self.post = self.db.vczh

    def get_all_numbers(self):
        word_list = []
        for i in self.post.find({}, {'_id': 0, 'time': 1, 'ID': 1, 'name': 1, 'text': 1, }):
            i['text'] = i['text'][0]
            word_list.append(i)
        df = pd.DataFrame(word_list)
        df['time'] = df.time.apply(lambda x:re.findall(r"(\d{4}-\d{1,2}-\d{1,2})",x)[0])
        df['time'] =pd.to_datetime(df['time']) #将时间列转换为时间格式
        return df


    # 单独成员情绪变化分析
    def single_number_sentiment(self,name):
        frame = self.get_all_numbers().sort_values(by=['ID', 'time'])
        frame = frame.set_index(['ID', 'time'])
        frame = frame.loc[name]
        # # 去除系统消息
        # frame = frame.drop(['10000', '1000000'], axis=0)
        frame['snlp_result'] = frame.text.apply(lambda x :SnowNLP(x).sentiments)
        return frame

    # 发言数数前几名成员的情绪变化
    def top_numbers_sentiment(self):
        frame = self.get_all_numbers().sort_values(by=['ID', 'time'])
        frame = frame.set_index(['ID','time'])
        ind = Individual()
        res_list = ind.most_speak('speak_num')
        res_list = res_list[0:5]
        ID_list = [i[0] for i in res_list]
        Name_list = [i[1] for i in res_list]
        frame = frame.loc[ID_list]
        frame['snlp_result'] = frame.text.apply(lambda x :SnowNLP(x).sentiments)
        frame = frame.reindex(columns=['snlp_result'])
        x = frame.groupby(by=['ID', 'time']).mean().sort_values(by=['ID', 'time'])

        plt.figure(figsize=(8, 4))
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.plot(list(x.loc[ID_list[0]].index), x.loc[ID_list[0]], color='green', label=Name_list[0])
        plt.plot(list(x.loc[ID_list[1]].index), x.loc[ID_list[1]], color='khaki', label=Name_list[1])
        plt.plot(list(x.loc[ID_list[2]].index), x.loc[ID_list[2]], color='skyblue', label=Name_list[2])
        plt.plot(list(x.loc[ID_list[3]].index), x.loc[ID_list[3]], color='crimson', label=Name_list[3])
        plt.plot(list(x.loc[ID_list[4]].index), x.loc[ID_list[4]], color='navy', label=Name_list[4])

        plt.legend()
        plt.ylabel('情绪积极度')
        plt.savefig('./img/sentiment_total.png', dpi=600)
        plt.show()
        plt.close()

    #todo 提取系统系统信息中群成员入群时间，展现群内人数动态变化


    #todo 由系统信息展现群内与自己关系（与**不是好友）

    def work(self):
        self.top_numbers_sentiment()




