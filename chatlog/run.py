from chatlog.analysis.collectivity import Collectivity
from chatlog.analysis.individual import Individual
from chatlog.analysis.interesting import Interesting
from chatlog.analysis.sentiment_analysis import Sentiment_Analysis
from chatlog.base.read_chatlog import ReadChatlog
from chatlog.base.seg_word import SegWord
from chatlog.base.user_profile import UserProfile
from chatlog.visualization.charts import Charts
from chatlog.visualization.word_img import WordImg
from chatlog.analysis.find_keywords import key_words

if __name__ == '__main__':
    RC = ReadChatlog('./chatlog.txt')
    RC.work()  # 进行聊天记录的清洗并入库
    UP = UserProfile()
    UP.work()  # 构建简单的用户画像

    # 分词
    seg_words = SegWord()
    seg_words.work()

    with open('./out.txt', 'a', encoding='utf-8') as f:
        # Collectivity
        col = Collectivity()
        print('群聊天时间分布')
        f.write('群聊天时间分布'+'\n')
        print(col.get_all_speak_info())  # 群聊天时间分布
        f.write(str(col.get_all_speak_info()) + '\n')
        ind = Individual()
        print('小黑屋常客')
        f.write('小黑屋常客' + '\n')
        print(ind.longest_ban())  # 禁言时长的排名
        f.write(str(ind.longest_ban()) + '\n')

        print("水群怪排名")
        f.write('水群怪排名' + '\n')
        print(ind.most_speak('speak_num'))  # 发言次数的排名
        f.write(str(ind.most_speak('speak_num')) + '\n')
        print("最喜高谈阔论")
        f.write('最喜高谈阔论' + '\n')
        print(ind.most_speak('word_num'))  # 发言次数的排名
        f.write(str(ind.most_speak('word_num')) + '\n')
        print("没了表情包就不会说话")
        f.write('没了表情包就不会说话' + '\n')
        print(ind.most_speak('photo_num'))  # 发言次数的排名
        f.write(str(ind.most_speak('photo_num')) + '\n')


        # Interesting
        interest = Interesting()
        print("群内最统一的队形")
        f.write('群内最统一的队形' + '\n')
        print(interest.longest_formation())
        f.write(str(interest.longest_formation()) + '\n')

        print("最长的马甲排名")
        f.write('最长的马甲排名' + '\n')
        print(interest.longest_name())  #  最长的马甲排名
        f.write(str(interest.longest_name()) + '\n')

        print("检测到的违禁词发言")
        f.write('检测到的违禁词发言' + '\n')
        print(interest.get_senstive_words())  #  最长的马甲排名
        f.write(str(interest.get_senstive_words()) + '\n')

        keyword = key_words()
        keyword_list = keyword.get_key_words()
        print("群内关键词提取")
        f.write('群内关键词提取' + '\n')
        print('TF-IDF模型结果：'+str(keyword_list[0]) + '\nTextRank模型结果：'+ str(keyword_list[1]))
        f.write('TF-IDF模型结果：'+str(keyword_list[0]) + '\nTextRank模型结果：'+ str(keyword_list[1]))

    # 数据分析可视化
    chart = Charts()
    chart.work()
    # 生成词云
    word_imgs = WordImg()
    word_imgs.work()
    # 情感分析折线图
    s = Sentiment_Analysis()
    s.work()



