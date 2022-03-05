from fpdf import FPDF



class PDF(FPDF):
    # 绘制边框
    def lines(self):
        self.set_fill_color(247, 247, 247)  # 外部矩形的颜色
        self.rect(5.0, 5.0, 200.0, 287.0, 'DF')
        self.set_fill_color(255, 255, 255)  # 内部矩形的颜色
        self.rect(6.0,6.0, 198.0, 285.0, 'FD')
    # 设置标题
    def titles(self):
        self.set_xy(0.0, 0.0)
        self.set_font('Arial', 'B', 20)
        self.set_text_color(220, 50, 50)
        self.cell(w=210.0, h=40.0, align='C', txt="QQ CHAT_LOG  ANALYSIS", border=0)
    # 设置标题头图
    def imagex(self):
        self.set_xy(6.0, 6.0)
        self.image('./source/floral.png',  w=1586 / 80, h=1920 / 80)
        self.set_xy(183.0, 6.0)
        self.image('./source/flower2.png', w=1586 / 80, h=1920 / 80)
    #添加文本
    def texts(self, name):
        with open(name, 'rb') as xy:
            txt = xy.read().decode(encoding='utf-8')
        self.set_xy(10.0, 80.0)
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", size=18)
        self.multi_cell(w=0,h=10, txt=txt, border=0,
                  align='C', fill=False)



    #添加图表
    def charts(self):
        self.set_xy(11, 30)
        self.image('./img/user_time_online_test.png', w=5550 / 28, h=2700 / 28,link='file:///E:/Rfirefox/down/floral.png')

if __name__ == '__main__':
    pdf = PDF()
    pdf.add_page()
    pdf.lines()
    pdf.titles()
    pdf.imagex()
    pdf.add_font('changan', '', './source/樱落长安楷体.ttf', uni=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("changan", size=18)

    # out.txt文本内容
    with open("./out.txt", 'r', encoding='utf-8') as f:  # 打开文件
        data = f.read()
    pdf.set_xy(10.0, 25)
    pdf.multi_cell(w=0, h=10, txt=data, border=0,
                   align='C', fill=False)

    # 新增一页
    pdf.add_page()
    pdf.lines()
    pdf.imagex()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("changan", size=18)
    # 群内发言分词产生的图云
    pdf.set_xy(10.0, 25)
    txt = 'Chat keyword word cloud'
    pdf.multi_cell(w=0, h=10, txt=txt, border=0,
                   align='C', fill=False)
    pdf.image('./img/all_wordcloud0.png', x=8, y=45, w=5120 / 28, h=3840 / 28,
              link='file:///E:/Rfirefox/down/floral.png')

    pdf.image('./img/longest_formation_wordcloud.png', x=8, y=150, w=5120 / 28.5, h=3840 / 28.5,
              link='file:///E:/Rfirefox/down/floral.png')

    # 新增一页
    pdf.add_page()
    pdf.lines()
    pdf.imagex()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("changan", size=12)
    # qq群总体聊天时间分布图
    pdf.set_xy(10.0, 25)
    txt = 'Group chat time distribution'
    pdf.multi_cell(w=0, h=10, txt=txt, border=0,
                   align='C', fill=False)

    pdf.set_xy(8, 35)
    pdf.image('./img/user_time_online_test.png', w=5550 / 28.5, h=2700 / 28.5,
              link='file:///E:/Rfirefox/down/floral.png')

    # qq群聊天排名
    pdf.set_xy(10.0, 150)
    txt = 'Speaking ranking and picture ratio'
    pdf.multi_cell(w=0, h=10, txt=txt, border=0,
                   align='C', fill=False)
    pdf.image('./img/speak_photo_in_total.png', x=8, y=180, w=5550 / 28.5, h=2700 / 28.5,
              link='file:///E:/Rfirefox/down/floral.png')

    pdf.set_author('Cracer')

    pdf.output('test.pdf', 'F')