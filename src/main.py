import os
import time

from Spider.Spider import Spider
from src.Spider.TiebaThread import TiebaThread
import jieba.analyse
import numpy as np
import re
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors
import jieba.analyse
import pandas as pd
from snownlp import SnowNLP

HTMLS_PATH = '../htmls/'
TIEBA_JIEBTA_DICT = '../dict/tieba-dict.txt'

TEST_HTML_PATH = '../htmls/threads/pg1_北京工业大学.html'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
STOP_WORDS_FILE_PATH = '停用词.txt'

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def read_txt():
    f = open("TiebaData.txt",'rt',encoding='utf-8')
    txt_list = f.readlines()
    txt_list = [x.strip() for x in txt_list]
    txt_list = [x for x in txt_list if len(x)>0]
    txt_list = [x for x in txt_list if x != "楼主："]
    txt_list = [x for x in txt_list if x != "评论："]
    txt_list = [find_chinese(x) for x in txt_list if x != "楼主："]
    txt_list = [find_chinese(x) for x in txt_list if x != "评论："]
    txt_list = [x for x in txt_list if len(x)>0]
    return txt_list

def draw_cloud(txt_list):

    t = "".join(txt_list)
    jieba.analyse.set_stop_words(STOP_WORDS_FILE_PATH)
    keywords_count_list = jieba.analyse.textrank(t, topK=100, withWeight=True)
    image = Image.open("base.jpg")  # 作为背景轮廓图
    graph = np.array(image)
    color_list=['#DC143C', '#00FF7F', '#FF6347', '#8B008B', '#00FFFF', '#0000FF', '#8B0000', '#FF8C00',
            '#1E90FF', '#00FF00', '#FFD700', '#008080', '#008B8B', '#8A2BE2', '#228B22', '#FA8072', '#808080']
    colormap=colors.ListedColormap(color_list)

    wc = WordCloud(font_path='simkai.ttf', background_color='white', max_words=100,colormap=colormap, random_state = 30,mask=graph)
    # wc = WordCloud(font_path='simkai.ttf', background_color='white', max_words=100, random_state = 30,mask=graph)
    frequencies_dic={}
    for key_word in keywords_count_list:
        frequencies_dic[key_word[0]] = key_word[1]
    print(frequencies_dic)
    wc.generate_from_frequencies(frequencies_dic)  # 根据给定词频生成词云
    image_color = ImageColorGenerator(graph)
    plt.imshow(wc)
    plt.axis("off")  # 不显示坐标轴
    plt.show()
    wc.to_file('词云.jpg')  # 图片命名

def emotion(txt_list):
    emotion_List = []
    s_list = []
    for txt in txt_list:
        s = SnowNLP(txt).sentiments
        s_list.append(s)
        if s >= 0.53:
            emotion_List.append("积极")
        elif s <= 0.47:
            emotion_List.append("消极")
        else:
            emotion_List.append("中性")
    df = pd.DataFrame()
    df['文本'] = txt_list
    df['情感指数'] = s_list
    df['情感'] = emotion_List
    df.to_csv("感.csv", index=False)
    x = ['积极', '中性', '消极']
    y = []
    for em in x:
        y.append(emotion_List.count(em))
    d_list = [i/sum(y)*100 for i in y]
    plt.pie(d_list,explode=None,labels=x, autopct='%1.2f%%',startangle=200,counterclock=False)
    plt.title("情感占比分布")
    plt.savefig("情感占比分布.jpg")
    plt.show()

if __name__ == '__main__':
    txt_list = read_txt()
    draw_cloud(txt_list)
    emotion(txt_list)
    # tieba_name = '北京工业大学'
    #
    # # 手动为词典增加贴吧相关用词 提升切割的精准度
    # jieba.load_userdict(TIEBA_JIEBTA_DICT)
    # jieba.add_word(tieba_name + '吧')
    #
    # spider = Spider(tieba_name)
    #
    # # 待爬取页数
    # target_pages = [1]
    #
    # # spider.load_page(TEST_HTML_PATH, 1)
    # spider.retrieve_page(target_pages)
    # spider.save_page(HTMLS_PATH, target_pages)
    #
    # # 按帖子获取全部内容
    # threads = spider.get_threads()
    # print(threads)
    #
    # print('\n' + '*' * 40 + '\n')
    #
    # cur_page = 0
    # for p in threads:
    #     cur_page += 1
    #     for t in p[(2 if cur_page == 1 else 0):]:
    #         print('-' * 40 + '\n' + t[0][1])
    #         time.sleep(1)
    #         new_thread = TiebaThread(t[0][0], t[0][1])
    #         new_thread.retrieve_thread()
    #         new_thread.save_thread()
    #         rep = new_thread.get_replies()
    #
    #         print(rep)
    #         # fileName = 'TiebaData.txt'
    #         # with open(fileName, 'a', encoding='utf-8') as file:
    #         #     for i in range(len(rep)):
    #         #         file.write(rep[i]+"\n")
    #
    #         print(new_thread.get_segments())
    #         print(new_thread.get_key_words())

