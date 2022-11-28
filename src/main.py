import math
import os
import sys
import time
import argparse

from Spider.Spider import Spider
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
from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import xlwt
import xlrd

from matplotlib import mlab

from matplotlib import rcParams

from src.Spider.TiebaThread import TiebaThread

HTMLS_PATH = '../htmls/'
TIEBA_JIEBTA_DICT = '../dict/tieba-dict.txt'

TEST_HTML_PATH = '../htmls/threads/pg1_北京工业大学.html'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
STOP_WORDS_FILE_PATH = '停用词.txt'

#找到中文
def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')   # 去除非中文字符
    chinese = re.sub(pattern, '', file)
    return chinese


def read_txt():
    f = open("TiebaData.txt",'rt',encoding='utf-8')
    txt_list = f.readlines()
    txt_list = [x.strip() for x in txt_list]    #x.strip()去除回车等符号
    txt_list = [x for x in txt_list if len(x)>0]
    txt_list = [x for x in txt_list if x != "楼主："]
    txt_list = [x for x in txt_list if x != "评论："]
    txt_list = [find_chinese(x) for x in txt_list if x != "评论："]
    txt_list = [x for x in txt_list if len(x)>0]

    return txt_list

#读取txt,转成列表
def read_txt_similarity():
    f = open("TiebaData.txt", 'rt', encoding='utf-8')
    txt_list = f.readlines()
    txt_list = [x.strip() for x in txt_list]
    txt_list = [x for x in txt_list if len(x)>0]
    txt_list = [find_chinese(x) for x in txt_list]
    txt_list = [x for x in txt_list if len(x)>0]

    return txt_list

#列表转字典，帖子作为关键字，评论作为键值
def list_dict(txt_list):
    index_=[]
    dict_data={}
    for i in range(len(txt_list)):
        if txt_list[i] == "楼主":
            index_.append(i)
    for x in range(1, len(index_)-1):
        a=(index_[x])      # 贴子a为关键字
        b=(index_[x-1])    # 评论b为键值
        dantiao=txt_list[b:a]
        dict_data[dantiao[1]]=dantiao[3:]
    return dict_data

def similarities(keyword, texts):
    req_list =[]
    # 分词
    texts = [lcut(text) for text in texts]
    print(texts)
    # 基于文本集建立词典，并获得词典特征数
    dictionary = Dictionary(texts)
    #(dictionary)
    num_features = len(dictionary.token2id)  # 词典特征数
    # 基于词典，将分词列表集转换成稀疏向量集，称作语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print(corpus)
    # 同理，把基础文本也要转成向量集
    keyword_vector = dictionary.doc2bow(lcut(keyword))
    print(lcut(keyword))  # doc2bow()函数对单词进行ID分配，(ID,有多少个)
    # 创建TF-IDF模型，训练语料库
    tfidf = TfidfModel(corpus)
    # 用训练好的模型处理比对文本和基础文本
    tf_texts = tfidf[corpus]
    tf_keyword = tfidf[keyword_vector]
    # 相似度计算
    sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
    similarities = sparse_matrix.get_similarities(tf_keyword)
    # 输出相似度
    for e, s in enumerate(similarities, 1):
        req_list.append(s)
    return req_list

def print_similarities():
    # 贴子和评论间的相似度分析 以贴子为基础本文
    txt_list_similarity = read_txt_similarity()
    dict_data = list_dict(txt_list_similarity)
    print(dict_data)
    res_list = []
    for key in dict_data.keys():
        tiezi_list = len(dict_data[key]) * [key]
        pinglun_list = dict_data[key]
        req_list = similarities(key, dict_data[key])
        res_list.extend(list(zip(tiezi_list, pinglun_list, req_list)))

    df = pd.DataFrame(res_list)
    df.columns = ['帖子', '评论', '相关度']
    df.to_csv("帖子评论相关度分析.csv", encoding="utf_8", index=False)

    # 一个贴子下 评论和评论间的相似度分析 以评论第一条为基础本文
    res1_list = []
    for key in dict_data.keys():
        if len(dict_data[key]) >= 2:
            keyword = dict_data[key][0]
            texts = dict_data[key][1:]
            tiezi = len(texts) * [key]
            jichu = len(texts) * [keyword]
            req_list = similarities(key, dict_data[key])
            res1_list.extend(list(zip(tiezi, jichu, texts, req_list)))
    df = pd.DataFrame(res1_list)
    df.columns = ['帖子', '基础语句', "评论", '相关度']
    df.to_csv("评论与评论相关度分析.csv", encoding="utf_8", index=False)

    print("相似度已输出至.csv文件")


def draw_cloud():
    plt.clf()
    txt_list = read_txt()
    # 首先通过停用词和分词技术得到词云的关键词
    t = "".join(txt_list)  #空格拼接
    jieba.analyse.set_stop_words(STOP_WORDS_FILE_PATH)     # 使用自己的停用词.txt对爬取数据进行分词
    keywords_count_list = jieba.analyse.textrank(t, topK=100, withWeight=True)  # 参数1 待处理语句 参数2 关键字的个数 参数3 返回每个关键字的权重值
    print(keywords_count_list)
    image = Image.open("base.jpg")  # 选取贴吧标志作为背景轮廓图
    graph = np.array(image)
    color_list=['#DC143C', '#00FF7F', '#FF6347', '#8B008B', '#00FFFF', '#0000FF', '#8B0000', '#FF8C00',
            '#1E90FF', '#00FF00', '#FFD700', '#008080', '#008B8B', '#8A2BE2', '#228B22', '#FA8072', '#808080']
    colormap=colors.ListedColormap(color_list)

    # simkai.ttf楷体 # 设置有多少种随机生成状态，即有多少种配色方案
    # wc = WordCloud(font_path='simkai.ttf', background_color='white', max_words=100, colormap=colormap, random_state=30, mask=graph)
    wc = WordCloud(font_path='simhei.ttf', background_color='white', max_words=100, colormap=colormap, random_state=30, mask=graph)
    # wc = WordCloud(font_path='simkai.ttf', background_color='white', max_words=100, random_state = 30,mask=graph)

    frequencies_dic={}
    for key_word in keywords_count_list:
        frequencies_dic[key_word[0]] = key_word[1]   # key_word[0]:关键字名 key_word[1]:权重
    print(frequencies_dic)

    wc.generate_from_frequencies(frequencies_dic)  # 根据给定词频生成词云
    image_color = ImageColorGenerator(graph)
#>>>>>>>>>>>>>>>>>>    plt.imshow(wc)
    plt.axis("off")  # 不显示坐标轴
    # plt.show()
    wc.to_file('词云.jpg')  # 图片命名


def emotion():
    plt.clf()
    txt_list = read_txt()
    emotion_List = []
    s_list = []
    sentiments_score = []
    txt_order = []
    for txt in txt_list:
        s = SnowNLP(txt).sentiments       # s是得出的情感分析概率
        s_list.append(s)

        if s >= 0.53:
            emotion_List.append("积极")
        elif s <= 0.47:
            emotion_List.append("消极")
        else:
            emotion_List.append("中性")
    df = pd.DataFrame()      # 每个df为一列 下面显示所获得数据
    df['文本'] = txt_list
    df['情感指数'] = s_list
    df['情感'] = emotion_List
    df.to_csv("感.csv", index=False)
    x = ['积极', '中性', '消极']
    y = []
    for em in x:
        y.append(emotion_List.count(em))
    d_list = [i/sum(y)*100 for i in y]    # autopct='%1.2f%%' 保留小数点后两位
    plt.pie(d_list, explode=None, labels=x, autopct='%1.2f%%', startangle=200, counterclock=False)
    plt.title("情感占比分布")
    plt.savefig("情感占比分布.jpg")
    # plt.show()
    for i in txt_list:
        txt_order.append(i)
    plt.clf()
    table = pd.DataFrame(txt_order, s_list)
    plt.plot(s_list, linestyle='-')
    plt.title("情感波动图")
    plt.savefig("情感波动图.jpg")
    # plt.show()


def save_replies_excel(title, time, content, book, cnt):

    sheet = book.add_sheet(str(cnt), cell_overwrite_ok=True)
    col = ('回复内容', '回复时间')
    for i in range(0, 2):
        sheet.write(0, i, col[i])

    for i in range(len(content)):
        data = [content[i], time[i]]
        for j in range(0, 2):
            sheet.write(i + 1, j, data[j])
    sheet.write(len(content) + 1, 0, title)
    savepath = '../excel/ex.xls'
    book.save(savepath)


def read_excel(path):
    # 导入需要读取Excel表格的路径
    try:

        data = xlrd.open_workbook(path)
        all_time = []
        all_title = []
        all_replies = []
        for i in range(len(data.sheets())):
            table = data.sheets()[i]
            all_title.append(table.cell_value(table.nrows - 1, 0))
            replies = []
            time = []
            for rown in range(1, table.nrows - 1):
                replies.append(table.cell_value(rown, 0))
                time.append(table.cell_value(rown, 1))
            all_time.append(time)
            all_replies.append(replies)
    except Exception as e:
        print(e)

    return all_time, all_title, all_replies


def analyze_post_time():
    plt.clf()
    all_time, all_title, all_replies = read_excel("../excel/ex.xls")
    all_post_time = []
    all_time_len = len(all_time)
    for i in range(all_time_len):
        all_time_i_len = len(all_time[i])
        for j in range(all_time_i_len):
            all_post_time.append([all_time[i][j], j])

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题

    all_day_an = [
        [
            int(time.mktime(time.strptime(item[0][0: 10], "%Y-%m-%d"))),
            int(item[0][11: 13]) * 3600 + int(item[0][14:16]) * 60,
            item[1]
        ]
        for item in all_post_time
    ]
    # sorted(all_day_an)
    mn_v = all_day_an[0][1]
    mx_v = all_day_an[0][1]
    mn_v_0 = all_day_an[0][0]
    mx_v_0 = all_day_an[0][0]
    all_day_len = len(all_day_an)
    for i in range(all_day_len):
        mn_v = min(mn_v, all_day_an[i][1])
        mx_v = max(mx_v, all_day_an[i][1])
        mn_v_0 = min(mn_v_0, all_day_an[i][0])
        mx_v_0 = max(mx_v_0, all_day_an[i][0])

    # arr_std = np.std([item[1] for item in all_day_an], ddof=1)
    # arr_std_0 = np.std([item[0] for item in all_day_an], ddof=1)

    sq = math.sqrt(all_day_len)
    for i in range(all_day_len):
        all_day_an[i][1] = int((all_day_an[i][1] - mn_v))
        all_day_an[i][0] = int((all_day_an[i][0] - mn_v_0))
        if all_day_an[i][2] == 0:
            s1 = plt.scatter(all_day_an[i][0], all_day_an[i][1], s=35, color='b', marker='o', )
        else:
            s2 = plt.scatter(all_day_an[i][0], all_day_an[i][1], s=15, color='r', marker='o', )

    """设置5-6个位点"""
    interval_x = (mx_v_0 - mn_v_0) / 5
    old_x = [(i * interval_x) for i in range(6)]
    new_x = [time.strftime("%Y-%m-%d", time.localtime(mn_v_0 + i * interval_x)) for i in range(6)]
    interval_y = (mx_v - mn_v) / 4

    old_y = [(i * interval_y) for i in range(5)]
    new_y = [time.strftime("%H:%M", time.localtime(mn_v + i * interval_y)) for i in range(5)]

    plt.xticks(
         old_x,
         new_x,
         rotation=30)
    plt.yticks(old_y, new_y, rotation=-30)

    plt.legend((s1, s2), ('楼主', '回复'), loc='best')
    plt.ylabel('具体时间')
    plt.xlabel('日期')
    plt.title('回复日期与具体时间散点图')
    plt.savefig("回复日期与具体时间散点图.jpg")


def analyze_reply_count():
    plt.clf()
    all_time, all_title, all_replies = read_excel("../excel/ex.xls")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
    x_list = [i for i in range(3, 3 + len(all_replies))]

    all_title_an = [item[0: 3] for item in all_title]
    all_replies_an = [len(item) for item in all_replies]

    rects = plt.bar(x=x_list, height=all_replies_an, width=0.6, align="center", yerr=0.001)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + 0.1, height + 0.2, int(height))

    plt.xticks(x_list, all_title_an, rotation=45)
    plt.ylabel('回复数')
    plt.xlabel('帖子标题')
    plt.title('帖子回复数统计')
    plt.savefig("帖子回复数统计.jpg")


def spider_init(tieba):
    tieba_name = '北京工业大学' if len(tieba) == 0 else tieba

    # 手动为词典增加贴吧相关用词 提升切割的精准度
    jieba.load_userdict(TIEBA_JIEBTA_DICT)
    jieba.add_word(tieba_name + '吧')

    spider = Spider(tieba_name)

    # 待爬取页数
    target_pages = [1]

    # spider.load_page(TEST_HTML_PATH, 1)
    spider.retrieve_page(target_pages)
    spider.save_page(HTMLS_PATH, target_pages)

    # 按帖子获取全部内容
    threads = spider.get_threads()
    print(threads)

    print('\n' + '*' * 40 + '\n')
    # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    cur_page = 0
    return threads


def decode_data(byte_data: bytes):
    """
    解码数据
    :param byte_data: 待解码数据
    :return: 解码字符串
    """
    try:
        return byte_data.decode('UTF-8')
    except UnicodeDecodeError:
        return byte_data.decode('GB18030')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=int, default=32)
    args = parser.parse_args()
    if args.op == 1:
        draw_cloud()
    if args.op == 2:
        emotion()
    if args.op == 3:
        analyze_reply_count()
    if args.op == 4:
        analyze_post_time()

