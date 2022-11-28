import re
import time
from urllib import request, parse
import bs4
import jieba.analyse

from src.Spider import UAPool

TIEBA_HOME_URL = 'https://tieba.baidu.com'  # TODO: change this later
THREADS_PATH = '../htmls/threads/'


# 处理特殊字符与特殊div
def post_process(r):
    # <br> -> \n
    # &lt; -> <
    # &gt; -> >
    # <img class...emoticon(.*>)> -> [emoticon_{group[1]}]
    # <img class="BDE_Image"...)> -> [Image]
    # <a href="(.*?)".*>(.*?)</a> -> {group[2]}#{group[1]}

    # 替换lesserthan greaterthan和换行符
    r = r.replace('<br>', '\n')
    r = r.replace('&lt;', '<')
    r = r.replace('&gt;', '>')

    # emoticon_regex = r'<img class="BDE_Smiley".*?src=".*?image_emoticon(.*?).png" >'
    # e_pattern = re.compile(emoticon_regex, re.S)
    # emoticons = e_pattern.findall(r)
    #
    # emoticon_replacement_regex = r'<img class="BDE_Smiley".*?>'
    # for i in range(len(emoticons)):
    #     r = re.sub(emoticon_replacement_regex, '[e' + emoticons[i] + ']', r, 1)

    # 替换表情图片和超链接
    emoticon_regex = r'<img class="BDE_Smiley".*?>'
    r = re.sub(emoticon_regex, '[Emoji]', r)

    image_regex = r'<img class="BDE_Image".*?>'
    r = re.sub(image_regex, '[Image]', r)

    hypierlink_regex = '<a href=".*?".*?>.*?</a>'
    r = re.sub(hypierlink_regex, '[Hyperlink]', r)

    shared_regex = r'.*<p\s.*\s*<img.*>.*</p>'
    r = re.sub(shared_regex, '[SharedHyperlink]', r)

    return r


# 使用jieba将帖子的一条回复切割成关键词数组segments
def get_reply_segments(reply):
    segments = jieba.cut(reply, cut_all=False, HMM=True, use_paddle=False)
    return segments


class TiebaThread(object):

    def __init__(self, pid, title):

        self.pid_str = pid  # /p/[pid]
        self.pid = self.pid_str.split('/')[-1]  # 提取出的pid数字
        self.title = title  # 帖子标题
        self.target_url = TIEBA_HOME_URL + pid  # 直接访问用的链接
        self.thread_content = None  # 帖子一楼内容
        self.replies = []  # 帖子回复内容

        self.word_segs = []  # 帖子所有回复的关键词集合

        self.word_freq = {}  # 词频统计

    # 向特定帖子的url发起一次请求并将其内容存储为字符串
    def retrieve_thread(self):
        header = {'User-Agent': UAPool.ua_gen()}
        # print(header)
        url = self.target_url

        print(url)

        req = request.Request(url=url, headers=header)
        response = request.urlopen(req, timeout=1000)
        # print(response.read().decode("utf-8"))
        self.thread_content = response.read().decode("utf-8")

    # 将所有帖子的时间数据写入表格
    def save_message_time(self):
        item = re.findall(r"date[\s\S]*", self.thread_content)
        all_message_time = []
        iter = re.findall(r'\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}', item[0])
        for j in iter:
            all_message_time.append(j)

        return all_message_time

    # 将特定帖子保存为.html文件用于后续分析
    def save_thread(self):
        filename = THREADS_PATH + self.pid + '.html'
        filepath = THREADS_PATH
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.thread_content)
            print(f'{filename} saved at {filepath}')

    # 获取特定帖子下的全部回复并返回
    def get_replies(self):
        # 回复时间在这里 l_post j_l_post l_post_bright
        # 提取回复 此时的回复可能仍含有html代码块
        reply_regex = r'<div id="post_content_(.*?)" class="d_post_content j_d_post_content(.*?)" style="display:;">(.*?)</div>'
        r_pattern = re.compile(reply_regex, re.S)
        replies_result = r_pattern.findall(self.thread_content)
        processed_replies_result = []

        for r in replies_result:

            content_string = r[2]

            # 提取带背景回复中的内容
            if '"post_bubble_top"' in r[2]:
                index_start = self.thread_content.find('<div class="post_bubble_top"')
                index_end = self.thread_content.find('<div class="post_bubble_bottom"')
                sub_area = self.thread_content[index_start:index_end]

                bubble_regex = r'<div class="post_bubble_top".*?div class="post_bubble_middle_inner">(.*?)</div>.*?'
                bf_pattern = re.compile(bubble_regex, re.S)
                bubbles = bf_pattern.findall(sub_area)
                if len(bubbles) != 0:
                    print(f'bubbled: {bubbles}')
                    content_string = bubbles[0]

            reply_body = post_process(content_string).strip()

            self.replies.append(reply_body)
            processed_replies_result.append(reply_body)

            self.update_segments(get_reply_segments(reply_body))

        return processed_replies_result

    # 将一个回复的关键词集合添加到帖子的关键词集合中
    def update_segments(self, segments):
        self.word_segs.append(segments)

    # 获取帖子的全部关键词集合
    def get_segments(self):
        return self.word_segs

    # 提取帖子的全部关键词并返回 默认topK = 20且不带权重
    def get_key_words(self):
        full_string = ''
        for r in self.replies:
            full_string += r
            full_string += ' '
        tags = jieba.analyse.extract_tags(full_string, topK=20, withWeight=False)
        return tags
