import re
from urllib import request, parse
import bs4

from src.Spider import UAPool

TIEBA_HOME_URL = 'https://tieba.baidu.com/'
THREADS_PATH = '../htmls/threads/'


def post_process(r):

    # remove unwanted tags and divs here
    # <br> -> \n
    # &lt; -> <
    # &gt; -> >
    # <img class...emoticon(.*>)> -> [emoticon_{group[1]}]
    # <img class="BDE_Image"...)> -> [Image]
    # <a href="(.*?)".*>(.*?)</a> -> {group[2]}#{group[1]}
    r = r.replace('<br>', '\n')
    r = r.replace('&lt;', '<')
    r = r.replace('&gt;', '>')

    emoticon_regex = r'<img class="BDE_Smiley".*?src=".*?image_emoticon(.*?).png" >'
    e_pattern = re.compile(emoticon_regex, re.S)
    emoticons = e_pattern.findall(r)

    emoticon_replacement_regex = r'<img class="BDE_Smiley".*?>'
    for i in range(len(emoticons)):
        r = re.sub(emoticon_replacement_regex, '[e' + emoticons[i] + ']', r, 1)

    image_replacement_regex = r'<img class="BDE_Image".*?>'
    r = re.sub(image_replacement_regex, '[Image]', r)
    return r


class TiebaThread(object):

    def __init__(self, pid, title):
        self.pid_str = pid
        self.pid = self.pid_str.split('/')[-1]
        self.title = title
        self.target_url = TIEBA_HOME_URL + pid
        self.thread_content = None

    def retrieve_thread(self):
        header = {'User-Agent': UAPool.ua_gen()}

        url = self.target_url
        req = request.Request(url=url, headers=header)
        response = request.urlopen(req, timeout=1000)
        self.thread_content = response.read().decode("utf-8")

    def save_thread(self):
        filename = THREADS_PATH + self.pid + '.html'
        filepath = THREADS_PATH
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.thread_content)
            print(f'{filename} saved at {filepath}')

    def get_replies(self):

        reply_regex = r'<div id="post_content_(.*?)" class="d_post_content j_d_post_content  clearfix" style="display:;">(.*?)</div>'
        t_pattern = re.compile(reply_regex, re.S)
        replies_result = t_pattern.findall(self.thread_content)
        processed_replies_result = []
        for r in replies_result:
            reply_item = (r[0].strip(), post_process(r[1]).strip())
            processed_replies_result.append(reply_item)

        return processed_replies_result
