import re
from urllib import request, parse
import bs4

from src.Spider import UAPool

TIEBA_HOME_URL = 'https://tieba.baidu.com/'
THREADS_PATH = '../htmls/threads/'


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
        pass
