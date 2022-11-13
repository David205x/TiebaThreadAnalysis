import re
from urllib import request, parse
import bs4
import jieba

from src.Spider import UAPool

TIEBA_BASE_URL = 'https://tieba.baidu.com/f?ie=utf-8&kw='
THREADS_PER_PAGE = 50


class Spider(object):

    def __init__(self, url):

        self.quoted_url = parse.quote(url, 'utf-8')             # 经过转码的贴吧名

        if url is None:
            self.target_url = TIEBA_BASE_URL + 'java'
        else:
            self.target_url = TIEBA_BASE_URL + self.quoted_url

        self.pages = []                                         # 按页存储的帖子标题与一楼
        self.pages_list = []                                    # 待爬取页数编号
        self.target_tieba = url                                 # 目标贴吧

        self.soup_page = None                                   # 经过soup读取的页面
        self.result = None                                      # 未使用的soup爬取结果
        self.conditions = []                                    # 特定爬取条件及过滤器

    # 向特定贴吧发起一次请求，读取一整夜的内容
    def retrieve_page(self, pages):
        header = {'User-Agent': UAPool.ua_gen()}

        print(header)
        print(self.target_url)

        self.pages_list = pages

        for p in pages:
            url = self.target_url + '&pn=' + str((p - 1) * THREADS_PER_PAGE)  # pn参数的值应为50*(page-1)
            req = request.Request(url=url, headers=header)
            response = request.urlopen(req, timeout=1000)
            self.pages.append(response.read().decode("utf-8"))

            # self.soup_page = bs4.BeautifulSoup(self.pages, 'html.parser')
            # self.soup_page.prettify()

            # print(self.soup_page.prettify())
            # print(self.soup_page.get_text())
            # print(self.soup_page.p)
            # print(self.soup_page.find_all('li'))

    # 保存一整页为.html
    def save_page(self, path, pages):

        cntr = 0
        for p in pages:
            filename = path + 'pg' + str(p) + '_' + self.target_tieba + '.html'
            filepath = path
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.pages[cntr])
                print(f'{filename} saved at {filepath}')
            cntr += 1

    # def load_page(self, path, page):
    #     f = open(path, 'r')
    #     if len(self.pages) == 0:
    #         self.pages.append(f.read())
    #     print(self.pages)

    # def show_page(self):
    #     print(self.soup_page)
    #
    # def show_result(self):
    #     print(self.result)d

    # 获取全部帖子的标题与预览内容，并返回按页的帖子list
    def get_threads(self):

        threads_by_page = []

        for i in range(len(self.pages_list)):

            # 提取置顶帖
            top_regex = r'<i class="icon-top" alt="置顶" title="(.*?)"></i>'
            t_pattern = re.compile(top_regex, re.S)
            top_result = t_pattern.findall(self.pages[i])
            top_thread_count = len(top_result)

            # print(top_thread_count)

            # 定位标题
            title_regex = r'<a rel="noopener" href="(/p/.*?)" title="(.*?)" target="_blank" class="j_th_tit ">.*?</a>'
            t_pattern = re.compile(title_regex, re.S)
            title_result = t_pattern.findall(self.pages[i])
            processed_title_result = []
            for r in title_result[-50:]:
                title_href_pair = (r[0].strip(), r[1].strip())
                processed_title_result.append(title_href_pair)

            # for r in processed_title_result:
            #     print(f'{r}')

            # 提取一楼内容
            overview_regex = r'<div class="threadlist_abs.*?">(.*?)</div>'
            o_pattern = re.compile(overview_regex, re.S)
            overview_result = o_pattern.findall(self.pages[i])
            processed_overview_result = []
            for r in overview_result:
                processed_overview_result.append(r.strip())

            # for r in processed_overview_result:
            #     print(f'{r}')

            thread_items = []
            for j in range(len(processed_title_result)):
                if j < top_thread_count:
                    thread_items.append([processed_title_result[j], 'None']) # 置顶帖
                else: # 其余帖子
                    thread_items.append([processed_title_result[j], processed_overview_result[j - top_thread_count]])

            threads_by_page.append(thread_items)

        # current_pg = 1
        # for pg in threads_by_page:
        #
        #     print('*' * 40)
        #     print(f'current page: {current_pg}')
        #     print('*' * 40)
        #
        #     for thd in pg:
        #         print(f'{thd[0]}:\n{thd[1]}')
        #     current_pg += 1

        return threads_by_page
