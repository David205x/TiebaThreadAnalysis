import os
import time

from Spider.Spider import Spider
from src.Spider.TiebaThread import TiebaThread
import jieba.analyse

HTMLS_PATH = '../htmls/'
TIEBA_JIEBTA_DICT = '../dict/tieba-dict.txt'

TEST_HTML_PATH = '../htmls/pg1_北京工业大学.html'

if __name__ == '__main__':

    tieba_name = '北京工业大学'

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

    print('\n' + '*' * 40 + '\n')

    cur_page = 0
    for p in threads:
        cur_page += 1
        for t in p[(2 if cur_page == 1 else 0):]:
            print('-' * 40 + '\n' + t[0][1])
            time.sleep(1)
            new_thread = TiebaThread(t[0][0], t[0][1])
            new_thread.retrieve_thread()
            new_thread.save_thread()
            rep = new_thread.get_replies()

            print(rep)
            print(new_thread.get_segments())
            print(new_thread.get_key_words())
