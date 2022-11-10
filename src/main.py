import os
from Spider.Spider import Spider
from src.Spider.TiebaThread import TiebaThread

HTMLS_PATH = '../htmls/'

if __name__ == '__main__':
    spider = Spider('spring')

    target_pages = [1]

    spider.retrieve_page(target_pages)
    # spider.save_page(HTMLS_PATH, target_pages)

    threads = spider.get_threads()

    print('\n' + '*' * 40 + '\n')

    for p in threads:
        for t in p[:5]:
            new_thread = TiebaThread(t[0][0], t[0][1])
            new_thread.retrieve_thread()
            new_thread.save_thread()
            rep = new_thread.get_replies()

            print(rep)
