import requests
from urllib.request import urlopen
import os
from bs4 import BeautifulSoup
from html_saver_CONROS import *
from html_saver_WOLMAR import *


def html_max(html_dir):
    htmlfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk(html_dir)
             for name in files
             if name.endswith((".html", ".htm"))]
    htmls = set()
    for html in htmlfiles:
        htmls.add(int(html.split('\\')[-1].split('.')[0].split('_')[0]))
    print(max(htmls))
    return max(htmls)


#html_dir = 'C:\\_PYTHON_learn\\CONROS\\HTMLs\\'
#chapter_list = [1, 2, 3, 4, 6, 7]
#auction_old = html_max(html_dir)
#print('Latest found auction: {}'.format(auction_old))
#while not html_max(html_dir) == auction_old:
#    htmls_conros(auction_old, auction_old + 10, chapter_list, html_dir)

#html_dir = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\'
#chapter_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#auction_old = html_max(html_dir)
#print('Latest found auction: {}'.format(auction_old))
#while not html_max(html_dir) == auction_old:
#    htmls_wolmar(auction_old, auction_old + 10, chapter_list, html_dir)

html_dir = 'D:\\_PROJECTS\\Coins_Classification\\HTMLs\\CONROS\\'
#chapter_list = [1, 2, 3, 4, 6, 7]
chapter_list = [1]
htmls_conros(1011, 1014, chapter_list, html_dir)

html_dir = 'D:\\_PROJECTS\\Coins_Classification\\HTMLs\\WOLMAR\\'
#chapter_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
chapter_list = [3]
#htmls_wolmar(1930, 1935, chapter_list, html_dir)

folders = ['0_monety-antika-srednevekove',
           '1_dopetrovskie-monety',
           '2_monety-rossii-do-1917-zoloto',
           '3_monety-rossii-do-1917-serebro',
           '4_monety-rossii-do-1917-med',
           '5_monety-rsfsr-sssr-rossii',
           '6_monety-inostrannye',
           '7_zoloto-platina-i-dr-do-1945-goda',
           '8_zoloto-platina-i-dr-posle-1945-goda',
           '9_serebro-i-dr-do-1800-goda',
           '10_serebro-i-dr-s-1800-po-1945-god',
           '11_serebro-i-dr-posle-1945-goda']