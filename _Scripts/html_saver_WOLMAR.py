import requests
from urllib.request import urlopen
import os


def htmls_wolmar(auction_old, auction_new, chapter_list, html_dir):

    chapters = ['monety-antika-srednevekove',
                'dopetrovskie-monety',
                'monety-rossii-do-1917-zoloto',
                'monety-rossii-do-1917-serebro',
                'monety-rossii-do-1917-med',
                'monety-rsfsr-sssr-rossii',
                'monety-inostrannye',
                'zoloto-platina-i-dr-do-1945-goda',
                'zoloto-platina-i-dr-posle-1945-goda',
                'serebro-i-dr-do-1800-goda',
                'serebro-i-dr-s-1800-po-1945-god',
                'serebro-i-dr-posle-1945-goda']

    cntr = 1
    for auction in range(auction_old + 1, auction_new, 1):

        for c in chapter_list:

            page_name = str("https://www.wolmar.ru/auction/" + str(auction) + '/' + str(chapters[c]) + '?all=1')
            print('Saving auction ' + str(cntr) + ' (#' + str(auction) + ') of ' + str(auction_new - auction_old))
            page_check = requests.get(page_name)

            if page_check.status_code == 200:
                page = urlopen(page_name)
                page_content = page.read()
                if len(page_content) > 20000:
                    chapter_dir = str(str(c) + '_' + chapters[c] + '/')

                    if not os.path.exists(html_dir + chapter_dir):
                        os.makedirs(html_dir + chapter_dir)

                    with open(html_dir + chapter_dir + str(auction) + ".html", "wb") as f:
                        f.write(page_content)
                    f.close()
        cntr+=1
