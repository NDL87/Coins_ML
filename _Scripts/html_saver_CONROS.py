import requests
from urllib.request import urlopen
import os
from bs4 import BeautifulSoup


def choose_folder(c, html_dir):
    global chapter_dir
    if c == 1:
        chapter_dir = '1_Rus_Emp_AgAu\\'
    if c == 2:
        chapter_dir = '2_Rus_Emp_Cu\\'
    if c == 3:
        chapter_dir = '3_Rus_before_Peter\\'
    if c == 4:
        chapter_dir = '4_Antique_MiddleAges\\'
    if c == 6:
        chapter_dir = '6_USSR_RF\\'
    if c == 7:
        chapter_dir = '7_Foreign\\'
    if not os.path.exists(html_dir + chapter_dir):
        os.makedirs(html_dir + chapter_dir)


def htmls_conros(auction_old, auction_new, chapter_list, html_dir):

    for auction in range(auction_old + 1, auction_new, 1):
        print(auction)
        for c in chapter_list:
            i = 0
            page_check = requests.get(
                "https://auction.conros.ru/clAuct/" + str(auction) + '/' + str(c) + '/' + str(i) + '/0/asc/')
            while page_check.status_code == 200:
                page_check = requests.get(
                    "https://auction.conros.ru/clAuct/" + str(auction) + '/' + str(c) + '/' + str(i) + '/0/asc/')
                page = urlopen(
                    "https://auction.conros.ru/clAuct/" + str(auction) + '/' + str(c) + '/' + str(i) + '/0/asc/')
                page_content = page.read()

                soup = BeautifulSoup(page_content, 'html.parser')
                kop20 = soup.findAll('td', class_="productListing-data")

                if len(kop20) > 50:
                    print(auction, c, i, len(kop20))

                    choose_folder(c, html_dir)

                    with open(html_dir + chapter_dir + str(auction) + '_' + str(c) + '_' + str(i) + ".html", "wb") as f:
                        f.write(page_content)
                    f.close()
                    i += 1
                else:
                    break
