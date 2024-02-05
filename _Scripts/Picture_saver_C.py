import requests
from urllib.request import urlopen
import re
import os
from bs4 import BeautifulSoup
from collections import defaultdict

auction_2 = 790
auction_1 = 762

#place = 'D:\\PYTHON\\Conros_parser\\'
place = 'D:\\_PROJECTS\\Coins_Classification\\'
image_dir = place + 'Images\\'

nominals_AgAuPt = [r'15 рублей',
                   r'12 рублей',
                   r'10 рублей',
                   r'7руб.50 коп.',
                   r'6 рублей',
                   r'\b5 рублей',
                   r'3 рубля',
                   r'1,5 рубля',
                   r'Рубль',
                   r'3/4 рубля.',
                   r'Полтина',
                   r'Полуполтинник|25 копеек',
                   r'30 копеек',
                   r'20 копеек',
                   r'15 копеек',
                   r'10 копеек|Гривна|Гривенник',
                   r'\b5 копеек',
                   r'2 марки',
                   r'1 марка',
                   r'50 пенни',
                   r'25 пенни']

nominals_Cu = [r'10 копеек|Гривна|Гривенник',
               r'\b5 копеек',
               r'4 копейки',
               r'3 копейки',
               r'2 копейки',
               r'Копейка',
               r'Денга|Деньга|Денежка|1/2 копейки',
               r'Полушка|1/4 копейки',
               r'10 пенни',
               r'5 пенни',
               r'1 пенни']

nominals_AgAuPt = [r'Рубль']
#nominals_Cu = [r'Полушка']
#period = [1895, 1917]
#period = [1818, 1830]
period = [1700, 1799]
period = [1700, 1999]
max_price = 20000000

results = defaultdict(list)

def get_week():
    conros_weeks = {}
    with open('G:\_PYTHON_learn\CONROS\Conros_Weeks.dat', 'r') as w:
        for a in w:
            conros_weeks[int(a.split()[0])] = a.split()[1]
    return conros_weeks

#conros_weeks = get_week()
#print(conros_weeks)

def check_letter(nominal_local, letter, material):
    if 'Рус.-Пол.' in letter:
        if nominal_local == '3рубля':
            nominal_local = '3рубля-20zlotych'
        if nominal_local == '1,5рубля':
            nominal_local = '1.5рубля-10zlotych'
        if nominal_local == '3/4рубля-':
            nominal_local = '0.75рубля-5zlot'
        if nominal_local == '25копеек':
            nominal_local = '25коп-50groszy'
        if nominal_local == '30копеек':
            nominal_local = '30коп-2zlote'
        if nominal_local == '15копеек':
            nominal_local = '15коп-1zloty'
        directory = str(image_dir + 'Русско-польские\\' + nominal_local + '\\')
    elif 'Pt' in material:
        directory = str(image_dir + 'Платина\\' + nominal_local + '\\')
    elif '"Сибирь"' in letter:
        nominal_local = nominal_local + '(Сиб)'
        directory = str(image_dir + 'Сибирь\\' + nominal_local + '\\')
    elif 'Валахия' in letter:
        nominal_local = nominal_local + '(Молд)'
        directory = str(image_dir + 'Молдавия\\' + nominal_local + '\\')
    elif 'Финляндия' in letter:
        nominal_local = nominal_local
        directory = str(image_dir + 'Финляндия\\' + nominal_local + '\\')
    else:
        if nominal_local == 'Рубль':
            if '300-летие' in letter:
                nominal_local = 'Рубль300лет'
            if 'колонны' in letter:
                nominal_local = 'РубльАлексКол'
            if 'памятника Николаю' in letter:
                nominal_local = 'РубльПамНикI'
            if 'памятника Александру' in letter:
                nominal_local = 'РубльПамАлII'
            if 'Бородинского памятника' in letter:
                nominal_local = 'РубльБорПам'
            if 'коронацию Александра' in letter:
                nominal_local = 'РубльКорАлIII'
            if 'коронацию Николая' in letter:
                nominal_local = 'РубльКорНикII'
            if 'войны 1812' in letter:
                nominal_local = 'Рубль1812'
        directory = str(image_dir + nominal_local + '\\')
    return nominal_local, directory


def multi_nominal(nominal_local):
    if nominal_local == 'Полуполтинник':
        return '25копеек'
    elif nominal_local == 'Гривенник' or nominal_local == 'Гривна':
        return '10копеек'
    elif nominal_local == '1/2копейки' or nominal_local == 'Денежка':
        return 'Денга'
    elif nominal_local == '1/4копейки':
        return 'Полушка'
    elif nominal_local == '7руб.50коп.':
        return '7.5рублей'
    else:
        return nominal_local


def img_save(auction, local_id, nominal, year, directory):
    global saved
    coin = requests.get("https://auction.conros.ru/img/" + str(auction) + '/' + str(local_id) + '.jpg')
    coin = coin.content
    image_name = directory + str(nominal) + '_' + str(year) + '_' + str(auction) + '_' + str(local_id) + "_1.png"
    if not os.path.exists(image_name):
        with open(image_name, "wb") as f:
            f.write(coin)
        f.close()
        saved += 1
    coin = requests.get("https://auction.conros.ru/img/" + str(auction) + '/' + str(local_id) + '+.jpg')
    coin = coin.content
    image_name = directory + str(nominal) + '_' + str(year) + '_' + str(auction) + '_' + str(local_id) + "_2.png"
    if not os.path.exists(image_name):
        with open(image_name, "wb") as f:
            f.write(coin)
        f.close()
        saved += 1

def get_date(lot_id):
    page = urlopen("https://auction.conros.ru/" + str(lot_id))
    page_content = page.read()
    soup2 = BeautifulSoup(page_content, 'html.parser')
    rub20 = soup2.findAll('p', class_="lot_info_box")
    date = rub20[-1].text[-10:]
    return date

def parser(auction, html_dir, html_id, nominal):
    for i in range(0, 40, 1):
        try:
            html = open(html_dir + str(auction) + '_' + str(html_id) + '_' + str(i) + '.html').read()
            soup = BeautifulSoup(html, 'html.parser')
            kop20 = soup.findAll('td', class_="productListing-data", onclick=True)
            for a in range(len(kop20)):
                if re.search(nominal, str(kop20[a])) is not None:
                    if nominal == r'2 копейки' or nominal == r'4 копейки':
                        if '1/' in kop20[a].text:
                            break
                    global nominal_local
                    nominal_local = re.search(nominal, str(kop20[a])).group()
                    nominal_local = nominal_local.replace(' ', '')
                    nominal_local = multi_nominal(nominal_local)
                    local_id = kop20[a - 1].text
                    year = kop20[a + 1].text
                    if len(year) != 4 or '?' in str(year) or ' ' in str(year)or 'в' in str(year):
                        print(year)
                        break
                    letter = kop20[a + 2].text
                    material = kop20[a + 3].text
                    condition = kop20[a + 4].text
                    price = kop20[a + 7].text
                    nominal_local, directory = check_letter(nominal_local, letter, material)
                    lot_id = kop20[a]['onclick'].split('=')[1][1:-2]
                    date = 'TBD'
                    #date = get_date(lot_id)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    print(auction, ' ', local_id, ' ', nominal_local, ' ', year, ' ', letter, ' ',
                          condition, ' ', price, ' ', date)
                    #if period[0] <= int(year) <= period[1] and int(price) < max_price:
                    #    #results[condition].append(int(auction), int(price))
                    #    aaa = str(conros_weeks[auction]) + ' ' + str(price) + '\n'
                    #    results[condition].append(aaa)
                    #    aaa = str()
                    try:
                        img_save(auction, local_id, nominal_local, year, directory)
                    except:
                        pass
        except:
            break

saved = 0
for auction in range(auction_2, auction_1 - 1, -1):

    for nominal in nominals_AgAuPt:
        parser(auction, place + 'HTMLs\\1_Rus_Emp_AgAu\\', 1, nominal)

    #for nominal in nominals_Cu:
    #    parser(auction, place + 'HTMLs\\2_Rus_Emp_Cu\\', 2, nominal)

print('\n', saved, 'images were saved')
print()

'''
with open('Stat_' + nominal_local + '_' + str(period[0]) + '-' + str(period[1]) + '_last_' + str(auction_2-auction_1) + '_auctions.txt', 'w') as f:
    for key, value in results.items():
        print(key, '\n', ' '.join(("{}".format(links)) for links in value))
        f.write(str(key + '\n' + ''.join(("{}".format(links)) for links in value) + '\n'))
        #print('AVERAGE:', round(sum(value)/len(value)), '\n')
        #f.write(str('AVERAGE: ' + str(round(sum(value)/len(value))) + '\n\n'))
f.close()
'''