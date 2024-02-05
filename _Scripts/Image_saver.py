import requests
from urllib.request import urlopen
import re
import os
from bs4 import BeautifulSoup
from collections import defaultdict
from functions_wolmar import *


work_dir = 'C:\\UserData\\z002setj\\OneDrive - Siemens AG\\_PYTHON_learn\\'
images_dir = work_dir + 'Images\\'
input_pkl = work_dir + 'Databases\\Joined_Database.pkl'


def check_letter(image_dir, nominal_local, letter, material):
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


def img_save(id, local_id, nominal, year, directory):
    global saved
    coin = requests.get("https://auction.conros.ru/img/" + str() + '/' + str(local_id) + '.jpg')
    coin = coin.content
    image_name = directory + str(nominal) + '_' + str(year) + '_' + str() + '_' + str(local_id) + "_1.png"
    if not os.path.exists(image_name):
        with open(image_name, "wb") as f:
            f.write(coin)
        f.close()
        saved += 1
    coin = requests.get("https://auction.conros.ru/img/" + str() + '/' + str(local_id) + '+.jpg')
    coin = coin.content
    image_name = directory + str(nominal) + '_' + str(year) + '_' + str() + '_' + str(local_id) + "_2.png"
    if not os.path.exists(image_name):
        with open(image_name, "wb") as f:
            f.write(coin)
        f.close()
        saved += 1


df = pd.read_pickle(input_pkl)

for index, row in df.iterrows():
    print(row['ID'], row['Date'], row['Nominal'], row['Year'], row['Price'], row['Condition'])
