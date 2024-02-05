import requests
from urllib.request import urlopen
import re
import os
from bs4 import BeautifulSoup
from collections import defaultdict
from functions_wolmar import *

work_dir = 'D:\\_PROJECTS\\Coins_Classification\\'
directory = 'D:\\_PROJECTS\\Coins_Classification\\Images\\_Roubles_XVIII_Wolmar\\'
input_pkl = work_dir + 'Databases\\Wolmar_3_monety-rossii-do-1917-serebro.pkl'


# ========== DataFrame Filtering ========== #
all_conditions = {1:'G', 2:'G 3', 3:'G 4',
              4:'VG', 5:'VG 8', 6:'VG/F-', 7:'VG/F', 8:'VG/F+',
              9:'F-', 10:'F 12', 11:'F', 12:'F+', 13:'F/VF', 14:'F 15', 15:'F-VF',
              16:'VF-', 17:'VF 20', 18:'VF', 19:'VF 25', 20:'VF+', 21:'VF 30', 22:'VF/VF-XF', 23:'VF/XF', 24:'VF 35', 25:'VF-XF',
              26:'XF-', 27:'XF-/XF', 28:'XF 40', 29: 'XF', 30:'XF/XF+', 31:'XF 45', 32: 'XF+', 33:'XF/AU', 34:'XF+/AU',
              35:'AU 50', 36:'AU', 37:'AU 55', 38:'AU/UNC', 39:'AU 58', 40:'UNC',
              41:'Proof-Like', 42:'Proof', 43:'MS 60', 44:'MS 61', 45:'MS 62', 46:'MS 63',
              47:'MS 64', 48:'MS 65', 49:'MS 66', 50:'MS 67', 51:'MS 68', 52:'MS 69', 53:'MS 70'}

all_nominals = {1: 'Полполушки (медь)', 2: 'Полушка (медь)', 3: 'Денга (медь)', 4: 'Копейка (медь)',
                5: '2 копейки (медь)', 6: '3 копейки (медь)', 7: '4 копейки (медь)', 8: '5 копеек (медь)',
                9: '10 копеек (медь)',
                10: "Копейка (серебро)", 11: 'Алтын (серебро)', 12: '5 копеек (серебро)', 13: '10 копеек (серебро)',
                14: '15 копеек (серебро)', 15: '20 копеек (серебро)', 16: '25 копеек (серебро)',
                17: 'Полтина (серебро)', 18: 'Рубль (серебро)',
                19: 'Полтина (золото)', 20: 'Рубль (золото)', 21: '2 рубля (золото)', 22: '3 рубля (золото)',
                23: '5 рублей (золото)', 24: '7.5 рублей (золото)', 25: '10 рублей (золото)', 26: '15 рублей (золото)',
                27: 'Червонец (золото)', 28: 'Двойной червонец (золото)',
                29: '3 рубля (платина)', 30: '6 рублей (платина)', 31: '12 рублей (платина)',
                32: 'Полушка (Сибирь)', 33: 'Денга (Сибирь)', 34: 'Копейка (Сибирь)', 35: "2 копейки (Сибирь)",
                36: "5 копеек (Сибирь)", 37: "10 копеек (Сибирь)"
                }

#nominal = 'талер'
nominal = 18
saved = 0


def img_save(auction, lot_id, nominal, year, price, date, directory):
    global saved
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in [1, 2]:
        image_name = f"{directory}{str(nominal)}_{str(year)}__{str(price)}__{date}__{i}.png"
        if os.path.exists(image_name):
            print(f"{saved}: {auction} {str(year)} {str(lot_id)} {i} is  SKIPPED")
        else:
            coin_img_path = f"https://www.wolmar.ru/images/auctions/{str(auction)}/{str(lot_id)}_{i}.jpg"
            coin = requests.get(coin_img_path)
            coin = coin.content
            if len(coin) > 10000:
                with open(image_name, "wb") as f:
                    f.write(coin)
                f.close()
                saved += 1
                print(f"{saved}: {auction} {str(year)} {str(lot_id)} {i} is  SAVED")
            else:
                print(f"{saved}: {auction} {str(year)} {str(lot_id)} {i} is  SKIPPED")


df = pd.read_pickle(input_pkl)
print("\nFound {} coins in the given Database".format(len(df)))
#df = filter_auction(df, 266, auction_2=2000)
df = filter_nominal(df, nominal=nominal)
#df = filter_metal(df, metal='Pt')
#df = filter_special(df, special='300')
df = filter_year(df, 1700, 1795)
#df = filter_letters(df, letters='ВМ')
df = filter_condition(df, condition1 = 18, condition2 = 40)
#df = filter_price(df, price_low = 15000, price_high = 300000)
df = filter_date(df, 2012, 2024)
df = df.sort_values('Date')
#print_filtered(df)
#df = filter_outstanding(df)
print()

#print(df)
time_start = time.time()
for index, row in df.iterrows():
    # print(df.lot_id)
    auction = row['ID'].split('_')[1]
    date = row['Date']

    img_save(auction, row['lot_id'], 'Rouble', row['Year'], row['Price'], date, directory)

time_end = time.time()
print('\nSaving took ' + str(datetime.timedelta(seconds=round(time_end - time_start))))