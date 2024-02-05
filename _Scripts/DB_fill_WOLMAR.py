from pandas import DataFrame as df
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
import time
import lxml
from functions_wolmar import *

time_start = time.time()

work_dir = 'D:\\_PROJECTS\\Coins_Classification\\HTMLs\\'
output_dir = 'D:\\_PROJECTS\\Coins_Classification\\Databases\\'

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
           '11_serebro-i-dr-posle-1945-goda',
           '__TEST']

folders_list = [3]

header_Wolmar = ["ID", "Nominal", "Year", "Letters", "Metal", "Condition", "Winner", "Bids", "Price", "Date"]
df_columns = {}
for key in range(len(header_Wolmar)):
    df_columns[key] = header_Wolmar[key]

for f in folders_list:

    table = {}
    all_res = []

    folder = folders[f]
    html_dir = work_dir + folder + '\\'
    htmlfiles = [os.path.join(root, name)
                 for root, dirs, files in os.walk(html_dir)
                 for name in files
                 if name.endswith((".html", ".htm"))]
    print('\nNow {} html files will be processed and saved into Database:\n'.format(len(htmlfiles)))

    for html_file in htmlfiles:
        print('Processing ' + html_file.split('\\')[-1])
        lot_id = []
        html_content = open(html_file, 'r', encoding='utf-8-sig')
        html_content = html_content.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        try:
            date_closure = str(soup.h1.span).split()[1]
        except:
            skipped = html_file.split('\\')[-1]
            print(f"{skipped} skipped")
            continue
        table_id = html_file.split('\\')[-1].split('.')[0]
        table[table_id] = pd.read_html(html_file, skiprows = 0)
        try:
            df = pd.DataFrame(table[table_id][5][3:])
        except Exception as e:
            if type(e).__name__ == 'IndexError':
                df = pd.DataFrame(table[table_id][4][3:])
        if df.shape[1] != len(header_Wolmar):
            print(f"{table_id} skipped")
            continue
        df.rename(columns=df_columns, inplace=True)
        df = df.dropna(thresh=3)
        df["Date"].replace({"Закрыто": str(date_closure)}, inplace=True)

        ID_orig = df["ID"].tolist()
        ID_new = []
        for i in ID_orig:
            ID_new.append(str('wol_'+str(table_id)+'_'+str(i)))
        df["ID"].replace(to_replace=ID_orig, value=ID_new, inplace=True)
        #''.join(filter(str.isdigit, str(val).replace(' ', '').replace(',', '.').split('-')[0]))
        #df['Price'] = [float(str(val).replace(' ', '').replace(',', '.').split('-')[0]) for val in df['Price'].values]
        df['Price'] = [float(''.join(filter(str.isdigit, str(val).replace(' ', '').replace(',', '.').split('-')[0]))) for val in df['Price'].values]
        df['Price'] = df['Price'].astype(int)
        df['Year'] = [str(val).replace('nan', '0') for val in df['Year'].values]
        #df['Year'] = df['Year'].astype(int)
        df.Date = pd.to_datetime(df.Date, format="%d.%m.%Y")
        df.Date = df.Date.dt.date
        coins = soup.findAll('tr')
        for coin in coins:
            try:
                lot_ids = coin.a['href'].split('/')
                if lot_ids[1] == 'auction':
                    lot_id.append(lot_ids[-1])
            except:
                pass

        lot_id = lot_id[1:]
        df.insert(1, "lot_id", lot_id, True)
        all_res.append(df)
        #print(df)

    df = pd.concat(all_res)
    #print(df)
    output_database_name = 'Wolmar_' + folder
    save_dataframe(df=df, directory=output_dir, filename=output_database_name)

time_end = time.time()
print('\nParsing took ' + str(datetime.timedelta(seconds=round(time_end - time_start))))


