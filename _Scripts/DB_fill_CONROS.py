from pandas import DataFrame as df
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
import time
import datetime
from functions_wolmar import *
from urllib.request import urlopen

time_start = time.time()

work_dir = 'D:\\_PROJECTS\\Coins_Classification\\HTMLs\\CONROS\\'
html_dir = work_dir + '1_Rus_Emp_AgAu\\'
output_database_name = 'Conros_Database_AgAu'
#html_dir = work_dir + '2_Rus_Emp_Cu\\'
#output_database_name = 'Conros_Database_Cu'


def get_date(lot_id):
    page = urlopen("https://auction.conros.ru/" + str(lot_id))
    page_content = page.read()
    soup2 = BeautifulSoup(page_content, 'html.parser')
    rub20 = soup2.findAll('p', class_="lot_info_box")
    date = rub20[-1].text[-10:]
    date = datetime.datetime.strptime(date, '%d.%m.%Y').date()
    return date


htmlfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk(html_dir)
             for name in files
             if name.endswith((".html", ".htm"))]

print('\nNow {} html files will be processed and saved into Database:\n'.format(len(htmlfiles)))

header_Wolmar = ["ID", "Nominal", "Year", "Letters", "Metal", "Condition", "Bids", "Winner", "Price", "Date"]
df_columns = {}
for key in range(0, 10):
    df_columns[key] = header_Wolmar[key]

table = {}
all_res = []
for html_file in htmlfiles:
    #html_file = 'C:\\_PYTHON_learn\\CONROS\\HTMLs\\1_Rus_Emp_AgAu\\819_1_0.html'
    print(html_file)
    try:
        html_content = open(html_file, 'r')
        html_content = html_content.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        kop20 = soup.findAll('td', class_="productListing-data", onclick=True)
        lot_id = kop20[0]['onclick'].split('=')[1][1:-2]
        date_closure = get_date(lot_id)

        table_id = html_file.split('\\')[-1].split('.')[0].split('_')[0]
        table[table_id] = pd.read_html(html_file, skiprows = 0)
        df = pd.DataFrame(table[table_id][19][1:])
        df.rename(columns=df_columns, inplace=True)
        df = df.dropna()
        df = df[df['Year'].isin([str(i) for i in range(1500, 2100)])]
        df['Year'] = df['Year'].astype(int)
        df["Date"] = [date_closure for i in range(len(df))]
        ID_orig = df["ID"].tolist()
        ID_new = []
        for i in ID_orig:
            ID_new.append(str('con_'+str(table_id)+'_'+str(i)))
        df["ID"].replace(to_replace=ID_orig, value=ID_new, inplace=True)
        df['Price'] = [float(str(val).replace(' ', '').replace(',', '.')) for val in df['Price'].values]
        df['Price'] = df['Price'].astype(int)
        all_res.append(df)
    except:
        pass

df = pd.concat(all_res)
save_dataframe(df=df, directory=work_dir, filename=output_database_name)

time_end = time.time()
print('\nParsing took ' + str(datetime.timedelta(seconds=round(time_end - time_start))))
