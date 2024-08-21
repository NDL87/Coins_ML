import pandas as pd
from transliterate import translit
import re
import time
from functools import wraps


# Adjust the display options
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


def function_timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{func.__name__}:  {duration:.3f} sec")
        return result
    return wrapper


nominal_replace_dict = {"Rubl'": "Rubl'", "1 rubl'": "Rubl'", 'Poltina': '50 kopeek', 
                         'Poltinnik': '50 kopeek', '50 kopeek': '50 kopeek', 'PoAtina': '50 kopeek', 
                         '25 kopeek': '25 kopeek', 'Polupoltinnik': '25 kopeek', 
                         '20 kopeek': '20 kopeek', '15 kopeek': '15 kopeek', '10 kopeek': '10 kopeek', 
                         'Grivennik': '10 kopeek', '1 grivnja': '10 kopeek', '10 kopee': '10 kopeek', 
                         '5 kopeek': '5 kopeek', '3 kopejki': '3 kopejki', 'Altyn': '3 kopejki', 
                         '2 kopejk': '2 kopejki', '1 kopejk': 'Kopejka', 'Kopejka': 'Kopejka', 
                         '1/2 kopejki': '0.5 kopeek', "Den'ga": '0.5 kopeek', 'Denezhka': '0.5 kopeek', 
                         'Denga': '0.5 kopeek', "1 den'ga": '0.5 kopeek', 'Polushka': '0.25 kopeek', 
                         '1/4 kopejki': '0.25 kopeek', '15 rublej': '15 rublej', '10 rublej': '10 rublej', 
                         'Chervonets': '10 rublej', '5 rublej': '5 rublej', '3 rublja': '3 rublja', 
                         '1 grosh': '1 grosh', '5 penni': '5 penni', '10 penni': '10 penni', 
                         '1 penni': '1 penni', 'Dvojnoj abaz': 'Dvojnoj abaz', '1 marka': '1 marka', 
                         'Taler': 'Taler', '7rub.50 kop.': '7 rublej 50 kopeek', '12 rublej': '12 rublej', 
                         '20 marok': '20 marok', '25 penni': '25 penni', '10 marok': '10 marok', 
                         '50 penni': '50 penni', '3/4 rublja': '3/4 rublja', 
                         '30 kopeek': '30 kopeek', '50 zlotyh': '50 zlotyh', 
                         '2 zlotyh': '2 zlotyh', '10 grosh': '10 groshej', '3 grosh': '3 grosha', 
                         '6 rubl': '6 rublej', '2 grosh': '2 grosha', '6 grosh': '6 groshej',
                         'Grivna': '10 kopeek', '1 1/2 rubl': '1.5 rublja',
                         '1 zlotyj': '1 zlotyj', '1,5 rublja': '1.5 rublja',
                         'Poluabaz': 'Poluabaz', 'Abaz': 'Abaz', '10 zlotyh': '10 zlotyh', 
                         '2 marki': '2 marki', '5 zlotyh': '5 zlotyh', '1 denga': '0.5 kopeek', 
                         '1/2 marki': '0.5 marki', '2 rublja': '2 rublja', 'Polpoltiny': '25 kopeek', 
                         '5 kopee': '5 kopeek', '4 kopejki': '4 kopejki', "1rubl'": "Rubl'", 
                         '1 zlotyj': '1 zlotyj', 'Bisti': 'Bisti', 'Tinf': 'Tymf', '25 zlotyh': '25 zlotyh', 
                         'Grivnja': 'Grivnja', 'Poluksha': '0.25 kopeek', "Timf": "Tymf",
                         "5 grosh": "5 groshej", "Polkopejki": "0.5 kopeek", 
                         "Tymf": "Tymf", "2 abaza": "2 abaza", "Tynf": "Tynf", "3 kopeek": "3 kopejki",
                         "Para": "Para", "Polubisti": "Polubisti",
                         "7 rublej 50 kopeek": "7 rublej 50 kopeek"}


metal_replace_dict = {"cu": "cu", "ag": "ag", "au": "au", "pt": "pt", "cu-ni": "cu-ni", 
                      "zn/sn": "zn/sn", "billon": "ag", "zn": "zn", "fe": "fe", "met": "?"}


nominal_dict = {}
for key, value in nominal_replace_dict.items():
    nominal_dict[key.lower()] = value.lower()


df_exchange_rates = pd.read_csv('D:\\_PROJECTS\\Coins_Classification\\Stock\\usdrub_as_of_08_2021.csv')
df_exchange_rates['Date'] = pd.to_datetime(df_exchange_rates['Date'])
df_exchange_rates = df_exchange_rates.sort_values('Date')

df = pd.read_csv('D:\_PROJECTS\Coins_Classification\Databases\Joined_Database.csv')
print(f"\nInitial df shape: {df.shape}")
#print(df.head())

pd.set_option('display.max_rows', None)
#condition_counts = df['Condition'].value_counts()
#print(condition_counts)
#mask = df['Condition'].isin(condition_counts[condition_counts >= 1000].index)
#df_filtered = df[mask]
#pd.set_option('display.max_rows', 20)

#print(*df['Condition'].unique(), sep='\n')
df = df.drop(columns=['Auction', 'Winner', '10', 'ID', 'Bids'])

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
start_date = pd.to_datetime('2008-01-01')
end_date = pd.to_datetime('2021-01-01')
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
df = df[(df['Year'] >= 1700) & (df['Year'] <= 1917)]
df_exchange_rates = df_exchange_rates[(df_exchange_rates['Date'] >= start_date) & (df_exchange_rates['Date'] <= end_date)]

df_merged = pd.merge_asof(df, df_exchange_rates, on='Date', direction='backward')

# Convert prices to USD
df_merged['Price_USD'] = (df_merged['Price'] / df_merged['Rate']).round(2)
df = df_merged.drop(columns=['Price', 'Rate'])


# Function to transliterate text
def transliterate_text(text):
    return translit(text, 'ru', reversed=True)


df['Metal_EN'] = df['Metal'].apply(transliterate_text).str.lower()
df['Nominal_EN'] = df['Nominal'].apply(transliterate_text).str.lower()
df.drop(columns=['Nominal', 'Metal'], inplace=True)

print(f"\ndf['Condition'].unique().size = {df['Condition'].unique().size}")
print(df.head())

print(f"\ndf['Nominal_EN'].unique().size = {df['Nominal_EN'].unique().size}")
for key, value in nominal_dict.items():
    pattern = r'\b' + re.escape(key.lower()) + r'.*'  
    df["Nominal_EN"] = df["Nominal_EN"].str.lower().str.replace(pattern, value, regex=True)

print(f"\ndf['Metal_EN'].unique().size = {df['Metal_EN'].unique().size}")
for key, value in metal_replace_dict.items():
    pattern = r'\b' + re.escape(key.lower()) + r'.*'  
    df["Metal_EN"] = df["Metal_EN"].str.lower().str.replace(pattern, value, regex=True)

print(f"\ndf['Nominal_EN'].unique().size = {df['Nominal_EN'].unique().size}")
print(f"\ndf['Metal_EN'].unique().size = {df['Metal_EN'].unique().size}")
indeces_to_drop = list()
for index, row in df.iterrows():
    if 'podborka' in row['Nominal_EN'].lower():
        indeces_to_drop.append(index)
    elif row['Nominal_EN'] not in nominal_dict.values():
        indeces_to_drop.append(index)
    elif row['Metal_EN'] not in metal_replace_dict.values():
        indeces_to_drop.append(index)
print(f"{len(indeces_to_drop)} will be droped\n")
#for n in indeces_to_drop:
#    print(df.loc[n, ['Nominal_EN', 'Metal_EN']])
df.drop(indeces_to_drop, inplace=True)


print(f"\ndf['Nominal_EN'].unique().size = {df['Nominal_EN'].unique().size}")
print(f"\ndf['Metal_EN'].unique().size = {df['Metal_EN'].unique().size}")

print(*df['Nominal_EN'].unique(), sep='\n')
print()
print(*df['Metal_EN'].unique(), sep='\n')

print(f"\nNew df shape: {df.shape}")
#print(*df['Nominal_EN'].unique()[0:10], sep='\n')

df.rename(columns={'Nominal_EN': 'Nominal', 'Metal_EN': 'Metal'}, inplace=True)
#df.drop(columns=['Nominal_EN', 'Metal_EN'], inplace=True)
df = df[["Year", "Nominal", "Letters", "Metal", "Condition", "Date", "Price_USD"]]

df['Nominal'] = df['Nominal'].str.capitalize()
df['Metal'] = df['Metal'].str.capitalize()
#df.sort_values(["Year"], ascending=[True])
df = df.sort_values(["Year", "Nominal", "Date"], ascending=[True, True, True])
df = df.reset_index(drop=True)
max_price_row = df[df['Price_USD'] == df['Price_USD'].max()]
print(max_price_row)
df.drop(max_price_row.index, inplace=True)
max_price_row = df[df['Price_USD'] == df['Price_USD'].max()]
print(max_price_row)
print(df.head())
df.to_csv('D:\_PROJECTS\Coins_Classification\Databases\Russian_Emp_Coins_1700-1917_Auction_Prices_2008-2020_in_USD_Cleaned.csv')
