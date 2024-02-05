from pandas import DataFrame as df
import pandas as pd
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import math
import re
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from scipy import stats
import sys, os
import time


def block_print():   # Disable printing
    sys.stdout = open(os.devnull, 'w')


def enable_print():  # Restore printing
    sys.stdout = sys.__stdout__


def roundup(x, step):
    return int(math.ceil(x / step)) * step


def print_filtered(df):
    print()
    headers = ["Date", "Price", "Nominal", "Year", "Letters", "Condition", "ID"]
    for i in range(len(df)):
        out = ''
        for h in headers:
            try:
                out = out + str(df[h].tolist()[i]) + ' '
            except: pass
        print(out)


def filter_nominal(df, nominal):
    df_filtered = pd.DataFrame()
    nominal_regex = []
    if nominal == 1: nominal_regex = ['Полполушки', '1/8 копейки']
    if nominal == 2: nominal_regex = ['Полушка', '1/4 копейки']
    if nominal == 3: nominal_regex = ['Денга', "Денежка", "Деньга", '1/2 копейки']
    if nominal == 4: nominal_regex = ['Копейка', "1 копейка"]
    if nominal == 5: nominal_regex = ['2 копейки']
    if nominal == 6: nominal_regex = ['3 копейки']
    if nominal == 7: nominal_regex = ['4 копейки']
    if nominal == 8: nominal_regex = ['5 копеек']
    if nominal == 9: nominal_regex = ['10 копеек']
    if nominal == 10: nominal_regex = ['Копейка', "1 копейка"]
    if nominal == 11: nominal_regex = ['Алтын', "Алтынник", "Алтынникъ"]
    if nominal == 12: nominal_regex = ['5 копеек']
    if nominal == 13: nominal_regex = ['10 копеек']
    if nominal == 14: nominal_regex = ['15 копеек']
    if nominal == 15: nominal_regex = ['20 копеек']
    if nominal == 16: nominal_regex = ['25 копеек', "Полуполтинник", "Полуполтинникъ"]
    if nominal == 17: nominal_regex = ['50 копеек', "Полтина", "Полтинникъ", "Полтинник"]
    if nominal == 18: nominal_regex = ['Рубль', "1 рубль", "рубль"]
    if nominal == 19: nominal_regex = ['Полтина', "полтина"]
    if nominal == 20: nominal_regex = ['Рубль', "1 рубль", "рубль"]
    if nominal == 21: nominal_regex = ["2 рубля"]
    if nominal == 22: nominal_regex = ["3 рубля"]
    if nominal == 23: nominal_regex = ["5 рублей"]
    if nominal == 24: nominal_regex = ["7.5 рублей"]
    if nominal == 25: nominal_regex = ["10 рублей"]
    if nominal == 26: nominal_regex = ["15 рублей"]
    if nominal in range(1, 9):
        df = filter_metal(df, metal='Cu')
    if nominal in range(10, 13):
        df = filter_metal(df, metal='Ag')
    if nominal in range(19, 20):
        df = filter_metal(df, metal='Au')
    for reg in nominal_regex:
        df_filtered = pd.concat([df_filtered, df[df['Nominal'].str.startswith(reg)]], ignore_index=True)
    if not isinstance(nominal, int):
        df_filtered = pd.concat([df_filtered, df[df['Nominal'].str.contains(nominal)]], ignore_index=True)
        print("Found {} coins with Nominal: '{}'".format(len(df_filtered), nominal))
    else:
        print("Found {} coins with Nominal: '{}'".format(len(df_filtered), all_nominals[nominal]))
    return df_filtered


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


def filter_nominal_old(df, nominal):
    df_filtered = df[df['Nominal'].str.startswith(nominal)]
    print("Found {} coins with Nominal: '{}'".format(len(df_filtered), nominal))
    return df_filtered


def filter_special(df, special):
    df_filtered1 = df[df['Nominal'].str.contains(special)]
    df_filtered2 = df[df['Letters'].str.contains(special)]
    df_filtered = df_filtered1.append(df_filtered2, ignore_index=True, sort=False)
    print("Found {} coins with Special condition: '{}'".format(len(df_filtered), special))
    return df_filtered


def filter_letters(df, letters):
    df_filtered = df[df['Letters'].str.contains(letters)]
    print("Found {} coins with Letters: '{}'".format(len(df_filtered), letters))
    return df_filtered


def filter_metal(df, metal):
    df_filtered = df[df['Metal'].str.contains(metal)]
    print("Found {} coins in Metal: '{}'".format(len(df_filtered), metal))
    return df_filtered


def filter_auction(df, auction):
    df_filtered = df[df['Auction'].str.contains(auction)]
    print("Found {} coins in Auctions: '{}'".format(len(df_filtered), auction))
    return df_filtered


'''
def filter_auction(df, auction, auction_2=0):
    df['Auction'] = df['Year'].astype(int)
    if auction_2 == 0:
        df_filtered = df[df['Year'] == year]
        print("Found {} coins with Year: {}".format(len(df_filtered), year))
    else:
        df_filtered = df[(df['Year'] >= year) & (df['Year'] <= year_2)]
        print("Found {} coins with Year between {} and {}".format(len(df_filtered), year, year_2))
    return df_filtered
'''


def filter_year(df, year, year_2=0):
    df['Year'] = df['Year'].astype(int)
    if year_2 == 0:
        df_filtered = df[df['Year'] == year]
        print("Found {} coins with Year: {}".format(len(df_filtered), year))
    else:
        df_filtered = df[(df['Year'] >= year) & (df['Year'] <= year_2)]
        print("Found {} coins with Year between {} and {}".format(len(df_filtered), year, year_2))
    return df_filtered


def filter_date(df, date, date_2=0):
    dat = date
    if not isinstance(date, datetime.date):
        dat = pd.DataFrame({'Date': [date]})
        dat = pd.to_datetime(dat['Date'], format='%Y')
        dat = dat.tolist()[0]
    if date_2 == 0:
        df_filtered = df[pd.to_datetime(df['Date']).dt.year == dat.year]
        print("Found {} coins sold in Year: {}".format(len(df_filtered), date.year))
    else:
        dat2 = date_2
        if not isinstance(date_2, datetime.date):
            dat2 = pd.DataFrame({'Date': [date_2]})
            dat2 = pd.to_datetime(dat2['Date'], format='%Y')
            dat2 = dat2.tolist()[0]
        df_filtered = df[(pd.to_datetime(df['Date']).dt.year >= dat.year) & (pd.to_datetime(df['Date']).dt.year <= dat2.year)]
        print("Found {} coins sold between {} and {}".format(len(df_filtered), dat.year, dat2.year))
    return df_filtered


def filter_condition(df, condition1, condition2):
    all_conditions = {1: 'G', 2: 'G 3', 3: 'G 4',
                      4: 'VG', 5: 'VG 8', 6: 'VG/F-', 7: 'VG/F', 8: 'VG/F+',
                      9: 'F-', 10: 'F 12', 11: 'F', 12: 'F+', 13: 'F/VF', 14: 'F 15', 15: 'F-VF',
                      16: 'VF-', 17: 'VF 20', 18: 'VF', 19: 'VF 25', 20: 'VF+', 21: 'VF 30', 22: 'VF/VF-XF',
                      23: 'VF/XF', 24: 'VF 35', 25: 'VF-XF',
                      26: 'XF-', 27: 'XF-/XF', 28: 'XF 40', 29: 'XF', 30: 'XF/XF+', 31: 'XF 45', 32: 'XF+', 33: 'XF/AU',
                      34: 'XF+/AU',
                      35: 'AU 50', 36: 'AU', 37: 'AU 55', 38: 'AU/UNC', 39: 'AU 58', 40: 'UNC',
                      41: 'Proof-Like', 42: 'Proof', 43: 'MS 60', 44: 'MS 61', 45: 'MS 62', 46: 'MS 63',
                      47: 'MS 64', 48: 'MS 65', 49: 'MS 66', 50: 'MS 67', 51: 'MS 68', 52: 'MS 69', 53: 'MS 70'}
    mykeys = [i for i in range(condition1, condition2+1)]
    conditions = [all_conditions[x] for x in mykeys]
    df_filtered = pd.DataFrame()
    for cond in conditions:
        df_filtered = pd.concat([df_filtered, df[df['Condition'] == cond]], ignore_index=True)
    print("Found {} coins with Conditions: {}".format(len(df_filtered), ', '.join(conditions)))
    return df_filtered


def filter_price(df, price_low, price_high):
    df_filtered = df[(df['Price'] >= price_low) & (df['Price'] <= price_high)]
    print("Found {} coins with Price between {} and {}".format(len(df_filtered), price_low, price_high))
    return df_filtered


def min_price(df):
    df_filtered = df[df['Price'] == min(df['Price'])]
    return df_filtered


def filter_outstanding(df):
    forbidden_names = ["КОПИЯ", "ПРОБНАЯ", "копия", "Копия", "отверстие", "Подделка", "РЕМОНТ"]
    for word in forbidden_names:
        df = df[~df.Nominal.str.contains(word)]
    print("Found {} coins with NO forbidden words: '{}'".format(len(df), ', '.join(forbidden_names)))
    return df


# apply Moving Average to the given list with a given window
def mov_avg(df, window):
    rolling_mean = df.Price.rolling(window=window).mean().tolist()
    return rolling_mean


# return nearest value to 'pivot' in the list 'items'
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def save_list_to_file(filename, list):
    with open(filename, 'w+') as f:
        for line in list:
            f.writelines(str(line))
            f.writelines('\n')
    f.close()


# This function returns modified DataFrame with prices in USD as of date of auction
def price_in_usd(df, usdrub):
    price_usd = []
    for index, row in df.iterrows():
        df_date = row['Date']
        df_price = row['Price']
        usd_date = nearest(usdrub.Date, df_date)
        rate_of_interest = usdrub.loc[usdrub.Date == usd_date, 'Price'].iloc[0]
        usd_price = df_price / rate_of_interest
        price_usd.append(usd_price)
    price_rub = df["Price"].tolist()
    df["Price"].replace(to_replace=price_rub, value=price_usd, inplace=True)
    return df


def save_dataframe(df, filename, directory=''):
    output_csv = directory + filename + '.csv'
    output_pkl = directory + filename + '.pkl'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    df.to_pickle(output_pkl)


def csv_to_df(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8-sig', sep=",")
    formats = ["%Y-%m-%d", "%d.%m.%Y"]
    for format in formats:
        try:
            df.Date = pd.to_datetime(df.Date, format=format)
        except:
            pass
    df.Date = pd.to_datetime(df.Date)
    df.Date = df.Date.dt.date
    df['Price'] = df['Price'].astype(float)
    return df


def reference_df(reference_pkl, reference_csv):
    if Path(reference_pkl+'.pkl').is_file():
        ref_df = pd.read_pickle(reference_pkl+'.pkl')
    else:
        ref_df = csv_to_df(reference_csv)
        save_dataframe(df=ref_df, filename=reference_pkl)
    return ref_df


def one_price_per_date(df):
    dates = []
    prices = []
    nominals = df.Nominal.tolist()[0]
    years = []
    letters = df.Letters.tolist()[0]
    conditions = df.Condition.tolist()[0]
    ids = df.ID.tolist()[0]
    metals = df.Metal.tolist()[0]
    for date in df.Date.tolist():
        if date not in dates:
            dates.append(date)
            df_tmp = df[df.Date == date]
            price_avg = int(round(sum(df_tmp.Price.tolist())/len(df_tmp.Price.tolist())))
            prices.append(price_avg)
            years.append(df.loc[df.Price == nearest(df.Price.tolist(), price_avg)].Year.tolist()[0])
    df_new = pd.DataFrame({'Date': dates, 'Price': prices, 'Nominal': nominals, 'Year': years,
                           'Letters': letters, 'Conditions': conditions, 'ID': ids, 'Metal': metals,})
    return df_new


def one_price_per_date_2(df, trim=0.2):
    dates = []
    prices = []
    nominals = df.Nominal.tolist()[0]
    years = []
    letters = df.Letters.tolist()[0]
    conditions = df.Condition.tolist()[0]
    ids = df.ID.tolist()[0]
    metals = df.Metal.tolist()[0]
    for index, day in enumerate(df.Date.tolist()):
        if day not in dates:
            dates.append(day)
            if index==0:
                df_tmp = df[(df.Date == day) | (df.Date == df.Date.tolist()[index + 1])]
            elif index == (len(df.Date.tolist())-1):
                df_tmp = df[(df.Date == day) | (df.Date == df.Date.tolist()[index - 1])]
            else:
                df_tmp = df[(df.Date == day) | (df.Date == df.Date.tolist()[index - 1]) | (
                    df.Date == df.Date.tolist()[index + 1])]
            price_list = df_tmp.Price.tolist()
            price_list.sort()
            price_list_new = price_list[int(len(price_list) * trim): int(len(price_list) * (1 - trim))]
            block_print()
            df_new = filter_price(df_tmp, price_low=min(price_list_new), price_high=max(price_list_new))
            enable_print()
            price_avg = stats.trim_mean(df_new.Price.tolist(), 0.1)
            prices.append(price_avg)
            years.append(df.loc[df.Price == nearest(df.Price.tolist(), price_avg)].Year.tolist()[0])
    df_upd = pd.DataFrame({'Date': dates, 'Price': prices, 'Nominal': nominals, 'Year': years,
                           'Letters': letters, 'Conditions': conditions, 'ID': ids, 'Metal': metals, })
    return df_upd


def step_finder(min_y, max_y):
    if (max_y-min_y) < 100000:
        step_y = 10000
    else:
        step_y = 50000
    if (max_y-min_y) < 50000:
        step_y = 5000
    if (max_y-min_y) < 10000:
        step_y = 1000
    if (max_y-min_y) < 5000:
        step_y = 500
    if (max_y-min_y) < 1000:
        step_y = 100
    if (max_y - min_y) < 500:
        step_y = 50
    if (max_y - min_y) < 100:
        step_y = 10
    return step_y


def plot_bar(df, param='Year', save='no', output_dir=''):
    param_list = sorted(list(set(df[param].tolist())))
    results_list = []
    for p in param_list:
        counted = len(df[df[param] == p])
        results_list.append(counted)
    fig = plt.figure(figsize=[8, 3], dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    bars = ax.bar(param_list, results_list)
    ax.set_xticks(param_list)
    min_y = 0
    max_y = max(results_list)
    step_y = step_finder(min_y, max_y)
    ax.set_yticks(np.arange(0, roundup(max_y, step_y)+step_y, step_y))
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    if save != 'no':
        fig1.savefig("{}Bars_{}_{}_{}.png".format(output_dir, param, min(param_list), max(param_list)), dpi=150)
        print("\nFigure is saved to: {}Bars_{}_{}_{}.png".format(output_dir, param, min(param_list), max(param_list)))
    plt.close(fig='all')


def plot_price(df, df_usd=0, df2=0, df2_label='', currency='rub', moving_average=0, scaling_df2=1, save='no',
               output_dir='', nominal=''):
    time_start = time.time()
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    dates = df['Date'].tolist()
    if isinstance(nominal, int):
        nominal = all_nominals[nominal]
    fig_x_size = 6
    fig_y_size = 4
    fig = plt.figure(figsize=[fig_x_size, fig_y_size], dpi=100)
    x_offset = 0.025
    ax1 = fig.add_subplot(1, 1, 1)
    if currency == 'usd' and isinstance(df_usd, pd.DataFrame):
        df = price_in_usd(df, df_usd)
    prices = df['Price'].tolist()
    min_y = min(prices)
    max_y = max(prices)
    color = 'tab:red'
    ax1.set_ylabel('Price ({})\n'.format(currency), color='black', fontsize=12, fontweight='bold')
    ax1.scatter(df.Date, df['Price'], label='Auction price', s=5,  c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    title_input = "Price of the coin '{}' \n".format(nominal)
    plt.title(title_input, fontsize=12, fontweight='bold')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1.get_legend_handles_labels()
    step_y = step_finder(min_y, max_y)
    gody = set(df['Year'].tolist())
    if len(gody) > 1:
        gody = 'Years: ' + str(min(gody)) + ' - ' + str(max(gody))
    else:
        gody = 'Year: ' + str(min(gody))
    legend = ax1.legend(h1, l1, borderaxespad=0, title='Nominal: {}\n{}\n'.format(nominal, gody),
                        ncol=1, loc="upper right", bbox_to_anchor=(1-(x_offset*fig_y_size/fig_x_size), 1-x_offset), fontsize=12, framealpha=1)
    if moving_average != 0:
        ax1.plot(df.Date, mov_avg(df, moving_average), label='Mov.avg.({}) for Auction price'.format(moving_average), c="blue")
    if isinstance(df2, pd.DataFrame):
        df2['Price'] = df2['Price'].apply(lambda x: x * scaling_df2)
        df2 = filter_date(df2, date=min(dates), date_2=max(dates))
        color = 'tab:green'
        ax2 = ax1.twinx()
        my_new_list = df2['Price']
        if moving_average != 0:
            ax2.plot(df2.Date, mov_avg(df2, moving_average), label='Mov.avg.({}) for {}'.format(moving_average, df2_label),
                     c=color, linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('{}'.format(df2_label), color='black', fontsize=12, fontweight='bold', rotation=270, labelpad=25)
        if min(my_new_list) < min_y:
            min_y = min(my_new_list)
        if max(my_new_list) > max_y:
            max_y = max(my_new_list)
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.set_yticks(np.arange(roundup(min_y, step_y) - step_y, roundup(max_y, step_y) + step_y, step_y))
        legend = ax1.legend(h1 + h2, l1 + l2, borderaxespad=0, title='Nominal: {}\n{}\n'.format(nominal, gody),
                            ncol=1, loc="upper right",
                            bbox_to_anchor=(1 - (x_offset * fig_y_size / fig_x_size), 1 - x_offset), fontsize=12,
                            framealpha=1)

    ax1.set_yticks(np.arange(roundup(min_y, step_y)-step_y, roundup(max_y, step_y)+step_y, step_y))
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
    ax1.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)    # format the ticks
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(years_fmt)
    ax1.xaxis.set_minor_locator(months)
    # round to nearest years.
    datemin = np.datetime64(sorted(df['Date'].tolist())[0], 'M')
    datemax = np.datetime64(sorted(df['Date'].tolist())[-1], 'M') + np.timedelta64(1, 'M')
    ax1.set_xlim(datemin, datemax)
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.autofmt_xdate()

    plt.setp(legend.get_title(), fontsize=12, fontweight='bold')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    time_end = time.time()
    print('\nPlotting took {} seconds'.format(str(datetime.timedelta(seconds=round(time_end - time_start)))))
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    if save != 'no':
        fig1.savefig("{}Plot_{}_{}.png".format(output_dir, nominal, currency), dpi=150)
        print("\nFigure is saved to: {}Plot_{}_{}.png".format(output_dir, nominal, currency))
    plt.close(fig='all')

