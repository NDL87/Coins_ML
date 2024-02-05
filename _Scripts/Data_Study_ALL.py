from functions_wolmar import *

work_dir = 'D:\\_PROJECTS\\Coins_Classification\\'
usdrub_csv_file = work_dir + 'Stock\\usdrub_as_of_08_2021.csv'
usdrub_pkl_file = work_dir + 'Stock\\usdrub_as_of_08_2021'
gold_csv_file = work_dir + 'Stock\\gold_as_of_08_2021.csv'
gold_pkl_file = work_dir + 'Stock\\gold_as_of_08_2021'
goldusd_csv_file = work_dir + 'Stock\\goldusd_as_of_08_2021.csv'
goldusd_pkl_file = work_dir + 'Stock\\goldusd_as_of_08_2021'
output_dir = work_dir + 'Plots\\'
input_pkl = work_dir + 'Databases\\Wolmar_3_monety-rossii-do-1917-serebro.pkl'

database = ['0_monety-antika-srednevekove',
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

selected_database = 3
# ========== Reference DataFrames ========== #
#input_pkl = work_dir + 'Databases\\Wolmar_' + database[selected_database] + '.pkl'
df_usd = reference_df(usdrub_pkl_file, usdrub_csv_file)     # USD/RUB Exchange Rate df
#print(df_usd)
df_gold = reference_df(gold_pkl_file, gold_csv_file)        # Gold/RUB Exchange Rate df
#print(df_gold)
gold_in_usd = reference_df(goldusd_pkl_file, goldusd_csv_file)
#if Path(goldusd_pkl_file + '.pkl').is_file():
#    gold_in_usd = reference_df(gold_pkl_file, goldusd_csv_file)
#else:
#    gold_in_usd = price_in_usd(df_gold, df_usd)        # Gold/USD Exchange Rate df
#    save_dataframe(df=gold_in_usd, filename=goldusd_pkl_file)
#print(gold_in_usd)

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

# ========== DataFrame (df) Filtering ========== #
df = pd.read_pickle(input_pkl)
print("\nFound {} coins in the given Database".format(len(df)))
#df = filter_auction(df, auction='con')
df = filter_nominal(df, nominal=nominal)
#df = filter_metal(df, metal='Pt')
#df = filter_special(df, special='300')
df = filter_year(df, 1700, 1799)
#df = filter_letters(df, letters='ВМ')
df = filter_condition(df, condition1 = 23, condition2 = 33)
df = filter_price(df, price_low = 15000, price_high = 300000)
#df = filter_date(df, 2020, 2021)
df = df.sort_values('Date')
#print_filtered(df)
#df = filter_outstanding(df)
print()


# ========== DataFrame_2 (df2) Filtering ========== #
check_df2 = 0
if check_df2 != 0:
    df2 = pd.read_pickle(input_pkl)
    df2 = filter_outstanding(df2)
    df2 = filter_auction(df2, auction='con')
    df2 = filter_nominal(df2, nominal='5 коп')
    #df2 = filter_metal(df2, metal='Cu')
    #df2 = filter_special(df2, special='Сибир')
    df2 = filter_year(df2, 1801)
    df2 = filter_letters(df2, letters='АИ')
    #df2 = filter_condition(df2, condition1=26, condition2=36)
    #df2 = filter_price(df2, price_low=0, price_high=25000)
    df2 = filter_date(df2, 2010, 2020)
    df2 = df2.sort_values('Date')
    #print_filtered(df2)
    df2 = one_price_per_date(df2)
else:
    df2 = 0

# ========== Plotting Filtered DataFrame ========== #
#plot_bar(df, param='Year', save='no', output_dir=output_dir)
#plot_bar(df2, param='Year', save='yes', output_dir=output_dir)
time_start = time.time()
#df = one_price_per_date_2(df, trim=0.05)
df = one_price_per_date(df)
time_end = time.time()
print('\nFunction performance took {} seconds'.format(str(datetime.timedelta(seconds=round(time_end - time_start)))))
plot_price(df=df, moving_average=10, currency='rub', df_usd=df_usd,
           df2=0, df2_label='gold_in_usd', scaling_df2=3.87,
           save='yes', output_dir=output_dir, nominal=nominal)

# TODO list:
# Plotter: never-ending improvements
# Wrap (last updates saver) around "save html" and "add to database"
