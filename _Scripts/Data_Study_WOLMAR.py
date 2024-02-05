from functions_wolmar import *

usdrub_csv_file = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\USDRUB.csv'
gold_csv_file = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\GOLD.csv'
usdrub_pkl_file = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\USDRUB_as_of_12_2020'
gold_pkl_file = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\GOLD_as_of_12_2020'
goldusd_pkl_file = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\GOLD_in_USD'
output_dir = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\Plots\\'
input_pkl = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\Wolmar_Database_GOLD.pkl'
#input_pkl = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\Wolmar_Database_SILVER.pkl'
#input_pkl = 'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\Wolmar_Database_COPPER.pkl'


# ========== Load base DataFrame ========== #
df = pd.read_pickle(input_pkl)
print("\nFound {} coins in the given Database".format(len(df)))

# ========== Reference DataFrames ========== #
df_usd = reference_df(usdrub_pkl_file, usdrub_csv_file)     # USD/RUB Exchange Rate df
df_gold = reference_df(gold_pkl_file, gold_csv_file)        # Gold/RUB Exchange Rate df
gold_in_usd = reference_df(goldusd_pkl_file, gold_csv_file)        # Gold/USD Exchange Rate df

# ========== DataFrame Filtering ========== #
all_conditions = {1:'G', 2:'G 3', 3:'G 4',
              4:'VG', 5:'VG 8', 6:'VG/F-', 7:'VG/F', 8:'VG/F+',
              9:'F-', 10:'F 12', 11:'F', 12:'F+', 13:'F/VF', 14:'F 15', 15:'F-VF',
              16:'VF-', 17:'VF 20', 18:'VF', 19:'VF 25', 20:'VF+', 21:'VF 30', 22:'VF/VF-XF', 23:'VF/XF', 24:'VF 35', 25:'VF-XF',
              26:'XF-', 27:'XF-/XF', 28:'XF 40', 29: 'XF', 30:'XF/XF+', 31:'XF 45', 32: 'XF+', 33:'XF/AU', 34:'XF+/AU',
              35:'AU 50', 36:'AU', 37:'AU 55', 38:'AU/UNC', 39:'AU 58', 40:'UNC',
              41:'Proof-Like', 42:'Proof', 43:'MS 60', 44:'MS 61', 45:'MS 62', 46:'MS 63',
              47:'MS 64', 48:'MS 65', 49:'MS 66', 50:'MS 67', 51:'MS 68', 52:'MS 69', 53:'MS 70'}
df = filter_outstanding(df)
df = filter_auction(df, auction='con')
df = filter_nominal(df, nominal='10 руб')
#df = filter_metal(df, metal='Cu')
#df = filter_special(df, special='Сибир')
df = filter_year(df, 1898, 1904)
#df = filter_letters(df, letters='ЕМ')
df = filter_condition(df, condition1=26, condition2=36)
df = filter_price(df, price_low=0, price_high=45000)
#df = filter_date(df, 2019, 2020)
df = df.sort_values('Date')
#print_filtered(df)

# ========== Plotting Filtered DataFrame ========== #
#plot_bar(df, param='Year', save='no', output_dir=output_dir)
df = one_price_per_date(df)
plot_price(df=df, moving_average=15, currency='usd', df_usd=df_usd,
           df2=gold_in_usd, df2_label='Gold in USD', scaling_df2=8.6, save='no', output_dir=output_dir)

# TODO list:
# Plotter: never-ending improvements
# Wrap (last updates saver) around "save html" and "add to database"
