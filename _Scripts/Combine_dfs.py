from functions_wolmar import *

input_pkls = ['C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\Wolmar_Database_GOLD.pkl',
              'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\Wolmar_Database_SILVER.pkl',
              'C:\\_PYTHON_learn\\WOLMAR\\HTMLs\\Wolmar_Database_COPPER.pkl']

result = pd.DataFrame()
for input_pkl in input_pkls:
     df = pd.read_pickle(input_pkl)
     df['Auction'] = ['wol' for i in range(len(df))]
     df = df.sort_values('Date')
     print(df)
     print()
     result = result.append(df, ignore_index=True, sort=False)

input_pkls = ['C:\\_PYTHON_learn\\CONROS\\HTMLs\\Conros_Database_AgAu.pkl',
              'C:\\_PYTHON_learn\\CONROS\\HTMLs\\Conros_Database_Cu.pkl']
for input_pkl in input_pkls:
     df = pd.read_pickle(input_pkl)
     df['Auction'] = ['con' for i in range(len(df))]
     df = df.sort_values('Date')
     print(df)
     print()
     result = result.append(df, ignore_index=True, sort=False)

print(len(result))
print(result)

work_dir = 'C:\\_PYTHON_learn\\'
output_database_name = 'Joined_Database'

save_dataframe(df=result, directory=work_dir, filename=output_database_name)