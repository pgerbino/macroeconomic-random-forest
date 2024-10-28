import pandas as pd

inflation = pd.read_csv('./series-281024.csv', index_col=0, skiprows=8)
inflation = inflation[inflation.index.str.match(r'^\d{4} [A-Z]{3}$')]
inflation.columns = ['inflation']
inflation.index = pd.to_datetime(inflation.index, format='mixed')
inflation.index.name = 'date'

government_expenditure = pd.read_csv('./government-expenditure.csv', index_col=0, skiprows=8)
government_expenditure = government_expenditure[government_expenditure.index.str.match(r'^\d{4} [A-Z]{3}$')]
government_expenditure.columns = ['government_expenditure']
government_expenditure.dropna(subset=['government_expenditure'], inplace=True)
government_expenditure.index = pd.to_datetime(government_expenditure.index, format='mixed')
government_expenditure.index.name = 'date'

inflation_government_exp_df = inflation.join(government_expenditure, how='inner')
