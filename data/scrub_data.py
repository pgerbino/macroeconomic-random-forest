import pandas as pd

inflation = pd.read_csv('./series-281024.csv', index_col=0, skiprows=8)
inflation = inflation[inflation.index.str.match(r'^\d{4} [A-Z]{3}$')]
inflation.columns = ['inflation']
inflation.index = pd.to_datetime(inflation.index, format='mixed')
inflation.index.name = 'date'
inflation.head()