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

bank_rate = pd.read_csv(r'Bank Rate history and data Bank of England Database.csv', index_col="Date Changed")
bank_rate.index = pd.to_datetime(bank_rate.index, format='mixed')
bank_rate.index = bank_rate.index.to_period('M').to_timestamp()
bank_rate = bank_rate[~bank_rate.index.duplicated(keep='first')]
bank_rate = bank_rate.resample('MS').ffill()


inflation_government_exp_bank_df = inflation_government_exp_df.join(bank_rate, how='inner')

employment = pd.read_csv('./employment.csv', index_col=0, skiprows=8)
employment = employment[employment.index.str.match(r'^\d{4} [A-Z]{3}$')]
employment.columns = ['employment']
employment.index = pd.to_datetime(employment.index, format='mixed')
employment.index.name = 'date'

inflation_government_exp_bank_emp_df = inflation_government_exp_bank_df.join(employment, how='inner')


balance_of_payments = pd.read_csv('./balance_of_payments.csv', index_col=0, skiprows=8)
balance_of_payments = balance_of_payments[balance_of_payments.index.str.match(r'^\d{4} Q[1-4]$')]
balance_of_payments.columns = ['balance_of_payments']
balance_of_payments.index = balance_of_payments.index.str.replace(r'Q1', '01-01')
balance_of_payments.index = balance_of_payments.index.str.replace(r'Q2', '04-01')
balance_of_payments.index = balance_of_payments.index.str.replace(r'Q3', '07-01')
balance_of_payments.index = balance_of_payments.index.str.replace(r'Q4', '10-01')
balance_of_payments.index = pd.to_datetime(balance_of_payments.index, format='mixed')
balance_of_payments.index.name = 'date'
balance_of_payments = balance_of_payments.resample('MS').ffill()

not_quite_final_df = inflation_government_exp_bank_emp_df.join(balance_of_payments, how='inner')

gdp = pd.read_csv('./gdp.csv', index_col=0, skiprows=8)
gdp = gdp[gdp.index.str.match(r'^\d{4} Q[1-4]$')]
gdp.columns = ['gdp']
gdp.index = gdp.index.str.replace(r'Q1', '01-01')
gdp.index = gdp.index.str.replace(r'Q2', '04-01')
gdp.index = gdp.index.str.replace(r'Q3', '07-01')
gdp.index = gdp.index.str.replace(r'Q4', '10-01')
gdp.index = pd.to_datetime(gdp.index, format='mixed')
gdp.index.name = 'date'
gdp = gdp.resample('MS').ffill()

final_df = not_quite_final_df.join(gdp, how='inner')

final_df.to_csv('./final_dataset.csv')