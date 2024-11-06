# Function to detrend a series
from pandas import DataFrame
from scipy import stats, signal
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def detrend_series(series):
    detrended = signal.detrend(series)
    return pd.Series(detrended, index=series.index)

def preprocess(df: DataFrame) -> DataFrame:
    # cannot display long column names
    if 'government_expenditure' in df.columns:
        df.rename(columns={'government_expenditure': 'ge'}, inplace=True)
    if 'balance_of_payments' in df.columns:
        df.rename(columns={'balance_of_payments': 'bop'}, inplace=True)
    if 'Rate' in df.columns:
        df.rename(columns={'Rate': 'rate'}, inplace=True)

    # Calculate the growth rates for 'gdp', 'ge', and 'bop' columns
    if 'gdp' in df.columns:
        df['gdp_growth'] = df['gdp'].pct_change().fillna(0)
    if 'ge' in df.columns:
        df['ge_growth'] = df['ge'].pct_change().fillna(0)
    if 'bop' in df.columns:
        df['bop_growth'] = df['bop'].pct_change().fillna(0)

    if 'gdp' in df.columns:
        df['gdp'] = detrend_series(df['gdp'])
    if 'ge' in df.columns:
        df['ge'] = detrend_series(df['ge'])
    if 'bop' in df.columns:
        df['bop'] = detrend_series(df['bop'])
    if 'gdp_growth' in df.columns:
        df['gdp_growth'] = detrend_series(df['gdp_growth'])
    if 'ge_growth' in df.columns:
        df['ge_growth'] = detrend_series(df['ge_growth'])
    if 'bop_growth' in df.columns:
        df['bop_growth'] = detrend_series(df['bop_growth'])

    scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    # there is a problem with the data - the dependent variable should be the first column
    # Move the 'gdp' column to the first position
    gdp_column = df.pop('gdp_growth')
    df.insert(0, 'gdp_growth', gdp_column)

    ### Dependent Variable
    my_var = "gdp_growth"
    y_pos = df.columns.get_loc(my_var)

    ### Exogenous Variables
    S_vars = df.columns.tolist()
    S_vars.remove(my_var)
    S_pos = [df.columns.get_loc(s) for s in S_vars]

    ### Variables Included in Linear Equation
    # changed from rate to balance of payments
    # ge - government expenditure is a straight line
    x_vars = ['inflation', 'ge_growth', 'rate', 'employment','bop_growth']
    # x_vars = ['inflation', 'Rate']
    x_pos = [df.columns.get_loc(x) for x in x_vars]
    oos_pos = np.arange(len(df) - (12) , len(df)) 