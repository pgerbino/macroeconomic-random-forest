# %% [markdown]
# This is a simple notebook that replicates https://mrf-web.readthedocs.io/en/latest/usage.html#usage

# %%
# needs correcting as the documentation a from MRF import *
from MacroRandomForest import *
import matplotlib.pyplot as plt
from scipy import stats, signal

# %%
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
n = 1000

# Create normally distributed data for each column
data = {
    'sim_y': np.random.normal(0, 1, n),
    'sim_x1': np.random.normal(0, 1, n),
    'sim_x2': np.random.normal(0, 1, n),
    'sim_x3': np.random.normal(0, 1, n),
    'sim_x4': np.random.normal(0, 1, n),
    'sim_x5': np.random.normal(0, 1, n),
    'sim_x6': np.random.normal(0, 1, n),
    'sim_x7': np.random.normal(0, 1, n),
    'sim_x8': np.random.normal(0, 1, n),
    'sim_x9': np.random.normal(0, 1, n),
    'sim_x10': np.random.normal(0, 1, n),
    'sim_x11': np.random.normal(0, 1, n),
    'sim_x12': np.random.normal(0, 1, n),
    'sim_x13': np.random.normal(0, 1, n),
    'sim_x14': np.random.normal(0, 1, n),
    'sim_x15': np.random.normal(0, 1, n),
    'trend': np.random.normal(0, 1, n)
}

# Create the DataFrame
simulated_data = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(simulated_data.head())

# Display basic information about the DataFrame
print(simulated_data.info())


# %%
simulated_data = pd.read_csv('./csvs/final_dataset.csv', index_col=0)
simulated_data.to_clipboard()


# %%
from sklearn.preprocessing import MinMaxScaler

simulated_data = pd.read_csv('./csvs/final_dataset.csv', index_col=0)
# simulated_data = simulated_data.tail(100)
# what is the purpose of trend?
# simulated_data['trend'] = np.arange(1, len(simulated_data) + 1)
# simulated_data.reset_index(drop=True, inplace=True)

# Function to detrend a series
def detrend_series(series):
    detrended = signal.detrend(series)
    return pd.Series(detrended, index=series.index)

# cannot display long column names
simulated_data.rename(columns={'government_expenditure': 'ge'}, inplace=True)
simulated_data.rename(columns={'balance_of_payments': 'bop'}, inplace=True)
simulated_data.rename(columns={'Rate': 'rate'}, inplace=True)

# Calculate the growth rates for 'gdp', 'ge', and 'bop' columns
simulated_data['gdp_growth'] = simulated_data['gdp'].pct_change().fillna(0)
simulated_data['ge_growth'] = simulated_data['ge'].pct_change().fillna(0)
simulated_data['bop_growth'] = simulated_data['bop'].pct_change().fillna(0)

# Detrend the 'gdp' column

simulated_data['gdp'] = detrend_series(simulated_data['gdp'])
simulated_data['ge'] = detrend_series(simulated_data['ge'])
simulated_data['bop'] = detrend_series(simulated_data['bop'])
simulated_data['gdp_growth'] = detrend_series(simulated_data['gdp_growth'])
simulated_data['ge_growth'] = detrend_series(simulated_data['ge_growth'])
simulated_data['bop_growth'] = detrend_series(simulated_data['bop_growth'])
# normalization 
# simulated_data = simulated_data.apply(stats.zscore)
scaler = MinMaxScaler()
# simulated_data = pd.DataFrame(scaler.fit_transform(simulated_data), columns=simulated_data.columns)
simulated_data = pd.DataFrame(scaler.fit_transform(simulated_data), columns=simulated_data.columns)
# there is a problem with the data - the dependent variable should be the first column
# Move the 'gdp' column to the first position
gdp_column = simulated_data.pop('gdp_growth')
simulated_data.insert(0, 'gdp_growth', gdp_column)

### Dependent Variable
my_var = "gdp_growth"
y_pos = simulated_data.columns.get_loc(my_var)

### Exogenous Variables
# S_vars = [f"sim_x{i}" for i in range(1, 16)] + ['trend']
S_vars = simulated_data.columns.tolist()
S_vars.remove(my_var)
S_pos = [simulated_data.columns.get_loc(s) for s in S_vars]

### Variables Included in Linear Equation
# changed from rate to balance of payments
# ge - government expenditure is a straight line
x_vars = ['inflation', 'ge_growth', 'rate', 'employment','bop_growth']
# x_vars = ['inflation', 'Rate']
x_pos = [simulated_data.columns.get_loc(x) for x in x_vars]
oos_pos = np.arange(len(simulated_data) - (12) , len(simulated_data)) # lower should be oos start, upper the length of your dataset

# %%


# %%
y_pos = np.atleast_1d(y_pos)
x_pos = np.atleast_1d(x_pos)
S_pos = np.atleast_1d(S_pos)
oos_pos = np.atleast_1d(oos_pos)

number_of_trees = 100

MRF = MacroRandomForest(data = simulated_data,
                        y_pos = y_pos,
                        x_pos = x_pos,
                        # S_pos = S_pos,
                        B = number_of_trees, # number of trees
                        parallelise = True,
                        n_cores = -1,
                        resampling_opt = 2, 
                        # see _process_subsampling_selection in MRF
                        oos_pos = oos_pos,
                        trend_push = 4,
                        quantile_rate = 0.3,
                        print_b = True,
                        fast_rw = True,
                        keep_forest=True,)

# %%
MRF_output = MRF._ensemble_loop()

# %%
plt.figure(figsize=(14, 7))
for tree in MRF_output['model']['forest']:
    plt.plot(MRF_output['model']['forest'][tree], alpha=0.3)
plt.title('Values of Each Tree in the Forest')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
 
MRF_output['pred_ensemble']

plt.figure(figsize=(14, 7))
for i in range(200):
    plt.plot(MRF_output['pred_ensemble'][i], alpha=0.3)
plt.title('100 Ensembles Predictions')
plt.xlabel('Time')
plt.ylabel('Predicted Value')
plt.show()

# %%
forecasts = MRF_output['pred']
betas = MRF_output['betas']


fig, ax = plt.subplots()
plt.rcParams['figure.figsize'] = (20, 8)

# Plotting actual versus original
# column names has dissapeared

ax.plot(simulated_data.iloc[oos_pos, 0].shift(1), label='Actual', linewidth=3, color='mediumseagreen', linestyle='--')
ax.plot(forecasts, color='lightcoral', linewidth=3, label='MRF Ensemble')

ax.legend(fontsize = 15)
ax.set_ylabel("Value", fontsize = 15)
ax.grid()
ax.set_xlabel(r"$t$", fontsize = 16)
ax.set_title("OOS predictions of MRF", fontsize = 15)

# %%
MRF.band_plots()

# %%
simulated_data = pd.read_csv('final_dataset.csv', index_col=0)
fig, ax1 = plt.subplots()

# Plot gdp on the primary y-axis
ax1.plot(simulated_data.index, simulated_data['gdp'], color='blue', label='GDP')
ax1.set_xlabel('Index')
ax1.set_ylabel('GDP', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis and plot ge
ax2 = ax1.twinx()
ax2.plot(simulated_data.index, simulated_data['government_expenditure'], color='red', label='Government Expenditure (GE)')
ax2.set_ylabel('GE', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add a title and show the plot
plt.title('GDP and Government Expenditure (GE) with Different Y-Axes')
fig.tight_layout()
plt.show()


