# This is a simple notebook that replicates https://mrf-web.readthedocs.io/en/latest/usage.html#usage
# needs correcting as the documentation a from MRF import *
from MacroRandomForest import *
import matplotlib.pyplot as plt
# not included in the documentation
# great the file isn't included - i have a sinking feeling about this...
# import pandas as pd

# url='https://drive.google.com/file/d/1Sp_2HGdIY0y9m5htlskIbNqWo02FCGNb/view?usp=sharing'
# url='https://drive.google.com/uc?id=' + url.split('/')[-2]
# simulated_data = pd.read_csv(url, index_col = "index")
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

# # Display the first few rows of the DataFrame
# print(df.head())

# # Display basic information about the DataFrame
# print(df.info())

### Dependent Variable
my_var = "sim_y"
y_pos = simulated_data.columns.get_loc(my_var)

### Exogenous Variables
S_vars = [f"sim_x{i}" for i in range(1, 16)] + ['trend']
S_pos = [simulated_data.columns.get_loc(s) for s in S_vars]

### Variables Included in Linear Equation
x_vars = ["sim_x1", 'sim_x2', 'sim_x3']
x_pos = [simulated_data.columns.get_loc(x) for x in x_vars]
oos_pos = np.arange(len(simulated_data) - 50 , len(simulated_data)) # lower should be oos start, upper the length of your dataset
oos_pos

y_pos = np.atleast_1d(y_pos)
x_pos = np.atleast_1d(x_pos)
S_pos = np.atleast_1d(S_pos)
oos_pos = np.atleast_1d(oos_pos)

MRF = MacroRandomForest(data = simulated_data,
                        y_pos = y_pos,
                        x_pos = x_pos,
                        S_pos = S_pos,
                        B = 5, # just five trees or we'll be here a while
                        parallelise = False,
                        n_cores = 1,
                        resampling_opt = 2,
                        oos_pos = oos_pos,
                        trend_push = 4,
                        quantile_rate = 0.3,
                        print_b = True,
                        fast_rw = True)
MRF_output = MRF._ensemble_loop()

print(MRF_output)