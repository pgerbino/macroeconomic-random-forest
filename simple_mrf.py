from pandas import DataFrame
from MacroRandomForest import *
import numpy as np


def run_model(df: DataFrame, independent_var, dependent_vars):

    y_pos = df.columns.get_loc(independent_var)
    x_pos = [df.columns.get_loc(x) for x in dependent_vars]
    # hard coded 12 quarters - 3 years
    oos_pos = np.arange(len(df) - (12), len(df))

    y_pos = np.atleast_1d(y_pos)
    x_pos = np.atleast_1d(x_pos)
    oos_pos = np.atleast_1d(oos_pos)

    # TODO hard coded number of trees
    number_of_trees = 100

    MRF = MacroRandomForest(data = df,
                            y_pos = y_pos,
                            x_pos = x_pos,
                            # S_pos = S_pos,
                            B = number_of_trees, 
                            parallelise = True,
                            n_cores = -1,
                            resampling_opt = 2, 
                            # see _process_subsampling_selection in MRF
                            oos_pos = oos_pos,
                            trend_push = 4,
                            quantile_rate = 0.3,
                            print_b = True,
                            fast_rw = True,
                            keep_forest = True,)    