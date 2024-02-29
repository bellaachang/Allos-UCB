import numpy as np
import pandas as pd

# pandas complains whenever you mask a dataframe without reindexing
import warnings
warnings.simplefilter(action='ignore', category=UserWarning) 

"""
Some utilities for calculating conditional probabilities from a pandas dataframe.
Created by Athan Massouras
"""

def cond_prob(df: pd.DataFrame, vars: dict, conds: dict = None) -> float:
    """
    Calculates the joint probability of the variables in `vars` being equal to some number, given the conditions in `cond`.
    
    eg. if we want to calculate P(x = 0, y = 1 | z1 = 2, z2 = 3), we would have vars = {"x": 0, "y": 1} and cond = {"z1": 2, "z2", 3}
    
    `df` is a pandas dataframe with all the revelant variables; the titles of the columns of the data frame must exactly match
    the keys of `vars` and `conds`.
    """

    # Mask for conditions
    cond_mask = pd.Series(np.repeat(True, df.size))

    if conds is not None:
        for cond in conds.keys():
            try:
                cond_mask = cond_mask & (df[cond] == conds[cond])
            except KeyError:
                print(f"Invalid key: {cond}. Variables and conditions must have the same names as columns in df.")

    # Mask for variables   
    var_mask = cond_mask.copy()

    for var in vars.keys():
        try:
            var_mask = var_mask & (df[var] == vars[var])
        except KeyError:
            print(f"Invalid key: {cond}. Variables and conditions must have the same names as columns in df.")

    result = df[var_mask].size / df[cond_mask].size
    assert 0 <= result, "Calculation failed: probability is negative."
    assert 1 >= result, "Calculation failed: probability is greater than 1."
    return result


def cartesian_product(*arrays):
    """
    Calculate the Cartesian product of np.arrays
    Taken from: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def sum_over_var(df: pd.DataFrame, sum_over: set, vars: dict, conds: dict = None) -> float:
    """
    `sum_over` contains a set of the variables to sum over, vars and conds are similar to the previous function
    """

    # Get unique values of relevant
    ordered_sum_values = []
    ordered_sum_uniques = []
    for sum_value in sum_over:
        unique_values = df[sum_value].unique()
        ordered_sum_values.append(sum_value)
        ordered_sum_uniques.append(unique_values)
    
    cart_sum_values = cartesian_product(*ordered_sum_uniques)

    prob_sum = 0.
    for sum_values in cart_sum_values:
        sum_conds = {}
        for (i, val) in enumerate(sum_values):
            sum_conds[ordered_sum_values[i]] = val

        sum_conds.update(conds)

        prob_sum += cond_prob(df, vars, sum_conds)

    return prob_sum
