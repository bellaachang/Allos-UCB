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


def sum_over_var(df: pd.DataFrame, sum_over: list, queries: list) -> float:
    """
    `sum_over` contains a set of the variables to sum over, vars and conds are similar to the previous function
    `queries` are pairs of dictionaries (vars, conds), where `vars` are the variables and `conds` the conditions
    
    If a summing variable wants to be included in a query, use a dummy value:
        eg. if we want \sum_{x} P(y = 1 | x, z = 1)P(z = 1), queries = 
            [({"y": 1}, {"x": -1, "z": 1}), ({"z": 1}, None)]
        where -1 can be any value (it will be overwritten)
    """

    # Get unique values of relevant
    ordered_sum_uniques = []

    for sum_value in sum_over:
        unique_values = df[sum_value].unique()
        ordered_sum_uniques.append(unique_values)
    
    cart_sum_values = cartesian_product(*ordered_sum_uniques)

    prob_sum = 0.
    for sum_values in cart_sum_values:
        sum_val_map = {}
        for i in range(len(sum_over)):
            sum_val_map.update({sum_over[i]: sum_values[i]})

        internal_product = 1.
        for (vars, conds) in queries:
            for sum_val in sum_val_map.keys():
                if sum_val in vars:
                    vars.update({sum_val: sum_val_map[sum_val]})
                if conds is not None and sum_val in conds:
                    conds.update({sum_val: sum_val_map[sum_val]})

            internal_product *= cond_prob(df, vars, conds)

        prob_sum += internal_product

    return prob_sum
