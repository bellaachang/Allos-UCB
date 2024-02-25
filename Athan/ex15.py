import numpy as np
import pandas as pd
from misc_utils import *

vars = {"x": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 
        "y": [1, 0, 0, 0, 1, 1, 0, 0, 0, 0]}

df = pd.DataFrame(vars)

p_x_y0 = cond_prob(df, {"x": 0}, {"y": 0})  # P(x = 0 | y = 0)
p_x_y1 = cond_prob(df, {"x": 0}, {"y": 1})  # P(x = 0 | y = 1)
p_y_x0 = cond_prob(df, {"y": 0}, {"x": 0})  # P(y = 0 | x = 0)
p_y_x1 = cond_prob(df, {"y": 0}, {"x": 1})  # P(y = 0 | x = 1)

# What does A mean in this context?
