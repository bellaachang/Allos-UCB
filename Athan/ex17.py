import numpy as np
import pandas as pd

N = 10_000

# Set up random variables
z3_list = np.random.rand(N)
z3_list = np.vectorize(lambda x: 1 if x > 0.4 else 0)(z3_list)  # P(z3 = 1) = 0.6

x = []
y = []

for z3 in z3_list:
    if z3 == 0:
        x.append(1 if np.random.rand() > 0.7 else 0)   #P(x = 1|z3 = 0) = 0.3
        y.append(1 if np.random.rand() > 0.25 else 0)  #P(y = 1|z3 = 0) = 0.75
    else:
        x.append(1 if np.random.rand() > 0.3 else 0)   #P(x = 1|z3 = 1) = 0.7
        y.append(1 if np.random.rand() > 0.55 else 0)  #P(y = 1|z3 = 1) = 0.45

x_list = np.array(x)
y_list = np.array(y)

z1_list = []

for i in range(N):
    x = x_list[i]
    y = y_list[i]

    if x == 0 and y == 0:
        #P(z1 = 1|x = 0, y = 0) = 0.8:
        z1_list.append(1 if np.random.rand() > 0.2 else 0)
    elif x == 0 and y == 1:
        #P(z1 = 1|x = 0, y = 1) = 0.5
        z1_list.append(1 if np.random.rand() > 0.5 else 0)
    elif x == 1 and y == 0:
        #P(z1 = 1|x = 1, y = 0) = 0.55
        z1_list.append(1 if np.random.rand() > 0.45 else 0)
    else:
        #P(z1 = 1|x = 1, y = 1) = 0.9
        z1_list.append(1 if np.random.rand() > 0.1 else 0)

z1_list = np.array(z1_list)

df = pd.DataFrame({"z3": z3_list, "x": x_list, "y": y_list, "z1": z1_list})
df_z30 = df.mask(df["z3"] == 0)
df_z31 = df.mask(df["z3"] == 1)


# Calculate probabilities. In all cases, we calculate P(y = 1|x, z):
# Very provisional (I get that it's messy lol), just for the sake of understanding the exercises

# P(y | x = 0, z1 = 0), denote P_0 --> eq. 42
p_y1_x0_z10 = df_z30.mask((df["y"] == 1) & (df["x"] == 0) & (df["z1"] == 0)).shape[0] / df_z30.shape[0]
p_x0 = df_z30.mask(df["x"] == 0).shape[0] / df_z30.shape[0]
p_z30 = df_z30.shape[0] / N
p_y_z30 = p_y1_x0_z10 * p_z30 / p_x0

p_y1_x0_z10 = df_z31.mask((df["y"] == 1) & (df["x"] == 0) & (df["z1"] == 0)).shape[0] / df_z31.shape[0]
p_x0 = df_z31.mask(df["x"] == 0).shape[0] / df_z31.shape[0]
p_z31 = df_z31.shape[0] / N
p_y_z31 = p_y1_x0_z10 * p_z31 / p_x0

numerator = p_y_z30 + p_y_z31


p_z10_x0_z30 = df_z30.mask((df["x"] == 0) & (df["z1"] == 0)).shape[0] / df_z30.shape[0]
p_x0_z30 = df_z30.mask(df["x"] == 0).shape[0] / df_z30.shape[0]

p_z10_x0_z31 = df_z31.mask((df["x"] == 0) & (df["z1"] == 0)).shape[0] / df_z30.shape[0]
p_x0_z31 = df_z31.mask(df["x"] == 0).shape[0] / df_z30.shape[0]

denominator = (p_z10_x0_z30 / p_x0_z30) * p_z30 + (p_z10_x0_z31 / p_x0_z31) * p_z31

eq42 = numerator / denominator


# eq. 43
p_z1_x_y = df.mask((df["z1"] == 0) & (df["x"] == 0) & (df["y"] == 1)).shape[0] / N
p_x_y = df.mask((df["x"] == 0) & (df["y"] == 1)).shape[0] / N
numerator = df["y"].mean() * (p_z1_x_y / p_x_y)

eq43 = numerator / denominator
