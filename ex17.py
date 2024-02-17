import numpy as np

N = 10_000

z3 = np.random.rand(N)
z3 = np.vectorize(lambda x: 1 if x > 0.4 else 0)(z3)  # P(z3 = 1) = 0.6

x = []
y = []

for z in z3:
    if z == 0:
        x.append(1 if np.random.rand() > 0.7 else 0)   #P(x = 1|z3 = 0) = 0.3
        y.append(1 if np.random.rand() > 0.25 else 0)  #P(y = 1|z3 = 0) = 0.75
    else:
        x.append(1 if np.random.rand() > 0.3 else 0)   #P(x = 1|z3 = 1) = 0.7
        y.append(1 if np.random.rand() > 0.55 else 0)  #P(y = 1|z3 = 1) = 0.45
