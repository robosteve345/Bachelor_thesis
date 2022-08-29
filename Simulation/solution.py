import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fftn, ifftn
from matplotlib.colors import Normalize
import os
import sys
import random

fig = plt.figure()
hkl = []
# parameter
a = 4.35
b = a
c = 11.0
alpha = np.pi / 2
beta = alpha
ganma = alpha

lamb = 1.5418

sigma = 1 / 20

incident = [0, 0, 1]

B = 2.0

sinthetalamb = 0.2
param_na = [[4.7626, 3.285], [3.1736, 8.8422], [1.2674, 0.3136], [1.1128, 129.424]]
x = [0.0, 0.5, 1.0]
cp = 0.676

G = [[a ** 2, a * b * np.cos(ganma), a * c * np.cos(beta)], [b * a * np.cos(ganma), b ** 2, b * a * np.cos(alpha)],
     [c * a * np.cos(beta), c * b * np.cos(alpha), c ** 2]]
invG = np.linalg.inv(G)

r_na = []
df_na = pd.read_csv("./pos.csv", encoding="UTF-8")
x = list(df_na['x'])
y = list(df_na['y'])
z = list(df_na['z'])
for i in range(4):
    r_na.append([x[i], y[i], z[i]])

df = pd.read_csv("./hkl.csv", encoding="UTF-8")
h = list(df['h'])
k = list(df['k'])
l = list(df['l'])


def f_atm(param, sinthetalamb):
    fa = 0
    for i in range(3):
        fa = fa + param[i][0] * np.exp((-param[i][1] * (sinthetalamb ** 2)))

    fa = fa + cp
    fa = fa * np.exp(-B * (sinthetalamb) ** 2)
    return fa


def f_func(r, h, k, l, param, sinthetalamb):
    rea = 0
    ima = 0
    fatm = f_atm(param, sinthetalamb)
    for j in r:
        rea = rea + np.cos(2 * np.pi * (j[0] * h + j[1] * k + j[2] * l))
        ima = ima + np.sin(2 * np.pi * (j[0] * h + j[1] * k + j[2] * l))
        f = fatm * (complex(rea, ima))

    return f


def gaussian(theta, lamb):
    gauss = (1 / (2 * np.pi * sigma)) * np.exp(-(theta ** 2 + lamb ** 2) / (2 * sigma))
    return gauss


I = np.zeros((100, 100))

f_result = np.zeros(800)
t2 = np.arange(0, 80, 0.1)
multiple = 1

for i in range(len(h)):
    hkl = [h[i], k[i], l[i]]
    thkl = np.transpose(hkl)
    dk = 1 / ((np.matmul(np.matmul(invG, thkl), hkl)) ** (1 / 2))
    sinthetak = lamb / (2 * dk)
    thetak = np.rad2deg(np.arcsin(sinthetak))

    Fna = f_func(r_na, h[i], k[i], l[i], param_na, sinthetak / lamb)
    F = Fna

    if (f_result[int(round(2 * thetak, 0) * 10)] != 0):
        f_result[int(round(2 * thetak, 0) * 10)] = f_result[int(round(2 * thetak, 0) * 10)] + abs(F) ** 2
    else:
        f_result[int(round(2 * thetak, 0) * 10)] = abs(F) ** 2

    if (np.dot(hkl, incident) == 0):
        I[49 - hkl[0] * 10][49 + hkl[1] * 10] = abs(F) ** 2

t, l = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
g = gaussian(t, l)
N_neighbors = scipy.signal.convolve2d(I, g, boundary='wrap', mode='same')
plt.imshow(N_neighbors, interpolation="nearest", cmap=plt.cm.gray)