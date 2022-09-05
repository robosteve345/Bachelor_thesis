import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

#
# I = np.zeros((100, 100))
#
# f_result = np.zeros(800)
# t2 = np.arange(0, 80, 0.1)
# multiple = 1
#
# for i in range(len(h)):
#     hkl = [h[i], k[i], l[i]]
#     thkl = np.transpose(hkl)
#     dk = 1 / ((np.matmul(np.matmul(invG, thkl), hkl)) ** (1 / 2))
#     sinthetak = lamb / (2 * dk)
#     thetak = np.rad2deg(np.arcsin(sinthetak))
#
#     Fna = f_func(r_na, h[i], k[i], l[i], param_na, sinthetak / lamb)
#     F = Fna
#
#     if (f_result[int(round(2 * thetak, 0) * 10)] != 0):
#         f_result[int(round(2 * thetak, 0) * 10)] = f_result[int(round(2 * thetak, 0) * 10)] + abs(F) ** 2
#     else:
#         f_result[int(round(2 * thetak, 0) * 10)] = abs(F) ** 2
#
#     if (np.dot(hkl, incident) == 0):
#         I[49 - hkl[0] * 10][49 + hkl[1] * 10] = abs(F) ** 2

# t, l = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
# g = gaussian(t, l)
# N_neighbors = scipy.signal.convolve2d(I, g, boundary='wrap', mode='same')
# plt.imshow(N_neighbors, interpolation="nearest", cmap=plt.cm.gray)
E = np.linspace(-0.2, 0.2, 1000)
T = np.array([0, 50, 100, 200])
kB = 1.38e-23

def fermidirac(T, E):
    """
    :param T: 
    :param E: 
    :return: 
    """
    return 1 / (np.exp( E*1.6e-19/(kB*T)) + 1)


fig = plt.figure(figsize=(8,6))
for i in T:
    plt.plot(E, fermidirac(i, E), label='{}K'.format(i), markersize=10)
    plt.ylabel(r'f($\epsilon$)', fontsize=23)
    plt.tick_params(axis='y', labelsize=20, direction='in')
    plt.tick_params(axis='x', labelsize=20, direction='in')
    plt.xlabel(r'$\epsilon$ - $\epsilon_F$', fontsize=23)
    plt.legend(fontsize=22)

plt.savefig("fermidirac.jpg", dpi=300)
plt.show()