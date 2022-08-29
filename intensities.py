"Investigation of K_alpha1 and K_alpha2 peaks"

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm


a = 4.353075 # in angstrom
c = 10.957564 # in angstrom
theta2 = np.array([75.05, 74.53, 65.19, 64.82, 63.73, 63.22, 58.41, 57.97, 56.63, 56.20, 61.84, 61.43]) # in degrees
omega = np.array([17.41,17.41, 19.41, 19.41,19.41,19.41,25.41, 25.41,55.41,55.41, 49.41,49.41]) # in degree
int = np.array([89, 126, 106, 313, 171, 291, 345, 660, 501, 530, 355, 641])
h = np.array([2.009, 1.963, 1.001, 0.965, 0.932, 0.896, -0.005, -0.026, -3.019, -3.016, -2.953, -2.957])
k = np.array([-5.048, -5.021, -4.027, -4.004, -5.048, -5.017, -3.018, -2.997, -2.995, -2.960, -2.012, -1.994])
l = np.array([13.151, 13.094, 13.143, 13.054, 10.142, 10.064, 13.104, 13.021, 10.047, 9.977, 13.121, 13.018])
d_hkl = a/(h**2 + k**2 + (a/c)**2 * l**2) # in angstrom

def lambda_bragg(d, theta2): # assume n=1
    """l = lambda"""
    l = 2*np.sin(np.radians(theta2))*d
    print("lambda = {}".format(l))

lambda_bragg(d_hkl, theta2)

def gaussian(theta, lamb, sigma):
    gauss = (1 / (2 * np.pi * sigma)) * np.exp(-((theta) ** 2 + (lamb) ** 2) / (2 * sigma))
    return gauss


def convolute2d(matrix, sigma):
    """Faltung eines 2d-Datensatzes mit Gauß-peaks"""
    boundary = 0.5
    x_conv, y_conv = np.meshgrid(np.arange(-boundary,  boundary, 0.1), np.arange(-boundary,  boundary, 0.1)) # np.meshgrid(np.arange(-boundary,  boundary, 0.1), np.arange(- boundary,  boundary, 0.1))
    print(x_conv, y_conv)
    g = gaussian(x_conv , y_conv, sigma)
    convolved = scipy.signal.convolve2d(matrix, g,  boundary='wrap', mode='same') #boundary='symm', mode='same')
    plt.imshow(convolved, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()

# I = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
#             [0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]] # nxn, n=10
# sigma=0.5
# convolute2d(np.asarray(I), sigma=sigma)
