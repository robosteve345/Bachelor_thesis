#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:52:02 2023

@author: stevengebel
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)

def maincubic():
    """XRD Simulation"""
    # F_fcc = f(1 + e^{-i\pi(h+l) + e^{-\pi(h+k) + e^{-\pi(k+l)} })
    # I_fcc = f^2(4 + 2*(\cos(\pi(h+l)) + \cos(\pi(h-l)) + \cos(\pi(k+h)) + \cos(\pi(k-h))
    # +\cos(\pi(k+l)) +\cos(\pi(k-l))) )
    # F_bodycc = f(1 + e^{-i\pi(h+k+l)})
    # I_bodycc = f^2(2+2\cos(\pi(h+k+l)))
    # F_basecc = f(1 + e^{-i\pi(h+k)})
    # I_basecc = f^2(2+2\cos(\pi(h+k)))

    """K-space creator"""
    k0, l0, kmax, lmax = 0, 0, 2, 2  # boundaries of K and L for the intensity maps
    deltak = 0.1  # or 0.1, k-space point distance
    h = 1  # H value in k space
    k = np.arange(k0 - kmax, k0 + kmax + deltak, deltak)
    l = np.arange(l0 - lmax, l0 + lmax + deltak, deltak)
    k2d, l2d = np.meshgrid(k, l)
    Unitary = np.ones((len(k2d), len(l2d)))  # Unitary matrix

    """Atomic positions
    Base centered cubic"""
    # Atom1, Atom2 = np.array([[0],[0],[0]]), np.array([[0],[1/2],[1/2]])
    """Body centered cubic ("full cell")"""
    Atom1, Atom2 = np.array([[0], [0], [0]]), np.array([[1 / 2], [1 / 2], [1 / 2]])
    # , Atom3, Atom4, Atom5, Atom6, Atom7, Atom8, Atom9 = np.array([[0],[1],[1]]), np.array([[1],[0],[1]]),\
    # np.array([[1],[1],[1]]), np.array([[1],[0],[0]]), np.array([[1],[1],[0]]), np.array([[0],[0],[1]]), np.array([[0],[1],[0]])
    """Face centered cubic (fcc)"""
    # Atom1, Atom2, Atom3, Atom4 = np.array([[0],[0],[0]]), np.array([[1/2],[1/2],[0]]), \
    #    np.array([[0],[1/2],[1/2]]), np.array([[1/2],[0],[1/2]])
    """Simple cubic"""
    # Atom1 = np.array([[0],[0],[0]])
    """Scattering amplitudes F"""
    # Form factors
    f_Atom1, f_Atom2 = 1, 1
    # f_Atom3, f_Atom4, f_Atom5, f_Atom6, f_Atom7, f_Atom8, f_Atom9= 1, 1, 1, 1, 1, 1, 1, 1, 1
    # Scattering Amplitudes
    F_Atom1 = f_Atom1 * np.exp(-2 * np.pi * 1j * (h * Unitary * Atom1[0] + k2d * Atom1[1] + l2d * Atom1[2]))
    F_Atom2 = f_Atom2 * np.exp(-2 * np.pi * 1j * (h * Unitary * Atom2[0] + k2d * Atom2[1] + l2d * Atom2[2]))
    # F_Atom3 = f_Atom3 * np.exp(-2*np.pi*1j*(h*Unitary*Atom3[0] + k2d*Atom3[1] + l2d*Atom3[2]))
    # F_Atom4 = f_Atom4 * np.exp(-2*np.pi*1j*(h*Unitary*Atom4[0] + k2d*Atom4[1] + l2d*Atom4[2]))
    # F_Atom5 = f_Atom5 * np.exp(-2*np.pi*1j*(h*Unitary*Atom5[0] + k2d*Atom5[1] + l2d*Atom5[2]))
    # F_Atom6 = f_Atom6 * np.exp(-2*np.pi*1j*(h*Unitary*Atom6[0] + k2d*Atom6[1] + l2d*Atom6[2]))
    # F_Atom7 = f_Atom7 * np.exp(-2*np.pi*1j*(h*Unitary*Atom7[0] + k2d*Atom7[1] + l2d*Atom7[2]))
    # F_Atom8 = f_Atom8 * np.exp(-2*np.pi*1j*(h*Unitary*Atom8[0] + k2d*Atom8[1] + l2d*Atom8[2]))
    # F_Atom9 = f_Atom9 * np.exp(-2*np.pi*1j*(h*Unitary*Atom9[0] + k2d*Atom9[1] + l2d*Atom9[2]))
    F = F_Atom1 + F_Atom2  # + F_Atom3 + F_Atom4 + F_Atom5 + F_Atom6 + F_Atom7 + F_Atom8 + F_Atom9# + 0.1*np.random.rand(len(k2d), len(k2d)) # + F_Ga + F_Al
    """Intensity I"""
    I = np.abs(np.round(F, 3)) ** 2  # I \propto F(Q)^2, F complex
    ##############################################################################
    # # Excluding unallowed K-points (ONLY FOR deltak=/1)
    k_intlist = np.arange(0, len(k2d), 1)  # erstelle indices aller k-Werte
    for i in range(0, 2 * kmax + 1):  # for kmax=lmax=2 and ∆k=0.1, up to 2*kmax=lmax + 1
        k_intlist = np.delete(k_intlist, i * 9)  # 9 for ∆k=0.1 and kmax=lmax=2, 99 for ∆k=0.01 and kmax=lmax=2
    for i in k_intlist:  # Set unallowed K-values for intensities to 0
        I[:, i] = 0
    # # Exluding unallowed L-points (ONLY FOR deltak=0.01 or deltak=0.001)
    # l_intlist = np.arange(0, len(l2d), 1)  # erstelle indices aller l-Werte
    # if deltak == 0.1:
    #     for i in range(0, 2 * kmax + 1):
    #         l_intlist = np.delete(l_intlist, i * 9)  # Lösche jeden zehnten index
    #     for i in l_intlist:  # Set unallowed L-values for intensities to 0
    #         I[i, :] = 0
    # else:
    #     for i in range(0, 2 * kmax * 10 + 1):
    #         l_intlist = np.delete(l_intlist, i * 9)  # Lösche jeden zehnten index
    #     for i in l_intlist:  # Set unallowed L-values for intensities to 0
    #         I[i, :] = 0
    ##############################################################################

    # Noise
    noisefactor = 0.0  # Amplitude of the noise for the intensity
    I = I + noisefactor * np.random.rand(len(k2d), len(k2d))  # Add random noise with maximum 1

    # Plotting
    figure(figsize=(9, 7), dpi=100)
    plt.suptitle(
        "Body centered cubic (bcc), F($\mathbf{Q}$)=$f(1 + e^{-i\pi(h+k+l)})$ \n $I=f^2(2+2\cos(\pi(h+k+l))), f=1$")
    # I_fcc = f^2(4 + 2*(\cos(\pi(h+l)) + \cos(\pi(h-l)) + \cos(\pi(k+h)) + \cos(\pi(k-h))".format(h))
    plt.subplot(2, 2, 1)
    plt.title('Countourplot')
    plt.contourf(l2d, k2d, I, cmap='viridis', extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax))
    plt.colorbar()
    plt.xlabel("K(rlu)")
    plt.ylabel("L(rlu)")
    plt.legend('H={}'.format(h), loc='upper center')

    plt.subplot(2, 2, 2)
    plt.title("Gaussian interpolation")
    plt.imshow(I, cmap='viridis',
               interpolation='gaussian',
               extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
               origin='lower'
               # norm=LogNorm(vmin=0.1, vmax=np.max(I))
               )
    plt.colorbar()
    plt.xlabel("K(rlu)")
    plt.ylabel("L(rlu)")

    plt.subplot(2, 2, 3)
    plt.scatter(k2d, l2d, c=I, s=I, cmap='viridis', label=r'$I \propto F(\mathbf{Q})^2$')
    plt.colorbar()
    plt.legend(loc='upper right')
    plt.ylabel("L(rlu)")
    plt.xlabel("K(rlu)")
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    # plt.title(r'$I(L)=f^2(4 + 2*(\cos(\pi(h+l)) + \cos(\pi(h-l)) + \cos(\pi(k+h)) + \cos(\pi(k-h))+\cos(\pi(k+l)) +\cos(\pi(k-l))), f=1$')
    plt.plot(l2d[:, 0], I[:, 0], ls='--', marker='.', label='K={}'.format(np.round(k[0], 2)))
    plt.plot(l2d[:, 0], I[:, 1], ls='--', marker='.', label='K={}'.format(np.round(k[1], 2)))
    plt.plot(l2d[:, 0], I[:, -1], ls='--', marker='.', label='K={}'.format(np.round(k[-1], 2)))
    plt.plot(l2d[:, 0], I[:, -4], ls='--', marker='.', label='K={}'.format(np.round(k[-4], 2)))
    # plt.plot(l2d[:,0], I[:,int(2/deltak)], ls='--', marker='.', label='K={}'.format(int(k0-kmax+2)))
    # plt.plot(l2d[:,0], I[:,int(3/deltak)], ls='--', marker='.', label='K={}'.format(int(k0-kmax+3)))
    plt.legend(loc='lower center')
    plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L(rlu)")
    # plt.savefig("BCC_9atoms_H={}.jpg".format(h), dpi=300)
    plt.subplots_adjust(wspace=0.3)
    plt.show()

if __name__ == '__main__':
    maincubic()

