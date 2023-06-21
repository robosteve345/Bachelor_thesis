#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:52:02 2023

@author: stevengebel
"""
"""XRD Simulation of a modulated bcc lattice with monoatomic basis"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)

"""K-space creator"""
k0, l0, kmax, lmax = 0, 0, 2, 2  #  boundaries of K and L for the intensity maps
deltak = 0.1  #  or 0.1, k-space point distance
h = 1  #  H value in k space
k = np.arange(k0-kmax, k0+kmax + deltak, deltak)
l = np.arange(l0-lmax, l0+lmax + deltak, deltak)
k2d, l2d = np.meshgrid(k, l)
Unitary = np.ones((len(k2d), len(l2d)))  # Unitary matrix


def translation(x, storage_x, storage_y, storage_z, i):
    """
    Translate atomic positions of X_x position
    """
    x_transl = x + np.array([0, 0, i])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


"""Atomic positions"""
x_Atom1, y_Atom1, z_Atom1 = [0], [0], [0]
x_Atom2, y_Atom2, z_Atom2 = [0.5], [0.5], [0.5]
"""Modulation of BCC: commensurate with q_CDW = 0.2*2*pi/a of atom 2 (1/2, 1/2, 1/2)"""
A = 1
q_cdw = 0.2 # in 2pi/a
# Full translation of Atom1 & Atom2 for 6 unit cells
for i in range(1,len(q_cdw**(-1))+1):
        translation(np.array([x_Atom1[0], y_Atom1[0], z_Atom1[0]]), x_Atom1, y_Atom1, z_Atom1, i)  # Atom1
        translation(np.array([x_Atom2[0], y_Atom2[0], z_Atom2[0]]), x_Atom2, y_Atom2, z_Atom2, i) # Atom2

for i in range(1, 6):
    print(i)
    print(np.sin(0.2*2*np.pi*i))
    Atom2_list[2].append(i + A*np.sin(q_cdw*2*np.pi*i))
    Atom2_list[0].append(Atom1_list[0])
    Atom2_list[1].append(Atom1_list[1])
    Atom1_list[2].append(i)
    Atom1_list[0].append(Atom1_list[0])
    Atom1_list[1].append(Atom1_list[1])
print("Atom1_list={}, Atom2_list={}".format(np.asarray(Atom1_list)[0,0], np.asarray(Atom2_list)[-1,-1]))
plt.plot(1/2*np.ones(6) + np.arange(0, 6, 1), np.ones(6), label='equilibrium', ls='', marker='x')
plt.xlabel('Unit cell')
plt.ylabel('z')
plt.plot(Atom2_list[2], np.ones(6), label='distorted', ls='', marker='o')
plt.legend()
plt.show()

# Final atomic positions
Atom1, Atom2 = np.array([x_Atom1, y_Atom1, z_Atom1]), np.array([x_Atom2, y_Atom2, z_Atom2]),

"""Scattering amplitudes F"""
# Form factors
f_Atom1, f_Atom2, f_Atom3, f_Atom4 = 1, 1, 1, 1
F_Atom1_list, F_Atom2_list = [],[]
# Scattering Amplitudes
for i in range(1):  # 2 atoms for each sort, for Eu with 1 position and 2 for each al and ga with translation
    F_Atom1_list.append(
        f_Atom1  * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Atom1[0, i] + k2d * Eu[1, i] + l2d * Eu[2, i]))
    )
F = np.sum(F_Atom1) + np.sum(F_Atom2) # + F_Atom3 + F_Atom4# + 0.1*np.random.rand(len(k2d), len(k2d)) # + F_Ga + F_Al
"""Intensity I"""
I = np.abs(np.round(F, 3))**2 # I \propto F(Q)^2, F complex
##############################################################################
# Excluding unallowed K-points (ONLY FOR deltak=/1)
k_intlist = np.arange(0,len(k2d), 1) # erstelle indices aller k-Werte
for i in range(0, 2*kmax + 1): # for kmax=lmax=2 and ∆k=0.1, up to 2*kmax=lmax + 1
    k_intlist = np.delete(k_intlist, i*9) # 9 for ∆k=0.1 and kmax=lmax=2, 99 for ∆k=0.01 and kmax=lmax=2
for i in k_intlist: # Set unallowed K-values for intensities to 0
    I[:, i] = 0
# Exluding unallowed L-points (ONLY FOR deltak=0.01 or deltak=0.001)
l_intlist = np.arange(0,len(l2d), 1) # erstelle indices aller l-Werte
if deltak == 0.1:
    for i in range(0, 2*kmax + 1):
        l_intlist = np.delete(l_intlist, i*9)  # Lösche jeden zehnten index
    for i in l_intlist: # Set unallowed L-values for intensities to 0
        I[i,:] = 0
else:
    for i in range(0, 2*kmax*10 + 1):
        l_intlist = np.delete(l_intlist, i*9)  # Lösche jeden zehnten index
    for i in l_intlist: # Set unallowed L-values for intensities to 0
        I[i,:] = 0
##############################################################################

# Noise
noisefactor = 0.0 # Amplitude of the noise for the intensity
I = I + noisefactor*np.random.rand(len(k2d), len(k2d)) # Add random noise with maximum 1

# Plotting
figure(figsize=(9,7), dpi=100)
plt.suptitle("Body centered cubic (bcc), F($\mathbf{Q}$)=$f(1 + e^{-i\pi(h+k+l)})$ \n $I=f^2(2+2\cos(\pi(h+k+l))), f=1$")
# I_fcc = f^2(4 + 2*(\cos(\pi(h+l)) + \cos(\pi(h-l)) + \cos(\pi(k+h)) + \cos(\pi(k-h))".format(h))
plt.subplot(2, 2, 1)
plt.title('Countourplot')
plt.contourf(l2d, k2d, I, cmap='viridis', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax))
plt.colorbar()
plt.xlabel("K(rlu)")
plt.ylabel("L(rlu)")

plt.subplot(2, 2, 2)
plt.title("Gaussian interpolation")
plt.imshow(I, cmap='viridis',
            interpolation='gaussian',
            extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax),
            origin='lower'
            #norm=LogNorm(vmin=0.1, vmax=np.max(I))
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
plt.plot(l2d[:,0], I[:,0], ls='--', marker='.', label='K={}'.format(np.round(k[0], 2)))
plt.plot(l2d[:,0], I[:,1], ls='--', marker='.', label='K={}'.format(np.round(k[1], 2)))
plt.plot(l2d[:,0], I[:,-7], ls='--', marker='.', label='K={}'.format(np.round(k[-7], 2)))
plt.plot(l2d[:,0], I[:,-4], ls='--', marker='.', label='K={}'.format(np.round(k[-4], 2)))
plt.legend(loc='lower center')
plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
plt.xlabel("L(rlu)")
# plt.savefig("BCC_H={}.jpg".format(h), dpi=300)
plt.subplots_adjust(wspace=0.3)
plt.show()

