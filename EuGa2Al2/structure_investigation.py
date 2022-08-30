"""Single Crystal characterisation refinements analysis from BRUKER, comparison with STAVINOAH
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from Praktikum import lin_reg

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def stavinoah(z0, dz0, a, c, da, dc, compound=None, sample=None):
    alpha = np.sqrt(a**2/4 + (0.25 - z0)**2*c**2)
    d12 = alpha
    dd12 = np.sqrt( (-2/alpha*c**2*(0.25-z0) * dz0)**2 + (0.25*a/alpha * da)**2 + ((0.25-z0)*c*dc/alpha)**2 )
    d22 = (1 - 2*z0)*c
    dd22 = 2 * np.sqrt((z0*dc)**2 + (c*dz0)**2)
    theta = np.rad2deg(np.arccos((-0.25*a**2 + (0.25 - z0)**2 * c**2)/(0.25*a**2 + (0.25 - z0)**2*c**2)))
    dtheta = np.rad2deg(dz0 * (-((z0 - 0.25)/(z0**2 - z0/2 + 5/16)**2))/np.sqrt(1 - ((-0.25 + (0.25 - z0)**2)/(0.25 + (0.25 - z0)**2))**2))
    print("Comparison with STAVINOAH for {} sample {}:".format(compound, sample))
    print(r'd12 = ({} $\pm$ {})'.format(d12, dd12))
    print(r'd22 = ({} $\pm$ {})'.format(d22, dd22))
    print(r'theta = ({} $\pm$ {})'.format(theta, dtheta))

    return d12, dd12, d22, dd22, theta, dtheta


########################################
"""EuGa2Al2: Lets say 2 refinements (temp one included)"""
########################################
# Sample 1a SAFECALL
sinlambda1a, obs1a, cal1a, sigma1a, DIFsigma1a = np.loadtxt("euga2al21a.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# Sample 1c
sinlambda1c, obs1c, cal1c, sigma1c, DIFsigma1c = np.loadtxt("euga2al21c.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)

# Sample 1a4
sinlambda1a4, obs1a4, cal1a4, sigma1a4, DIFsigma1a4 = np.loadtxt("euga2al21a4.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)

# Sample 3b2
sinlambda3b2, obs3b2, cal3b2, sigma3b2, DIFsigma3b2 = np.loadtxt("euga2al23b2.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# Sample 4b2
sinlambda4b2, obs4b2, cal4b2, sigma4b2, DIFsigma4b2 = np.loadtxt("euga2al24b2.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)


# #######################################
# # Temperature dependent measurements
# #######################################
# sinlambda1a_2_100K, obs1a_2_100K, cal1a_2_100K, sigma1a_2_100K, DIFsigma1a_2_100K = np.loadtxt("euga2al2100K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# sinlambda1a_2_150K, obs1a_2_150K, cal1a_2_150K, sigma1a_2_150K, DIFsigma1a_2_150K = np.loadtxt("euga2al2150K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# sinlambda1a_2_200K, obs1a_2_200K, cal1a_2_200K, sigma1a_2_200K, DIFsigma1a_2_200K = np.loadtxt("euga2al2200K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# sinlambda1a_2_303K, obs1a_2_303K, cal1a_2_303K, sigma1a_2_303K, DIFsigma1a_2_303K = np.loadtxt("euga2al2303K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)

# a,c,V,z
T = np.array([100, 150, 200, 293]) # temperature in K
a = np.array([4.326090, 4.329725, 4.340179, 4.353075]) # a in angstrom cell parameter
da = np.array([0.000354, 0.000300, 0.000358, 0.000397])
c = np.array([10.916447, 10.922541, 10.935386, 10.970318]) # c in angstrom cell parameter
dc = np.array([0.001149, 0.001033, 0.001208, 0.001332])
z = np.array([0.38626856, 0.38572708, 0.38559416, 0.38490966]) # np.array([0.38716856, 0.38572708, 0.38559416, 0.38626856]) # np.array([0.3860138, 0.38572708, 0.38559416, 0.38490966]) # Wyckoff z-position Gallium # 0.38716856
dz = np.array([0.12992020e-02, 0.87138213e-03, 0.79795939e-03, 0.71762921e-03])
V = a**2*c # Volume in angstrom^3 cell parameter
# print("V_euga2al2 = {}".format(V))
dV = V*np.sqrt((2*da/a)**2 + (dc/c)**2)


# ########################################
# """EuGa4: 1-2 refinements + 1 temp. measurement??"""
# ########################################
# #Sample 1a
# sinlambda1a, obs1a, cal1a, sigma1a, DIFsigma1a = np.loadtxt("sample1a4.txt",
#                                                             usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
#Sample 1f
sinlambda1f, obs1f, cal1f, sigma1f, DIFsigma1f = np.loadtxt("euga41f.txt",
                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# #temp measurements
sinlambda200K, obs200K, cal200K, sigma200K, DIFsigma200K = np.loadtxt("euga4_200K.txt",
                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda293K, obs293K, cal293K, sigma293K, DIFsigma293K = np.loadtxt("euga4_293K.txt",
                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda250K, obs250K, cal250K, sigma250K, DIFsigma250K = np.loadtxt("euga4_250K.txt",

                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda150K, obs150K, cal150K, sigma150K, DIFsigma150K = np.loadtxt("euga4_150K.txt",
                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda100K, obs100K, cal100K, sigma100K, DIFsigma100K = np.loadtxt("euga4_100K.txt",
                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)


def main():
    print(__doc__)
    n = 4 # scalefactor for blue differnce plots

    """EuGa2Al2"""
    # Plot Refinements
    # 1a
    # plt.errorbar(sinlambda1a4, obs1a4, yerr=sigma1a4, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda1a4, cal1a4, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda1a4, abs(cal1a4 - obs1a4) - np.max(obs1a4) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda1a4) +0.05, 1000), - np.max(obs1a4) / n * np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0.1, np.max(sinlambda1a4) + 0.025)
    # plt.ylabel(r'Intensity (arb. units)', fontsize=13)
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)', fontsize=13)
    # plt.yticks([0, 5000, 10000, 15000, 20000])
    # plt.legend(fontsize=12)
    # plt.savefig("euga2al2_refinement_crystal4", dpi=300)
    # plt.show()

    # # # # 1c
    # plt.errorbar(sinlambda1c, obs1c, yerr=sigma1c, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda1c, cal1c, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda1c, abs(cal1c - obs1c) - np.max(obs1c) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda1c) +0.05, 1000), - np.max(obs1c) / n* np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0.1, np.max(sinlambda1c)+0.025)
    # plt.ylabel(r'Intensity (arb. units)', fontsize=13)
    # plt.tick_params(axis='y', labelsize=12, direction='in')
    # plt.tick_params(axis='x', labelsize=12, direction='in')
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)', fontsize=13)
    # plt.yticks([0, 200000, 400000, 600000, 800000])
    # plt.legend(fontsize=12)
    # plt.savefig("euga2al2_refinement_crystal1", dpi=300)
    # plt.show()
    #
    # # 3b2
    # plt.errorbar(sinlambda3b2, obs3b2, yerr=sigma3b2, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda3b2, cal3b2, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda3b2, abs(cal3b2 - obs3b2) - np.max(obs3b2) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda3b2) + 0.05, 1000), - np.max(obs3b2) / n* np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0.1, np.max(sinlambda3b2) + 0.025)
    # plt.ylabel(r'Intensity (arb. units)')
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)')
    # plt.legend()
    # plt.savefig("euga2al2_refinement_crystal3", dpi=300)
    # plt.show()
    #
    # # 4b2
    # plt.errorbar(sinlambda4b2, obs4b2, yerr=sigma4b2, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda4b2, cal4b2, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda4b2, abs(cal4b2 - obs4b2) - np.max(obs4b2) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda4b2) + 0.05, 1000), - np.max(obs4b2) / n* np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0, np.max(sinlambda4b2) + 0.1)
    # plt.ylabel(r'Intensity (arb. units)')
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$')
    # plt.legend()
    # plt.show()
    # # plt.savefig('EuGa2Al2_refinements.svg')
    # # plt.savefig('EuGa2Al2_T_refinements.svg')


    # D12, D22, Theta = [],[],[]
    # dD12, dD22, dTheta = [],[],[]
    # """STAVINOAH STUFF"""
    # for i in range(0,4):
    #     D12.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[0])
    #     dD12.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[1])
    #     D22.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[2])
    #     dD22.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[3])
    #     Theta.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[4])
    #     dTheta.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[5])
    # stavinoah(z[0], dz[0], a=a[0], da=da[0], c=c[0], dc=dc[0], sample="100K")
    # stavinoah(z[1], dz[1], a=a[1], da=da[1], c=c[1], dc=dc[1], sample="150K")
    # stavinoah(z[2], dz[2], a=a[2], da=da[2], c=c[2], dc=dc[2], sample="200K")
    # stavinoah(z[3], dz[3], a=a[3], da=da[3], c=c[3], dc=dc[3], sample="293K")
    # print("euga2al2 d12={}+-{}, d22={}+-{}, theta={}+-{}".format(D12,dD12, D22, dD22,Theta, dTheta))
    #
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    #
    # ax1.set_xlabel(r'T(K)', fontsize=30)
    # ax1.set_ylabel(r'$a$(Å)', color='b', fontsize=30)
    # ax1.errorbar(T, a, yerr=da, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$a$')
    # m, sigma_m , dm, t, sigma_t, dt = lin_reg(T, a, dy=np.zeros(4), sigma_y=da, plot=False)
    # ax1.plot(np.linspace(80, 310, 1000), m*np.linspace(80, 310, 1000) + t, lw=1.5, c='b', ls='-')
    # print("a: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.tick_params(axis='y', labelcolor='b', labelsize=30)
    # # ax1.set_xticks([100, 180, 200, 293])
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.legend(fontsize=27, loc='upper left')
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Temperature', color='g')
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, c, dy=np.zeros(4), sigma_y=dc, plot=False)
    # ax2.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='g', ls='-')
    # print("c: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax2.errorbar(T, c, yerr=dc, color='g', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$c$')
    # ax2.set_ylabel(r'$c$(Å)', color='g', fontsize=30)
    # ax2.tick_params(axis='y', labelcolor='g', labelsize=30, **tkw)
    # ax2.text(x=210, y=10.965, s=r'(a)', style='oblique', fontsize=27)
    # ax2.text(x=80, y=10.950, s=r'EuGa$_2$Al$_2$', style='italic', fontsize=30, fontweight='bold')
    # ax2.legend(fontsize=27, loc='lower right')
    # plt.tight_layout()
    # plt.savefig("stavinoah1_euga2al2_1.jpg", dpi=300)
    #
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    # ax1.set_xlabel(r'T(K)', fontsize=30)
    # ax1.set_ylabel(r'$d_{12}$(Å)', color='m', fontsize=30)
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, D12, dy=np.zeros(4), sigma_y=dD12, plot=False)
    # ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='m', ls='-')
    # print("d12: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.errorbar(T, D12, yerr=dD12, color='m', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$d_{12}$')
    # ax1.tick_params(axis='y', labelcolor='m', labelsize=30)
    # # ax1.set_xticks([100, 150, 200, 293])
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.legend(fontsize=27, loc='upper left')
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, D22, dy=np.zeros(4), sigma_y=dD22, plot=False)
    # ax2.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='y', ls='-')
    # print("d22: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax2.errorbar(T, D22, yerr=dD22, color='y', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$d_{22}$')
    # ax2.set_ylabel(r'$d_{22}$(Å)', color='y', fontsize=30)
    # ax2.tick_params(axis='y', labelcolor='y', labelsize=30, **tkw)
    # ax2.locator_params(axis='y', nbins=3)
    # ax2.text(x=220, y=2.536, s=r'(b)', style='oblique', fontsize=27)
    # ax2.legend(fontsize=27, loc='lower right')
    # plt.tight_layout()
    # plt.savefig("stavinoah1_euga2al2_2.jpg", dpi=300)
    #
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    # ax1.set_xlabel(r'T(K)', fontsize=30)
    # ax1.set_ylabel(r'$z$', color='r', fontsize=30)
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, z, dy=np.zeros(4), sigma_y=dz, plot=False)
    # ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='r', ls='-')
    # print("z: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.errorbar(T, z, yerr=dz, color='r', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$z$')
    # ax1.tick_params(axis='y', labelcolor='r', labelsize=30)
    # # ax1.set_xticks([100, 180, 200, 293])
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.legend(fontsize=27, loc='upper center')
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, Theta, dy=np.zeros(4), sigma_y=dTheta, plot=False)
    # ax2.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='c', ls='-')
    # print("a: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax2.errorbar(T, Theta, yerr=dTheta, color='c', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$\theta$')
    # ax2.set_ylabel(r'$\theta$(°)', color='c', fontsize=30)
    # ax2.tick_params(axis='y', labelcolor='c', labelsize=30, **tkw)
    # ax2.text(x=255, y=111.625, s=r'(c)', style='oblique', fontsize=27)
    # ax2.locator_params(axis='y', nbins=3)
    # ax2.legend(fontsize=27, loc='lower center')
    # plt.tight_layout()
    # plt.savefig("stavinoah1_euga2al2_3.jpg", dpi=300)
    # plt.show()


    """
    ============ 
    EuGa4
    ============ 
    """
    # # Plot Refinements
    # # # 1f
    # plt.errorbar(sinlambda1f, obs1f, yerr=sigma1f, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda1f, cal1f, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda1f, abs(cal1f - obs1f) - np.max(obs1f) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda1f) +0.05, 1000), - np.max(obs1f) / n * np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0.05, np.max(sinlambda1f) + 0.025)
    # plt.ylabel(r'Intensity (arb. units)', fontsize=13)
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)', fontsize=13)
    # plt.legend(fontsize=12)
    # plt.tick_params(axis='y', labelsize=12, direction='in')
    # plt.tick_params(axis='x', labelsize=12, direction='in')
    # plt.yticks([0, 40000, 80000, 120000])
    # plt.savefig("euga4_refinement_crystal1", dpi=300)
    # plt.show()

    # """Temperature measurements"""
    # # a,c,V,z
    # T = np.array([100, 200, 250, 293])  # temperature in K    150
    # a = np.array([4.375400, 4.387414, 4.387225, 4.403015])  # a in angstrom cell parameter 4.360766
    # da = np.array([0.000735, 0.000352, 0.000246, 0.000308]) # 0.000603
    # c = np.array([10.640713, 10.667181, 10.656446, 10.680388])  # c in angstrom cell parameter    10.605626
    # dc = np.array([0.002541, 0.001104, 0.000836, 0.000977])              #0.002070
    # B_eu = np.array([0.48221183, 0.89008331, 0.90516782, 1.1800470 ])      #0.95252866
    # dB_eu = np.array([0.48541449E-01 , 0.97667180E-01, 0.45610957E-01, 0.75068019E-01])    #0.18984486
    # B_ga1 = np.array([ 0.58138323 , 0.84271991, 0.83874643, 0.90734732])  #1.3838449
    # dB_ga1 = np.array([0.61789177E-01, 0.10146012, 0.46784237E-01, 0.71538962E-01])    #0.26306602
    # B_ga2 = np.array([0.40126890, 1.0198165, 0.83857411, 0.98966235])   #0.83032626
    # dB_ga2 = np.array([0.70558026E-01, 0.10149224, 0.58203440E-01, 0.85263073E-01]) #0.18114553
    # z = np.array([0.38288972, 0.38317543, 0.38373634,  0.38377333])  # Wyckoff z-position Gallium # 0.38716856   0.38448524
    # dz = np.array([0.28512880E-03, 0.31877682E-03, 0.17599505E-03, 0.26537123E-03])           #0.59160584E-03
    # V = a ** 2 * c  # Volume in angstrom^3 cell parameter
    # # print("V_euga2al2 = {}".format(V))
    # dV = V * np.sqrt((2 * da / a) ** 2 + (dc / c) ** 2)
    # #
    # D12, D22, Theta = [],[],[]
    # dD12, dD22, dTheta = [],[],[]
    # """STAVINOAH STUFF"""
    # for i in range(0,4):
    #     D12.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[0])
    #     dD12.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[1])
    #     D22.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[2])
    #     dD22.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[3])
    #     Theta.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[4])
    #     dTheta.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[5])
    # stavinoah(z[0], dz[0], a=a[0], da=da[0], c=c[0], dc=dc[0], sample="100K")
    # stavinoah(z[1], dz[1], a=a[1], da=da[1], c=c[1], dc=dc[1], sample="200K")
    # stavinoah(z[2], dz[2], a=a[2], da=da[2], c=c[2], dc=dc[2], sample="250K")
    # stavinoah(z[3], dz[3], a=a[3], da=da[3], c=c[3], dc=dc[3], sample="250K")
    # print("EuGa4 Stavinoha parameters from 100K to 293K: d12={}+-{}, d22={}+-{}, theta={}+-{}".format(D12, dD12, D22, dD22, Theta, dTheta))

    # ##############################
    # # Temperature measurement Plot
    # ##############################

    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.8)
    # tkw = dict(size=4, width=1.5)
    #
    # ax1.set_xlabel(r'T(K)', labelsize=30)
    # ax1.set_ylabel(r'$a$(Å)', color='b', labelsize=30)
    # ax1.errorbar(T, a, yerr=da, color='b', linestyle='--', lw=0.8, marker='o', markersize=10, capsize=4, label=r'$a$(Å)')
    # ax1.tick_params(axis='y', labelcolor='b', labelsize=30)
    # ax1.set_xticks([100, 200, 250, 293])
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.legend(labelsize=30, loc='upper left')
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # ax2.errorbar(T, c, yerr=dc, color='g', linestyle='--', lw=0.8, marker='s', markersize=10, capsize=4, label=r'$c$(Å)')
    # ax2.set_ylabel(r'$c$(Å)', color='g', labelsize=30)
    # ax2.tick_params(axis='y', labelcolor='g', labelsize=30, **tkw)
    # ax2.text(x=200, y=10.68, s=r'(a)', style='oblique', labelsize=30)
    # ax2.text(x=100, y=10.67, s=r'EuGa$_4$', style='italic', fontsize=18, fontweight='bold')
    # ax2.legend(labelsize=30, loc='lower right')
    # plt.savefig("stavinoah1_euga4_1.jpg", dpi=300)
    #
    #
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.8)
    # tkw = dict(size=4, width=1.5)
    #
    # ax1.set_xlabel(r'T(K)', labelsize=30)
    # ax1.set_ylabel(r'$d_{12}$(Å)', color='m', labelsize=30)
    # ax1.errorbar(T, D12, yerr=dD12, color='m', linestyle='--', lw=0.8, marker='o', markersize=10, capsize=4, label=r'$d_{12}$(Å)')
    # ax1.tick_params(axis='y', labelcolor='m', labelsize=30)
    # ax1.set_xticks([100, 200, 250, 293])
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.legend(labelsize=30, loc='upper left')
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # ax2.errorbar(T, D22, yerr=dD22, color='y', linestyle='--', lw=0.8, marker='s', markersize=10, capsize=4, label=r'$d_{22}$(Å)')
    # ax2.set_ylabel(r'$d_{22}$(Å)', color='y', labelsize=30)
    # ax2.tick_params(axis='y', labelcolor='y', labelsize=30, **tkw)
    # ax2.text(x=250, y=2.4975, s=r'(b)', style='oblique', labelsize=30)
    # ax2.legend(labelsize=30, loc='lower right')
    # plt.savefig("stavinoah1_euga4_2.jpg", dpi=300)
    #
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.8)
    # tkw = dict(size=4, width=1.5)
    # ax1.set_xlabel(r'T(K)', labelsize=30)
    # ax1.set_ylabel(r'$z$', color='r', labelsize=30)
    # ax1.errorbar(T, z, yerr=dz, color='r', linestyle='--', lw=0.8, marker='o', markersize=10, capsize=4, label=r'$z$')
    # ax1.tick_params(axis='y', labelcolor='r', labelsize=30)
    # ax1.set_xticks([100, 200, 250, 293])
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.legend(labelsize=30, loc='upper center')
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # ax2.errorbar(T, Theta, yerr=dTheta, color='c', linestyle='--', lw=0.8, marker='s', markersize=10, capsize=4, label=r'$\theta$(°)')
    # ax2.set_ylabel(r'$\theta$(°)', color='c', labelsize=30)
    # ax2.tick_params(axis='y', labelcolor='c', labelsize=30, **tkw)
    # ax2.text(x=130, y=114.27, s=r'(c)', style='oblique', labelsize=30)
    # ax2.legend(labelsize=30, loc='lower center')
    # plt.savefig("stavinoah1_euga4_3.jpg", dpi=300)
    # plt.show()

    ax1.set_xlabel(r'T(K)', fontsize=30)
    ax1.set_ylabel(r'$a$(Å)', color='b', fontsize=30)
    ax1.errorbar(T, a, yerr=da, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$a$')
    m, sigma_m , dm, t, sigma_t, dt = lin_reg(T, a, dy=np.zeros(4), sigma_y=da, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m*np.linspace(80, 310, 1000) + t, lw=1.5, c='b', ls='-')
    print("a: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.tick_params(axis='y', labelcolor='b', labelsize=30)
    # ax1.set_xticks([100, 180, 200, 293])
    ax1.locator_params(axis='y', nbins=3)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.legend(fontsize=27, loc='upper left')
    # Adding Twin Axes
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature', color='g')
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, c, dy=np.zeros(4), sigma_y=dc, plot=False)
    ax2.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='g', ls='-')
    print("c: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax2.errorbar(T, c, yerr=dc, color='g', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$c$')
    ax2.set_ylabel(r'$c$(Å)', color='g', fontsize=30)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=30, **tkw)
    ax2.text(x=210, y=10.965, s=r'(a)', style='oblique', fontsize=27)
    ax2.text(x=80, y=10.950, s=r'EuGa$_2$Al$_2$', style='italic', fontsize=30, fontweight='bold')
    ax2.legend(fontsize=27, loc='lower right')
    plt.tight_layout()
    plt.savefig("stavinoah1_euga2al2_1.jpg", dpi=300)

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)
    tkw = dict(size=4, width=1.5)
    ax1.set_xlabel(r'T(K)', fontsize=30)
    ax1.set_ylabel(r'$d_{12}$(Å)', color='m', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, D12, dy=np.zeros(4), sigma_y=dD12, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='m', ls='-')
    print("d12: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T, D12, yerr=dD12, color='m', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$d_{12}$')
    ax1.tick_params(axis='y', labelcolor='m', labelsize=30)
    # ax1.set_xticks([100, 150, 200, 293])
    ax1.locator_params(axis='y', nbins=3)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.legend(fontsize=27, loc='upper left')
    # Adding Twin Axes
    ax2 = ax1.twinx()
    # ax2.set_ylabel('Temperat', color='g')
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, D22, dy=np.zeros(4), sigma_y=dD22, plot=False)
    ax2.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='y', ls='-')
    print("d22: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax2.errorbar(T, D22, yerr=dD22, color='y', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$d_{22}$')
    ax2.set_ylabel(r'$d_{22}$(Å)', color='y', fontsize=30)
    ax2.tick_params(axis='y', labelcolor='y', labelsize=30, **tkw)
    ax2.locator_params(axis='y', nbins=3)
    ax2.text(x=220, y=2.536, s=r'(b)', style='oblique', fontsize=27)
    ax2.legend(fontsize=27, loc='lower right')
    plt.tight_layout()
    plt.savefig("stavinoah1_euga2al2_2.jpg", dpi=300)

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)
    tkw = dict(size=4, width=1.5)
    ax1.set_xlabel(r'T(K)', fontsize=30)
    ax1.set_ylabel(r'$z$', color='r', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, z, dy=np.zeros(4), sigma_y=dz, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='r', ls='-')
    print("z: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T, z, yerr=dz, color='r', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$z$')
    ax1.tick_params(axis='y', labelcolor='r', labelsize=30)
    # ax1.set_xticks([100, 180, 200, 293])
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.locator_params(axis='y', nbins=3)
    ax1.legend(fontsize=27, loc='upper center')
    # Adding Twin Axes
    ax2 = ax1.twinx()
    # ax2.set_ylabel('Temperat', color='g')
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, Theta, dy=np.zeros(4), sigma_y=dTheta, plot=False)
    ax2.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='c', ls='-')
    print("a: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax2.errorbar(T, Theta, yerr=dTheta, color='c', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$\theta$')
    ax2.set_ylabel(r'$\theta$(°)', color='c', fontsize=30)
    ax2.tick_params(axis='y', labelcolor='c', labelsize=30, **tkw)
    ax2.text(x=255, y=111.625, s=r'(c)', style='oblique', fontsize=27)
    ax2.locator_params(axis='y', nbins=3)
    ax2.legend(fontsize=27, loc='lower center')
    plt.tight_layout()
    plt.savefig("stavinoah1_euga2al2_3.jpg", dpi=300)
    plt.show()

    # """CDW"""
    # # # mixed data points from Nakamura et. al: Q || [100] [110] [001]
    # # ADD OUR EXPERIMENTAL POINTS IF POSSIBLE
    # p_st = np.array([0.7404580152671756, 0.7518904648977581, 1.1155308962369683, 1.4053872656494697, 1.4961832061068703,
    # 1.6412213740458015, 1.8558324757259834, 2.1583516040986175, 2.0687022900763354, 2.2900763358778624, 2.5877862595419847])
    # T_st = np.array([100.00000000000003, 104.50579444270437, 122.74822139693052, 141.60812564320554, 141.9847328244275, 122.13740458015272,
    # 153.24175578325654, 160.776768535505, 138.9312977099237, 161.83206106870233, 173.28244274809165])
    # lin_reg(p_st, T_st, dy=np.ones(11), sigma_y=np.ones(11), yaxis=r'T$_{CDW}$(K)', xaxis=r'p(GPa)', plot=True, Titel="EuGa4_CDW_T")
    # plt.xlim(0, 6)
    # plt.ylim(0, 300)
    # plt.savefig("euga4_cdw_linfit.png", dpi=300)
    # plt.show()
    # # # plt.scatter(p_st, T_st)

    #######################################
    """DAC Pressure and Temperature data"""
    #######################################
    # a_2_293, a_3_293, a_3_250, a_5_293, a_5_250 = 0,0,0,0,0
    # c_2_293, c_3_293, c_3_250, c_5_293, c_5_250 = 0,0,0,0,0

    # """T,p,a,c,z"""
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # twin1 = ax.twinx()
    # twin2 = ax.twinx()
    # # Offset the right spine of twin2.  The ticks and label have already been
    # # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    # # print("Theta={}".format(Theta))
    # p1, = ax.plot(T, D12, "b-", marker='o', linestyle='--', lw='0.8'
    #                    )
    # p2, = twin1.plot(T, D22, "g-", marker='o', linestyle='--', lw='0.8'
    #                      )
    # p3, = twin2.plot(T, Theta, "r-", marker='o', linestyle='--', lw='0.8'
    #                      )
    # ax.yaxis.label.set_color(p1.get_color())
    # twin1.yaxis.label.set_color(p2.get_color())
    # twin2.yaxis.label.set_color(p3.get_color())
    #
    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis="y", colors=p1.get_color(), labelsize=12, direction='in', marker='v', **tkw)
    # twin1.tick_params(axis="y", colors=p2.get_color(), labelsize=12, direction='in', marker='o',**tkw)
    # twin2.tick_params(axis="y", colors=p3.get_color(), labelsize=12, direction='in', marker='q',**tkw)
    # ax.tick_params(axis='x',labelsize=12, direction='in', **tkw)
    # ax.set_ylabel(r'', fontsize=13)
    # twin1.set_ylabel(r'', fontsize=13)
    # twin2.set_ylabel(r'', fontsize=13)
    # ax.set_xticks([100, 150, 200, 250, 293])
    # ax.set_xlabel(r'T(K)', fontsize=12)
    #
    # # # ax.legend(handles=[p1, p2, p3])
    # # plt.savefig("DAC_data", dpi=300)
    # plt.show()



if __name__ == '__main__':
    main()

    # """OLD"""
    # # Temperature measurement Plot
    # # a, c, z
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # twin1 = ax.twinx()
    # twin2 = ax.twinx()
    # # Offset the right spine of twin2.  The ticks and label have already been
    # # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    # p1, = ax.plot(T, z, "b-", marker='o', linestyle='--', lw='0.8'
    #                    )
    # p2, = twin1.plot(T, a, "g-", marker='o', linestyle='--', lw='0.8'
    #                      )
    # p3, = twin2.plot(T, c, "r-", marker='o', linestyle='--', lw='0.8'
    #                      )
    # ax.yaxis.label.set_color(p1.get_color())
    # twin1.yaxis.label.set_color(p2.get_color())
    # twin2.yaxis.label.set_color(p3.get_color())
    #
    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis="y", colors=p1.get_color(), labelsize=12, direction='in', **tkw)
    # twin1.tick_params(axis="y", colors=p2.get_color(), labelsize=12, direction='in', **tkw)
    # twin2.tick_params(axis="y", colors=p3.get_color(), labelsize=12, direction='in', **tkw)
    # ax.tick_params(axis='x', labelsize=12, direction='in', **tkw)
    # ax.set_ylabel(r'$z$', fontsize=13) #V(Å$^3$)
    # twin1.set_ylabel(r'$a(Å)$', fontsize=13)
    # twin2.set_ylabel(r'$c(Å)$', fontsize=13)
    # ax.set_xticks([100,150,200,250,293])
    # ax.set_yticks([0.3850, 0.3853, 0.3856, 0.3859, 0.3862])
    # # linear fits for a,c,z
    # # lin_reg(T, z, dy=np.zeros(4), sigma_y=dz, plot=True)
    # lin_reg(T, a, dy=np.zeros(4), sigma_y=da)
    # lin_reg(T, c, dy=np.zeros(4), sigma_y=dc)
    # # print("EuGa2Al2 cell parameters at T=0K with linear fit with statistical uncertainties from \
    # # Crysalis and Fullprof: a={}, c={}, z={}".format(lin_reg(T, z, dy=np.zeros(4), sigma_y=dz)[3], lin_reg(T, a, dy=np.zeros(4), sigma_y=da)[3],
    #        #                                          lin_reg(T, c, dy=np.zeros(4), sigma_y=dc)[3]))
    # ax.set_xlabel(r'T(K)', fontsize=12)
    # plt.savefig("stavinoah1_euga2al2", dpi=300)

    #"""d12, d22, theta"""
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # twin1 = ax.twinx()
    # twin2 = ax.twinx()
    # # Offset the right spine of twin2.  The ticks and label have already been
    # # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    # # print("Theta={}".format(Theta))
    # p1, = ax.plot(T, D12, "b-", marker='o', linestyle='--', lw='0.8'
    #                    )
    # p2, = twin1.plot(T, D22, "g-", marker='o', linestyle='--', lw='0.8'
    #                      )
    # p3, = twin2.plot(T, Theta, "r-", marker='o', linestyle='--', lw='0.8'
    #                      )
    # ax.yaxis.label.set_color(p1.get_color())
    # twin1.yaxis.label.set_color(p2.get_color())
    # twin2.yaxis.label.set_color(p3.get_color())
    #
    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis="y", colors=p1.get_color(), labelsize=12, direction='in', **tkw)
    # twin1.tick_params(axis="y", colors=p2.get_color(), labelsize=12, direction='in', **tkw)
    # twin2.tick_params(axis="y", colors=p3.get_color(), labelsize=12, direction='in', **tkw)
    # ax.tick_params(axis='x',labelsize=12, direction='in', **tkw)
    # ax.set_ylabel(r'd$_{12}$(Å)', fontsize=13)
    # twin1.set_ylabel(r'd$_{22}$(Å)', fontsize=13)
    # twin2.set_ylabel(r'$\theta$(°)', fontsize=13)
    # ax.set_xticks([100, 150, 200, 250, 293])
    # ax.set_xlabel(r'T(K)', fontsize=12)
    #
    # # # # ax.legend(handles=[p1, p2, p3])
    # plt.savefig("stavinoah2_euga2al2", dpi=300)
    # plt.show()

    # """OLD"""
    # fig, ax1 = plt.subplots()
    # # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    #
    # ax1.set_xlabel(r'Temperature', fontsize=15)
    # ax1.set_ylabel(r'$a$(Å)', color='b', fontsize=15)
    # ax1.errorbar(T, a, yerr=da, color='b', linestyle='--', lw=0.8, marker='o', markersize=8, capsize=4)
    # ax1.tick_params(axis='y', labelcolor='b', labelsize=25)
    # ax1.set_xticks([100, 200, 250, 293])
    # ax1.tick_params(axis='x', labelsize=25, direction='in', **tkw)
    #
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # ax2.errorbar(T, c, yerr=dc, color='g', linestyle='--', lw=0.8, marker='s', markersize=8, capsize=4)
    # ax2.set_ylabel(r'$c$(Å)', color='g', fontsize=15)
    # ax2.tick_params(axis='y', labelcolor='g', labelsize=25, **tkw)
    # plt.savefig("stavinoah1_euga4_1", dpi=300)
    #
    #
    # fig, ax1 = plt.subplots()
    # # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    #
    # ax1.set_xlabel(r'Temperature', fontsize=15)
    # ax1.set_ylabel(r'$d_{12}$(Å)', color='m', fontsize=15)
    # ax1.errorbar(T, D12, yerr=dD12, color='m', linestyle='--', lw=0.8, marker='o', markersize=8, capsize=4)
    # ax1.tick_params(axis='y', labelcolor='m', labelsize=25)
    # ax1.set_xticks([100, 200, 250, 293])
    # ax1.tick_params(axis='x', labelsize=25, direction='in', **tkw)
    #
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # ax2.errorbar(T, D22, yerr=dD22, color='y', linestyle='--', lw=0.8, marker='s', markersize=8, capsize=4)
    # ax2.set_ylabel(r'$d_{22}$(Å)', color='y', fontsize=15)
    # ax2.tick_params(axis='y', labelcolor='y', labelsize=25, **tkw)
    # plt.savefig("stavinoah1_euga4_2", dpi=300)
    #
    # fig, ax1 = plt.subplots()
    # # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    #
    # ax1.set_xlabel(r'Temperature', fontsize=15)
    # ax1.set_ylabel(r'$z$', color='r', fontsize=15)
    # ax1.errorbar(T, z, yerr=dz, color='r', linestyle='--', lw=0.8, marker='o', markersize=8, capsize=4)
    # ax1.tick_params(axis='y', labelcolor='r', labelsize=25)
    # ax1.set_xticks([100, 200, 250, 293])
    # ax1.tick_params(axis='x', labelsize=25, direction='in', **tkw)
    #
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # # ax2.set_ylabel('Temperat', color='g')
    # ax2.errorbar(T, Theta, yerr=dTheta, color='c', linestyle='--', lw=0.8, marker='s', markersize=8, capsize=4)
    # ax2.set_ylabel(r'$\theta$(°)', color='c', fontsize=15)
    # ax2.tick_params(axis='y', labelcolor='c', labelsize=15, **tkw)
    # plt.show()
    # plt.savefig("stavinoah1_euga4_3", dpi=300)



    # a, c, z
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # twin1 = ax.twinx()
    # twin2 = ax.twinx()
    # # Offset the right spine of twin2.  The ticks and label have already been
    # # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    # p1 = ax.errorbar(x=T, y=z, yerr=dz, fmt='b-', marker='o', linestyle='--', lw='0.8', markersize='10'
    #                    )
    # p2, = twin1.plot(T, a, "g-", marker='v', linestyle='--', lw='0.8', markersize='10'
    #                      )
    # p3, = twin2.plot(T, c, "r-", marker='x', linestyle='--', lw='0.8', markersize='10'
    #                      )
    # # ax.yaxis.label.set_color(p1.get_children()[0].get_color)
    # twin1.yaxis.label.set_color(p2.get_color())
    # twin2.yaxis.label.set_color(p3.get_color())
    #
    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis="y", colors='g', labelsize=12, direction='in', **tkw)
    # twin1.tick_params(axis="y", colors=p2.get_color(), labelsize=12, direction='in', **tkw)
    # twin2.tick_params(axis="y", colors=p3.get_color(), labelsize=12, direction='in', **tkw)
    # ax.tick_params(axis='x', labelsize=12, direction='in', **tkw)
    # ax.set_ylabel(r'$z$', fontsize=13) #V(Å$^3$)
    # twin1.set_ylabel(r'a(Å)', fontsize=13)
    # twin2.set_ylabel(r'c(Å)', fontsize=13)
    # ax.set_xticks([100,200,250,293])
    # # ax.set_yticks([0.3850, 0.3853, 0.3856, 0.3859, 0.3862])
    # # linear fits for a,c,z
    # # lin_reg(T, z, dy=np.zeros(4), sigma_y=dz)
    # lin_reg(T, a, dy=np.zeros(4), sigma_y=da)
    # lin_reg(T, c, dy=np.zeros(4), sigma_y=dc)
    # # print("EuGa2Al2 cell parameters at T=0K with linear fit with statistical uncertainties from \
    # # Crysalis and Fullprof: a={}, c={}, z={}".format(lin_reg(T, z, dy=np.zeros(4), sigma_y=dz)[3], lin_reg(T, a, dy=np.zeros(4), sigma_y=da)[3],
    #        #                                          lin_reg(T, c, dy=np.zeros(4), sigma_y=dc)[3]))
    # ax.set_xlabel(r'T(K)', fontsize=12)
    # plt.show()
    # plt.savefig("stavinoah1_euga4", dpi=300)

    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # twin1 = ax.twinx()
    # twin2 = ax.twinx()
    # # Offset the right spine of twin2.  The ticks and label have already been
    # # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    # # print("Theta={}".format(Theta))
    # p1, = ax.plot(T, D12, "b-", marker='o', linestyle='--', lw='0.8', markersize='10'
    #               )
    # p2, = twin1.plot(T, D22, "g-", marker='v', linestyle='--', lw='0.8', markersize='10'
    #                  )
    # p3, = twin2.plot(T, Theta, "r-", marker='x', linestyle='--', lw='0.8', markersize='10'
    #                  )
    # ax.yaxis.label.set_color(p1.get_color())
    # twin1.yaxis.label.set_color(p2.get_color())
    # twin2.yaxis.label.set_color(p3.get_color())
    #
    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis="y", colors=p1.get_color(), labelsize=12, direction='in', **tkw)
    # twin1.tick_params(axis="y", colors=p2.get_color(), labelsize=12, direction='in', **tkw)
    # twin2.tick_params(axis="y", colors=p3.get_color(), labelsize=12, direction='in', **tkw)
    # ax.tick_params(axis='x', labelsize=12, direction='in', **tkw)
    # ax.set_ylabel(r'd$_{12}$(Å)', fontsize=13)
    # twin1.set_ylabel(r'd$_{22}$(Å)', fontsize=13)
    # twin2.set_ylabel(r'$\theta$(°)', fontsize=13)
    # ax.set_xticks([100, 200, 250, 293])
    # ax.set_xlabel(r'T(K)', fontsize=12)

    # # # ax.legend(handles=[p1, p2, p3])
    # plt.savefig("stavinoah2_euga4", dpi=300)

    # # Temp plots # 200K measurement yields less reflections,
    # plt.errorbar(sinlambda293K, obs293K, yerr=sigma293K, ls='', marker='x', capsize=1.0, c='tab:brown', label='293K')
    # plt.errorbar(sinlambda250K, obs250K, yerr=sigma250K, ls='', marker='x', capsize=1.0, c='tab:red', label='250K')
    # plt.errorbar(sinlambda200K, obs200K, yerr=sigma200K, ls='', marker='x', capsize=1.0, c='tab:blue', label='200K')
    # plt.errorbar(sinlambda150K, obs150K, yerr=sigma150K, ls='', marker='x', capsize=1.0, c='tab:green', label='150K')
    # plt.errorbar(sinlambda100K, obs100K, yerr=sigma100K, ls='', marker='x', capsize=1.0, c='tab:orange', label='100K')
    # plt.xlim(0.1, np.max(sinlambda200K)+0.025)
    # plt.tick_params(axis='y', labelsize=12, direction='in')
    # plt.tick_params(axis='x', labelsize=12, direction='in')
    # plt.ylabel(r'Intensity (arb. units)', fontsize=13)
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)', fontsize=13)
    # plt.legend(fontsize=12)
    # plt.savefig("euga4_temp_intensities", dpi=300)
    # plt.show()