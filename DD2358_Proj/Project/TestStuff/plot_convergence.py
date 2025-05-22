import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c as c0
import miepython

epsr = 3.5 - 0j
f0 = 10e9
lambda0 = c0/f0
a = lambda0*.45
#hfactors = np.array([5, 10, 20, 30, 40], dtype=float)
hfactors = np.array([5, 10, 15, 20], dtype=float)

def ComputeErrors(sol, ref):
    abs_error = np.sqrt(np.sum(np.abs(sol - ref)**2)/len(sol))
    rel_error = np.sqrt(np.sum(np.abs(sol - ref)**2/np.abs(ref)**2)/len(sol))
    return abs_error, rel_error

abs_errors_Eplane = []
rel_errors_Eplane = []
abs_errors_Hplane = []
rel_errors_Hplane = []

for hfactor in hfactors:
    filename = f'sim_{hfactor}.dat'
    data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1)
    cut = np.real(data[:,0])
    ffsq_Eplane = np.abs(data[:,1])**2 + np.abs(data[:,2])**2
    ffsq_Hplane = np.abs(data[:,3])**2 + np.abs(data[:,4])**2
    
    # Mie solution
    x = 2*np.pi*f0*a/c0
    m = np.sqrt(epsr, dtype=complex)
    mie_E = miepython.i_par(m, x, np.cos(cut*np.pi/180), norm='qsca')*np.pi*a**2
    mie_H = miepython.i_per(m, x, np.cos(cut*np.pi/180), norm='qsca')*np.pi*a**2

    # Compute errors
    abs_error_Eplane, rel_error_Eplane = ComputeErrors(ffsq_Eplane, mie_E)
    abs_error_Hplane, rel_error_Hplane = ComputeErrors(ffsq_Hplane, mie_H)
    abs_errors_Eplane.append(abs_error_Eplane)
    rel_errors_Eplane.append(rel_error_Eplane)
    abs_errors_Hplane.append(abs_error_Hplane)
    rel_errors_Hplane.append(rel_error_Hplane)
        
    # Plotting comparison
    plt.figure()
    plt.semilogy(cut, ffsq_Eplane, label='E plane')
    plt.semilogy(cut, ffsq_Hplane, label='H plane')
    plt.semilogy(cut, mie_E, '--', label='Mie E')
    plt.semilogy(cut, mie_H, '--', label='Mie H')
    plt.xlabel('Angle (degrees)')
    plt.grid()
    plt.legend(loc='best')
    plt.title(f'lambda/h = {hfactor}')

plt.figure()
plt.loglog(hfactors, abs_errors_Eplane, label='abs')
plt.loglog(hfactors, rel_errors_Eplane, label='rel')
plt.xlabel('lambda/h')
plt.legend(loc='best')
plt.grid()
plt.title('E plane')

plt.figure()
plt.loglog(hfactors, abs_errors_Hplane, label='abs')
plt.loglog(hfactors, rel_errors_Hplane, label='rel')
plt.xlabel('lambda/h')
plt.legend(loc='best')
plt.grid()
plt.title('H plane')

plt.show()
