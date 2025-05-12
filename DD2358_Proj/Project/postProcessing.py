# encoding: utf-8
## this file will have much of the postprocessing

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import basix
import functools
from timeit import default_timer as timer
import gmsh
import sys
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi
from matplotlib import pyplot as plt
import h5py

def testSVD(prob): ## takes a problem, does test svd on it, using DASK
    ## load in all the data
    Nf = prob.Nf
    N_antennas = prob.refMeshdata.N_antennas
    #data = np.load(prob.dataFolder+prob.name+'output-qs.xdmf')
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, prob.dataFolder+prob.name+'output-qs.xdmf', 'r') as f:
        mesh = f.read_mesh()
    idx = mesh.topology.original_cell_index
    Nb = Nf*N_antennas*N_antennas
    with h5py.File(prob.dataFolder+prob.name+'output-qs.h5', 'r') as f:
        cell_volumes = np.array(f['Function']['real_f']['-3']).squeeze()
        cell_volumes[:] = cell_volumes[idx]
        epsr_array_ref = np.array(f['Function']['real_f']['-2']).squeeze() + 1j*np.array(f['Function']['imag_f']['-2']).squeeze()
        epsr_array_ref = epsr_array_ref[idx]
        epsr_array_dut = np.array(f['Function']['real_f']['-1']).squeeze() + 1j*np.array(f['Function']['imag_f']['-1']).squeeze()
        epsr_array_dut = epsr_array_dut[idx]
        N = len(cell_volumes)
        A = np.zeros((Nb, N), dtype=complex) ## the matrix of scaled E-field stuff
        for n in range(Nb):
            A[n,:] = np.array(f['Function']['real_f'][str(n)]).squeeze() + 1j*np.array(f['Function']['imag_f'][str(n)]).squeeze()
            A[n,:] = A[n,idx]

    ## create b-array, of S-parameters
    b = np.zeros(Nf*N_antennas*N_antennas, dtype=complex)
    for nf in range(Nf):
        for m in range(N_antennas):
            for n in range(N_antennas):
                b[nf*N_antennas*N_antennas + m*N_antennas + n] = prob.solutions_dut[nf, m, n] - prob.solutions_ref[nf, n, m]
                
    if (True):
        idx = np.nonzero(np.abs(epsr_array_ref) > 1)[0] ## a priori
        A = A[idx,:]
        b = b[idx]
                
    print('do SVD now')