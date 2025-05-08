# encoding: utf-8
## this file will have much of the postprocessing

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import basix
import functools
from timeit import default_timer as timer
from memory_profiler import memory_usage
import gmsh
import sys
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi
import memTimeEstimation
from matplotlib import pyplot as plt
eta0 = np.sqrt(mu0/eps0)
