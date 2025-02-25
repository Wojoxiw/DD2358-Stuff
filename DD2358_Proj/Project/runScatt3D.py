# encoding: utf-8
### Modification of scatt2d to handle 3d geometry
# Stripped down for simplicity for DD2358 course
# Simple scattering case of 2 antennas at a 90 degree angle, with a sphere in the center
#
# Daniel Sjoberg, 2024-12-13
# Alexandros Pallaris, after that

# use pytest
# use pylint
# use sphinx
# use logging instead of printing, maybe?

import os
os.environ["OMP_NUM_THREADS"] = "1" # seemingly needed for MPI speedup
import subprocess
import Scatt3D
from memory_profiler import memory_usage
from mpi4py import MPI
import sys

##MAIN STUFF
if __name__ == '__main__':
    N = 3
    child=MPI.COMM_SELF.Spawn(sys.executable,args=['Scatt3D.py'],maxprocs=N) ## will run it like mpirun -n N Scatt3D.py, I think
    
    #mem_usage = memory_usage(subprocess.run(command, stdout=subprocess.PIPE), max_usage=True)
    #print(mem_usage)
    #print('Max. memory:',mem_usage/1000,'GiB')
    #memTimeAppend(size, Nf, mem_usage/1000, totT, reference=False) ## '0' memory cost to ignore this one (or later fill in manually) - not sure how to easily estimate this without slowing the code
      
    