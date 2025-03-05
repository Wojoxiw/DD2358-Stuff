'''
Some utility scripts.

Created on 27 feb. 2025
@author: al8032pa
'''
import numpy as np
from matplotlib import pyplot as plt
import scipy
import os
    
class runTimesMems():
    '''
    Stores data about the previous runtimes, to be saved/loaded with numpy. This holds them like a list, and has some utility functions.
    Only the master (rank 0) process's times are saved - there shouldn't be much of a difference between processes, though.
    Only the total memory taken is saved - should be about equally spread between the processes.
    '''
    def __init__(self, folder, comm, filename = 'prevRuns.prev'):
        '''
        'Create' it every time - if it already exists, load the data. If not, we start empty
        :param folder: Folder where it is saved
        :param comm: MPI communicator
        '''
        self.comm = comm
        self.fpath = folder+filename
        if os.path.isfile(self.fpath):
            self.prevRuns = np.load(self.fpath, allow_pickle=True)['prevRuns']
            if(self.comm.rank == 0):
                print(f'Loaded previous run data from {len(self.prevRuns)} runs')
            
        #np.savez(self.fpath, prevRuns = self)
        
    class runTimeMem():
        '''
        The data of one previous run.
        '''
        def __init__(self, prob):
            '''
            Initialize the class.
            :param prob: The scattering problem run
            :param mem: Total memory used in the problem
            '''
            self.meshingTime = prob.refMeshdata.meshingTime
            self.calcTime = prob.calcTimes
            self.MPInum = prob.MPInum
            self.Nf = prob.Nf # number of frequencies - computations are for each frequency
            self.Nants = prob.refMeshdata.N_antennas # number of antennas - computations are for each antenna pair
            self.size = prob.refMeshdata.ncells # Computation size (number of FEM elements)
            self.mem = prob.memCost # Total memory cost
            
    def memTimeAppend(self, run):
        '''
        Takes a scatteringProblem class, updates the list and saves it
        :param run: The run
        '''
        if(self.comm.rank == 0): ## only use the master rank
            prev = self.runTimeMem(run) ## the runTimeMem
            if(hasattr(self, 'prevRuns')): ## check if these exist
                n = len(self.prevRuns)
                prevs = np.empty(n+1, dtype=object)
                for i in range(n):
                    prevs[i] = self.prevRuns[i]
                prevs[n] = prev    
                self.prevRuns = prevs
            else: ## hasn't been made yet
                self.prevRuns = [prev]
                
                
            np.savez(self.fpath, prevRuns = self.prevRuns)
    
    def fitLine(self, x, a, b): ## curve to fit the data to - assume some power dependance on various parameters
        return a*x + b
    
    def calcStats(self):
        '''
        Prepares arrays of runtime and memory costs by computation size
        '''
        numRuns = len(self.prevRuns)
        self.sizes = np.zeros(numRuns)
        self.mems = np.zeros(numRuns)
        self.times = np.zeros(numRuns)
        self.numProcesses = np.zeros(numRuns)
        self.Nfs = np.zeros(numRuns)
        self.Nants = np.zeros(numRuns)
        
        for i in range(numRuns):
            run = self.prevRuns[i]
            self.sizes[i] = run.size
            self.mems[i] = run.mem
            self.times[i] = run.meshingTime + run.calcTime
            self.numProcesses[i] = run.MPInum
            self.Nfs[i] = run.Nf
            self.Nants[i] = run.Nants
        
        ## Assume computations time scales by problem size, Nfs, and Nants**2, and maybe 1/MPInum
        self.timeFit = scipy.optimize.curve_fit(self.fitLine, self.sizes*self.Nfs*self.Nants**2, self.times)[0]
        ## Assume memory cost scales just by problem size
        self.memFit = scipy.optimize.curve_fit(self.fitLine, self.sizes, self.mems)[0]
          
    def memTimeEstimation(self):
        pass
    
    def makePlots(self):
        if(self.comm.rank == 0):
            self.calcStats()
            ## Times plot
            xs = np.linspace(np.min(self.sizes), np.max(self.sizes), 1000)
            plt.plot(self.sizes, self.times/3600, '-o', label='runs on computer')
            plt.title('Computation time by size')
            plt.xlabel(r'(# elements)*')
            plt.ylabel('Time [hours]')
            plt.grid()
            plt.plot(xs, self.fitLine(xs, self.timeFit[0], self.timeFit[1])/3600, label='curve_fit')
            #if(numCells>0 and Nf>0):
                #plt.scatter(numCells*Nf, time/3600, s = 80, facecolors = None, edgecolors = 'red', label = 'Estimated Time')
            plt.legend()
            plt.tight_layout()
            plt.show()
            ## Memory plot
            plt.plot(self.sizes, self.mems, '-o', label='runs on computer')
            plt.title('Memory Requirements by size')
            plt.xlabel(r'# elements')
            plt.ylabel('Memory [GB] (Approximate)')
            plt.grid()
            plt.plot(xs, self.fitLine(xs, self.memFit[0], self.memFit[1]), label='curve_fit')
            #if(numCells>0 and Nf>0):
                #plt.scatter(numCells, mem, s = 80, facecolors = None, edgecolors = 'red', label = 'Estimated Memory')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
    def makePlotsSTD(self, binVals):
        '''
        Makes plots with standard deviation error bars for specific values (for DD2358). These are specific sizes, and MPInums
        Does its own version of calcStats, and makes plots herein
        :param binVals: the size-values (# elements) that were used for calculations 
        '''
        if(self.comm.rank == 0):
            numRuns = len(self.prevRuns) 
            
            for binVal in binVals:
                ## first count the runs in this bin
                l = 0
                for i in range(numRuns):
                    run = self.prevRuns[i]
                    if(np.isclose(run.size, binVal)):
                        l+=1
                ## then make the arrays
                self.sizes = np.zeros(l)
                self.mems = np.zeros(l)
                self.times = np.zeros(l)
                self.numProcesses = np.zeros(l)
                self.Nfs = np.zeros(l)
                self.Nants = np.zeros(l)
                l = 0
                for i in range(numRuns):
                    run = self.prevRuns[i]
                    if(np.isclose(run.size, binVal)):
                        self.sizes[i] = run.size
                        self.mems[i] = run.mem
                        self.times[i] = run.meshingTime + run.calcTime
                        self.numProcesses[i] = run.MPInum
                        self.Nfs[i] = run.Nf
                        self.Nants[i] = run.Nants
                        l+=1
            
            ## do some curve fitting, assuming some semi-arbitrary dependencies
            xdata = np.vstack((self.sizes, self.numProcesses, self.Nfs, self.Nants))
            ## Assume computations time scales by problem size, Nfs, and Nants**2, and maybe 1/MPInum
            self.timeFit = scipy.optimize.curve_fit(self.fitLine, xdata, self.times)[0]
            ## Assume memory cost scales just by problem size
            self.memFit = scipy.optimize.curve_fit(self.fitLine, xdata, self.mems)[0]
            
            #plt.errorbar(x, y, yerr, linewdith = 2, capsize = 6)