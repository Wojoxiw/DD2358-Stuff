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
    Stores data about the previous runtimes, to be saved/loaded with numpy. This holds them in a numpy array, and has some utility functions.
    Only the master (rank 0) process's times are saved - there shouldn't be much of a difference between processes, though.
    Only the total memory taken is saved - should be about equally spread between the processes.
    '''
    def __init__(self, folder, comm, filename = 'prevRuns.npz', otherPrevs = []):
        '''
        'Create' it every time - if it already exists, load the data. If not, we start empty
        :param folder: Folder where it is saved
        :param comm: MPI communicator
        :param filename: Filename to load/save this to
        :param otherPrevs: List of other filenames to load in - these should be in the same folder
        '''
        self.comm = comm
        if(self.comm.rank == 0):
            self.fpath = folder+filename
            if os.path.isfile(self.fpath):
                self.prevRuns = np.load(self.fpath, allow_pickle=True)['prevRuns']
                if(self.comm.rank == 0):
                    print(f'Loaded previous run data from {len(self.prevRuns)} runs')
                    
            if(len(otherPrevs) > 0): ## if there are other previous runs to load
                for other in otherPrevs:
                    runs = np.load(folder+other, allow_pickle=True)['prevRuns']
                    if(not hasattr(self, 'prevRuns')): ## if no other prev runs
                        self.prevRuns = runs
                    else:
                        self.prevRuns = np.hstack((self.prevRuns, runs))
                np.savez(self.fpath, prevRuns = self.prevRuns) ## after loading, save them all together
        
    class runTimeMem():
        '''
        The data of one previous run.
        '''
        def __init__(self, prob, extraInfo):
            '''
            Initialize the class.
            :param prob: The scattering problem run
            :param extraInfo: Optional string one can add to classify the run
            :param mem: Total memory used in the problem
            '''
            self.meshingTime = prob.refMeshdata.meshingTime
            self.calcTime = prob.calcTimes
            self.MPInum = prob.MPInum
            self.Nf = prob.Nf # number of frequencies - computations are for each frequency
            self.Nants = prob.refMeshdata.N_antennas # number of antennas - computations are for each antenna pair
            self.size = prob.refMeshdata.ncells # Computation size (number of FEM elements)
            self.mem = prob.memCost # Total memory cost
            self.extraInfo = extraInfo
            
    def memTimeAppend(self, run, extraInfo = ''):
        '''
        Takes a scatteringProblem class, updates the list and saves it
        :param run: The run
        :param extraInfo: Optional string one can add to classify the run
        '''
        if(self.comm.rank == 0): ## only use the master rank
            prev = np.empty(1, dtype=object)
            prev[0] = self.runTimeMem(run, extraInfo) ## the runTimeMem, in a numpy array
            if(hasattr(self, 'prevRuns')): ## check if these exist
                self.prevRuns = np.hstack((self.prevRuns, prev))
            else: ## hasn't been made yet
                self.prevRuns = prev
                
                
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
        
    def makePlotsSTD(self):
        '''
        Makes plots with standard deviation error bars for specific values (for DD2358). These are specific sizes, and MPInums
        Does its own version of calcStats, and makes plots herein
        '''
        binVals = [109624, 143465, 189130, 233557, 290155, 355864, 430880, 512558, 609766, 707748, 825148] ## the size-values (# elements) that were used for calculations
        MPInums = [1, 1, 12, 24] ## MPInum of runs to plot
        runType = [[1, 'noOMPNUMTHREADS'], [0, 'bindtocore'], [12, '2threads'],  [1, ''], [12, ''], [24, '']] ## MPInum + extraInfo of runs to plot. If zero, allows any
        
        if(self.comm.rank == 0):
            numRuns = len(self.prevRuns)
            fig1 = plt.figure()
            ax1 = plt.subplot(1, 1, 1)
            fig2 = plt.figure()
            ax2 = plt.subplot(1, 1, 1)
            ax1.grid(True)
            ax2.grid(True)
            ax1.set_title('Computation Time by Problem Size')
            ax2.set_title('Memory Cost by Problem Size')
            ax1.set_xlabel('# FEM Elements')
            ax2.set_xlabel('# FEM Elements')
            ax1.set_ylabel('Time [s]')
            ax2.set_ylabel('Memory [GiB]')
            
            for type in runType:
                MPInum = type[0]
                exInfo = type[1]
                avgstdTimes = np.zeros((2, len(binVals))) ## array of average and standard-deviations of computation times
                avgstdMems = np.zeros((2, len(binVals))) ## array of average and standard-deviations of memory costs
                for j in range(len(binVals)):
                    binVal = binVals[j]
                    ## first count the runs in this bin
                    l = 0
                    for i in range(numRuns):
                        run = self.prevRuns[i]
                        if(hasattr(run, 'extraInfo')):
                            eI = run.extraInfo
                        else:
                            eI = ''
                        if(np.isclose(run.size, binVal) and (run.MPInum == MPInum or MPInum == 0) and eI == exInfo): ## if correct size and MPInum, count
                            l+=1
                    ## then make the arrays
                    sizes = np.zeros(l)
                    mems = np.zeros(l)
                    times = np.zeros(l)
                    numProcesses = np.zeros(l)
                    Nfs = np.zeros(l)
                    Nants = np.zeros(l)
                    l = 0
                    for i in range(numRuns):
                        run = self.prevRuns[i]
                        if(hasattr(run, 'extraInfo')):
                            eI = run.extraInfo
                        else:
                            eI = ''
                        if(np.isclose(run.size, binVal) and (run.MPInum == MPInum or MPInum == 0) and eI == exInfo): ## if correct size and MPInum, fill in the array vals
                            sizes[l] = run.size
                            mems[l] = run.mem
                            times[l] = run.meshingTime + run.calcTime
                            numProcesses[l] = run.MPInum
                            Nfs[l] = run.Nf
                            Nants[l] = run.Nants
                            l+=1
                    avgstdTimes[0, j] = np.mean(times)
                    avgstdTimes[1, j] = np.std(times)
                    avgstdMems[0, j] = np.mean(mems)
                    avgstdMems[1, j] = np.std(mems)
                linestyle = '-' # default style
                if(exInfo == 'noOMPNUMTHREADS'):
                    label = '1 MPI Process, No NumThreads'
                elif(exInfo == 'bindtocore'):
                    label = 'bind-to-core - uses 24 processes'
                elif(exInfo == '2threads'):
                    label = '12 MPI Processes, 2 threads'
                elif(exInfo == 'MPI64nodes'):
                    label = '6 MPI Processes, 4 nodes'
                    linestyle = ':'
                elif(exInfo == 'MPIBTC4nodes'):
                    label = 'MPI BTC, 4 nodes'
                    linestyle = ':'
                elif(exInfo == 'MPI1NNT4nodes'):
                    label = 'MPI 1, 4 nodes'
                    linestyle = ':'
                elif(exInfo == 'MPI4-6T4nodes'):
                    label = 'MPI 4, 6 threads each, 4 nodes'
                    linestyle = ':'
                elif(MPInum == 1):
                    label = f'{MPInum} MPI Process'
                else:
                    label = f'{MPInum} MPI Processes'
                ax1.errorbar(binVals, avgstdTimes[0], avgstdTimes[1], linewidth = 2, capsize = 6, label = label, linestyle = linestyle)
                ax2.errorbar(binVals, avgstdMems[0], avgstdMems[1], linewidth = 2, capsize = 6, label = label, linestyle = linestyle)
                
            ax1.legend()
            ax2.legend()
            fig1.tight_layout()
            fig2.tight_layout()
            plt.show()