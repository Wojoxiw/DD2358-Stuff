'''
Some utility scripts.

Created on 27 feb. 2025
@author: al8032pa
'''
import numpy as np
from matplotlib import pyplot as plt
import scipy
import os

def memTimeEstimation(numCells = 0, Nf = 0, printPlots = False):
    '''
    Estimates the execution time and memory requirements of the Scatt3d run, based on previous runs.
    
    Previous run information is stored in prevRuns.info in the same folder as this script.
    Assumes that the memory cost scales with the volume of the computation/h^3 (the number of mesh cells).
    Time taken should then scale with the memory times the number of frequency points.
    
    
    :param numCells: estimated number of mesh cells, if asking for an estimated time/memory cost
    :param Nf: number of freq. points, when asking for an estimated time
    :param printPlots: if True, plots the memory and time requirements of previous runs, along with the fit used for estimation
    '''
    data = np.loadtxt('prevRuns.info', skiprows = 2) ## mems, times, ncells, Nfs
    
    line = lambda x, a, b: a*x + b # just fit the data to a line
    
    ###############
    # TIME STUFF
    ###############
    idx = np.argsort(data[:,1]) ## sort by time
    times, ncells, Nfs = data[idx, 1], data[idx, 2], data[idx, 3]
    fit = scipy.optimize.curve_fit(line, ncells*Nfs, times)[0]
    time = line(numCells*Nf, fit[0], fit[1])
    
    if(printPlots):
        xs = np.linspace(np.min(ncells*Nfs), np.max(ncells*Nfs), 1000)
        plt.plot(ncells, times/3600, '-o', label='runs on computer')
        plt.title('Computation time by size')
        plt.xlabel(r'(# of mesh cells)*(# of frequencies)')
        plt.ylabel('Time [hours]')
        plt.grid()
        plt.plot(xs, line(xs, fit[0], fit[1])/3600, label='curve_fit')
        if(numCells>0 and Nf>0):
            plt.scatter(numCells*Nf, time/3600, s = 80, facecolors = None, edgecolors = 'red', label = 'Estimated Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    ###############
    # MEMORY STUFF
    ###############
    idxMemsRecorded = np.argwhere(data[:,0] > 0)[:, 0] ## only take data where the memory cost is actually recorded, for memory estimation (this should be always now)
    mems, ncells = data[idxMemsRecorded, 0], data[idxMemsRecorded, 2]
    idx = np.argsort(mems) ## sort by mem
    mems, ncells= mems[idx], ncells[idx]
    
    fitMem = scipy.optimize.curve_fit(line, ncells, mems)[0]
    mem = line(numCells, fitMem[0], fitMem[1])
    
    if(printPlots):
        xs = np.linspace(np.min(ncells), np.max(ncells), 1000)
        plt.plot(ncells, mems, '-o', label='runs on computer')
        plt.title('Memory Requirements by size')
        plt.xlabel(r'# of mesh cells')
        plt.ylabel('Memory [GB] (Approximate)')
        plt.grid()
        plt.plot(xs, line(xs, fitMem[0], fitMem[1]), label='curve_fit')
        if(numCells>0 and Nf>0):
            plt.scatter(numCells, mem, s = 80, facecolors = None, edgecolors = 'red', label = 'Estimated Memory')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    
    return mem, time

def memTimeAppend(numCells, Nf, mem, time, reference, folder = ''):
    '''
    Appends a run's data to the estimation datafile
    
    :param folder: folder to store and retrieve prevRuns.info
    :param numCells: estimated number of mesh cells, if asking for an estimated time/memory cost
    :param Nf: number of freq. points, when asking for an estimated time
    :param mem: max. memory usage, in GiB
    :param time: total runTime, in s
    :param reference: True if this is a reference run (prints 1), False (or 0) otherwise
    '''
    file = open(folder+'prevRuns.info','a')
    #file.write("\n")
    np.savetxt(file, np.array([mem, time, numCells, Nf, reference]).reshape(1, 5), fmt='%1.5e')
    file.close()
    
class runTimesMems():
    '''
    Stores data about the previous runtimes, to be saved/loaded with numpy. This holds them like a list, and has some utility functions.
    '''
    def __init__(self, folder):
        self.fpath = folder+'prevRuns.npz'
        if os.path.isfile(self.fpath):
            np.load(self.fpath)['prevRuns']
        else:
            np.savez(self.fpath, prevRuns = self)
        
    class runTimeMem():
        '''
        The data of one previous run.
        '''
        def __init__(self, meshingTime, calcTime, MPInum):
            self.meshingTime = meshingTime
            self.calcTime = calcTime
            self.MPInum = MPInum
            
    def memTimeAppend(self):
        pass
    
    def memTimeEstimation(self):
        pass
    
    def showPlots(self):
        pass