# encoding: utf-8
## this file makes the mesh

import numpy as np
import dolfinx
from mpi4py import MPI
import gmsh
import sys
import pyvista
from scripts import memTimeEstimation
from scipy.constants import pi, c as c0

class MeshData():
    """Data structure for the mesh (all geometry) and related metadata."""
    def __init__(self,  
                 problem,           
                 fname = '',
                 verbosity = 0,     
                 h = 1/15,          
                 domain_geom = 'domedCyl', 
                 object_geom = 'sphere',
                 defect_geom = 'cylinder',
                 domain_radius = 2,
                 domain_height = 1.5,
                 PML_thickness = 1,
                 dome_height = 0.3,
                 antenna_width = 0.7625, 
                 antenna_height = 0.3625,
                 antenna_depth = 1/10,      
                 N_antennas = 7,
                 antenna_radius = 0,
                 antenna_z_offset = 0,
                 object_radius = 0.5,
                 defect_radius = 0.2,
                 defect_height = 0.6,
                 defect_angles = [45, 67, 32],
                ):
        '''
        Makes it - given various inputs
        :param problem: The scattering problem which is creating this mesh. Some variables are copied/used from it.
        :param fname: Mesh filename (+location from script)
        :param verbosity: Verbosity passed on to gmsh. If greater than 0, also print some computation times/sizes maybe
        :param h: typical mesh size, in fractions of a wavelength
        :param domain_geom: The geometry of the domain (and PML). Only the default for now
        :param object_geom: Geometry of the object. Only the default for now
        :param object_geom: Geometry of the defect. Only the default for now
        :param domain_radius:
        :param domain_height:
        :param PML_thickness:
        :param dome_height:
        :param antenna_width: Width of antenna apertures, 22.86 mm
        :param antenna_height: Height of antenna apertures
        :param antenna_depth: Depth of antenna box
        :param N_antennas:
        :param antenna_radius: Radius at which antennas are placed
        :param antenna_z_offset: Height (from the middle of the sim.) at which antennas are placed. Default to centering on the x-y plane
        :param object_radius: If object is a sphere (or cylinder), the radius
        :param defect_radius: If defect is a sphere (or cylinder), the radius
        :param defect_height: If defect is a cylinder, the height
        :param defect_angles: [x, y, z] angles to rotate about these axes
        '''
        
        self.prob = problem                                 # problem will contain this mesh, which links back to grab some info
        self.comm = self.prob.comm                               # MPI communicator
        self.model_rank = self.prob.model_rank                   # Model rank
        self.lambda0 = self.prob.lambda0                         # Design wavelength (things are scaled from this)
        if(fname == ''): # if not given a name, use the problem's name
            fname = problem.dataFolder+problem.name+'mesh.msh'
        self.fname = fname                                  # Mesh filename (+location from script)
        self.verbosity = verbosity
        self.h = h
        self.domain_geom = domain_geom            # setting to choose domain geometry - current only 'domedCyl' exists
        
        ## Set up geometry variables - in units of lambda0 unless otherwise specified
        if(self.domain_geom == 'domedCyl'): ## a cylinder with an oblate-spheroid 'dome' added over and underneath
            self.domain_radius = domain_radius * self.lambda0
            self.domain_height = domain_height * self.lambda0
            
            self.PML_radius = (self.domain_radius + PML_thickness) * self.lambda0
            self.PML_height = (self.domain_height + 2*PML_thickness) * self.lambda0
            
            self.dome_height = dome_height * self.lambda0 # How far beyond the domain the cylindrical base the spheroid extends
            if(self.dome_height > 0): ## if 0, just make the cylinder. Otherwise, calculate the spheroid parameters R_extra and a
                self.domain_spheroid_extraRadius = self.domain_radius*(-1 + np.sqrt(1-( 1 - 1/(1- (self.domain_height/2/(self.domain_height/2+self.dome_height))**2 ) )) )       
                self.domain_a = (self.domain_height/2+self.dome_height)/(self.domain_radius+self.domain_spheroid_extraRadius)
                self.PML_spheroid_extraRadius = self.PML_radius*(-1 + np.sqrt(1-( 1 - 1/(1- (self.PML_height/2/(self.PML_height/2+self.dome_height))**2 ) )))
                self.PML_a = (self.PML_height/2+self.dome_height)/(self.PML_radius+self.PML_spheroid_extraRadius)
        else:
            print('Invalid geometry type in MeshData, exiting...')
            exit()
        
        ## Antenna geometry/other parameters:
        self.kc = pi/antenna_width ## cutoff wavenumber
        self.N_antennas = N_antennas ## number of antennas
        if(antenna_radius == 0): ## if not given a radius, put them near the edge of the domain
            self.antenna_radius = domain_radius - antenna_height
        else:
            self.antenna_radius = antenna_radius
        self.antenna_z_offset = antenna_z_offset
        self.antenna_width = antenna_width
        self.antenna_height = antenna_height
        self.antenna_depth = antenna_depth
        self.phi_antennas = np.linspace(0, 2*pi, N_antennas + 1)[:-1] ## placement angles
        self.pos_antennas = np.array([[self.antenna_radius*np.cos(phi), self.antenna_radius*np.sin(phi), self.antenna_z_offset] for phi in self.phi_antennas]) ## placement positions
        self.rot_antennas = self.phi_antennas + np.pi/2 ## rotation so that they face the center
        
        ## Object + defect(s) parameters
        if(object_geom == 'sphere'):
            self.object_geom = object_geom
            self.object_radius = object_radius * self.lambda0
        else:
            print('Nonvalid object geom, exiting...')
            exit()
        
        if(defect_geom == 'cylinder'):
            self.defect_geom = defect_geom
            self.defect_radius = defect_radius * self.lambda0
            self.defect_height = defect_height * self.lambda0
        else:
            print('Nonvalid defect geom, exiting...')
            exit()
        
        ## Finally, actually make the mesh
        self.createMesh()
        
    def createMesh(self):
        def makeAntennas(): ## makes the antennas
            pass
        
        def makeObject(): ## makes the object and defects
            pass
        
        gmsh.initialize()
        if self.comm.rank == self.model_rank: ## make all the definitions through the master-rank process
            if(self.verbosity > 0): ## start by giving some estimated calculation times/memory costs
                size = pi*self.PML_radius**2*self.PML_height/self.h**3 *4 ### a rough estimation. added factor at the end to get closer
                estmem, esttime = memTimeEstimation(size, self.problem.Nf)
                print('Variables created, generating mesh...')
                print(f'Estimated memory requirement for size {size:.3e}: {estmem:.3f} GB')
                print(f'Estimated computation time for size {size:.3e}, Nf = {self.problem.Nf}: {esttime/3600:.3f} hours')
            
            ## Give some mesh settings: verbosity, max. and min. mesh lengths
            gmsh.option.setNumber('General.Verbosity', self.verbosity)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.h)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.h)
            
            makeAntennas()
            makeObject()
            
            