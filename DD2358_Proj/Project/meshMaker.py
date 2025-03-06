# encoding: utf-8
## this file makes the mesh

import os ## presumably dont need this import here
os.environ["OMP_NUM_THREADS"] = "1" # seemingly needed for MPI speedup
from mpi4py import MPI
import numpy as np
import dolfinx
import gmsh
from scipy.constants import pi, c as c0
from timeit import default_timer as timer
import sys

##memory profiling
from memory_profiler import profile

class MeshData():
    """Data structure for the mesh (all geometry) and related metadata."""
    def __init__(self,
                 comm,
                 fname = '',
                 reference = True,
                 f0 = 10e9,
                 verbosity = 0,
                 model_rank = 0,
                 h = 1/13,
                 domain_geom = 'domedCyl', 
                 object_geom = 'sphere',
                 defect_geom = 'cylinder',
                 domain_radius = 1.3,
                 domain_height = 0.9,
                 PML_thickness = .5,
                 dome_height = 0.3,
                 antenna_width = 0.7625, 
                 antenna_height = 0.3625,
                 antenna_depth = 1/10,      
                 N_antennas = 3,
                 antenna_radius = 0,
                 antenna_z_offset = 0,
                 object_radius = 0.46,
                 defect_radius = 0.2,
                 defect_height = 0.5,
                 defect_angles = [45, 67, 32],
                 viewGMSH = False,
                ):
        '''
        Makes it - given various inputs
        :param comm: the MPI communicator
        :param fname: Mesh filename + directory. If empty, does not save it (currently does not ever save it)
        :param reference: Does not include any defects in the mesh.
        :param f0: Design frequency - things will be scaled by the corresponding wavelength
        :param h: typical mesh size, in fractions of a wavelength
        :param verbosity: This is passed to gmsh, also if > 0, I print more stuff
        :param model_rank: Rank of the master model - for saving, plotting, etc.
        :param domain_geom: The geometry of the domain (and PML).
        :param object_geom: Geometry of the object
        :param defect_geom: Geometry of the defect.
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
        :param viewGMSH: If True, plots the mesh after creation then exits
        '''
        
        self.comm = comm                               # MPI communicator
        self.model_rank = model_rank                   # Model rank
        self.lambda0 = c0/f0                         # Design wavelength (things are scaled from this)
        if(fname == ''): # if not given a name, just use 'mesh'
            fname = 'mesh.msh'
        self.fname = fname                                  # Mesh filename (+location from script)
        self.h = h * self.lambda0
        self.domain_geom = domain_geom            # setting to choose domain geometry - current only 'domedCyl' exists
        self.verbosity = verbosity
        self.reference = reference
        
        ## Set up geometry variables - in units of lambda0 unless otherwise specified
        if(self.domain_geom == 'domedCyl'): ## a cylinder with an oblate-spheroid 'dome' added over and underneath
            self.domain_radius = domain_radius * self.lambda0
            self.domain_height = domain_height * self.lambda0
            
            self.PML_radius = self.domain_radius + PML_thickness * self.lambda0
            self.PML_height = self.domain_height + 2*PML_thickness * self.lambda0
            
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
        
        self.N_antennas = N_antennas ## number of antennas
        if(antenna_radius == 0): ## if not given a radius, put them near the edge of the domain
            self.antenna_radius = self.domain_radius - antenna_height * self.lambda0
        else:
            self.antenna_radius = antenna_radius * self.lambda0
        self.antenna_z_offset = antenna_z_offset * self.lambda0
        self.antenna_width = antenna_width * self.lambda0
        self.antenna_height = antenna_height * self.lambda0
        self.antenna_depth = antenna_depth * self.lambda0
        self.phi_antennas = np.linspace(0, 2*pi, N_antennas + 1)[:-1] ## placement angles
        self.pos_antennas = np.array([[self.antenna_radius*np.cos(phi), self.antenna_radius*np.sin(phi), self.antenna_z_offset] for phi in self.phi_antennas]) ## placement positions
        self.rot_antennas = self.phi_antennas + np.pi/2 ## rotation so that they face the center
        self.kc = pi/self.antenna_width ## cutoff wavenumber
        ## Object + defect(s) parameters
        if(object_geom == 'sphere'):
            self.object_geom = object_geom
            self.object_radius = object_radius * self.lambda0
        else:
            print('Nonvalid object geom, exiting...')
            exit()
        
        self.defect_angles = defect_angles ## [x, y, z] rotations
        if(defect_geom == 'cylinder'):
            self.defect_geom = defect_geom
            self.defect_radius = defect_radius * self.lambda0
            self.defect_height = defect_height * self.lambda0
        elif(defect_geom == ''):
            pass ## no defect
        else:
            print('Nonvalid defect geom, exiting...')
            exit()
            
        ## Finally, actually make the mesh
        self.createMesh(viewGMSH)
        
    @profile
    def createMesh(self, viewGMSH):
        t1 = timer()
        tdim = 3 ## Tetrahedra dimensionality - 3D
        fdim = tdim - 1 ## Facet dimensionality - 2D
        gmsh.initialize()
         
        if self.comm.rank == self.model_rank: ## make all the definitions through the master-rank process
            
            ## Give some mesh settings: verbosity, max. and min. mesh lengths
            gmsh.option.setNumber('General.Verbosity', self.verbosity)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.h)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.h)
             
            inPECSurface = []; inAntennaSurface = []; antennas_DimTags = []
            ## Make the antennas
            x_antenna = np.zeros((self.N_antennas, 3))
            x_pec = np.zeros((self.N_antennas, 5, 3)) ### for each antenna, and PEC surface (of which there are 5), a position of that surface
            for n in range(self.N_antennas): ## make each antenna, and prepare its surfaces to either be PEC or be the excitation surface
                box = gmsh.model.occ.addBox(-self.antenna_width/2, -self.antenna_depth, -self.antenna_height/2, self.antenna_width, self.antenna_depth, self.antenna_height) ## the antenna surface at (0, 0, 0)
                gmsh.model.occ.rotate([(tdim, box)], 0, 0, 0, 0, 0, 1, self.rot_antennas[n])
                gmsh.model.occ.translate([(tdim, box)], self.pos_antennas[n,0], self.pos_antennas[n,1], self.pos_antennas[n,2])
                antennas_DimTags.append((tdim, box))
                x_antenna[n] = self.pos_antennas[n, :] ## the translation to the antenna's position
                Rmat = np.array([[np.cos(self.rot_antennas[n]), -np.sin(self.rot_antennas[n]), 0],
                                 [np.sin(self.rot_antennas[n]), np.cos(self.rot_antennas[n]), 0],
                                 [0, 0, 1]]) ## matrix for rotation about the z-axis
                x_pec[n, 0] = x_antenna[n] + np.dot(Rmat, np.array([0, -self.antenna_depth/2, -self.antenna_height/2])) ## bottom surface (in z)
                x_pec[n, 1] = x_antenna[n] + np.dot(Rmat, np.array([0, -self.antenna_depth/2,  self.antenna_height/2])) ## top surface (in z)
                x_pec[n, 2] = x_antenna[n] + np.dot(Rmat, np.array([-self.antenna_width/2, -self.antenna_depth/2, 0])) ## left surface (in x)
                x_pec[n, 3] = x_antenna[n] + np.dot(Rmat, np.array([self.antenna_width/2, -self.antenna_depth/2, 0])) ## right surface (in x)
                x_pec[n, 4] = x_antenna[n] + np.dot(Rmat, np.array([0, -self.antenna_depth, 0])) ## back surface (in y)
                inAntennaSurface.append(lambda x: np.allclose(x, x_antenna[n])) ## (0, 0, 0) - the antenna surface
                inPECSurface.append(lambda x: np.allclose(x, x_pec[n,0]) or np.allclose(x, x_pec[n,1]) or np.allclose(x, x_pec[n,2]) or np.allclose(x, x_pec[n,3]) or np.allclose(x, x_pec[n,4]))
        
            matDimTags = []; defectDimTags = []
            ## Make the object and defects (if not a reference case)
            if(self.object_geom == 'sphere'):
                obj = gmsh.model.occ.addSphere(0,0,0, self.object_radius) ## add it to the origin
                matDimTags.append((tdim, obj)) ## the material fills the object
            if(not self.reference):
                if(self.defect_geom == 'cylinder'):
                    defect1 = gmsh.model.occ.addCylinder(0,0,-self.defect_height/2,0,0,self.defect_height, self.defect_radius) ## cylinder centered on the origin
                    ## apply some rotations around the origin, and each axis
                    gmsh.model.occ.rotate([(tdim, defect1)], 0, 0, 0, 1, 0, 0, self.defect_angles[0])
                    gmsh.model.occ.rotate([(tdim, defect1)], 0, 0, 0, 0, 1, 0, self.defect_angles[1])
                    gmsh.model.occ.rotate([(tdim, defect1)], 0, 0, 0, 0, 0, 1, self.defect_angles[2])
                    defectDimTags.append((tdim, defect1))
             
            ## Make the domain and the PML
            if(self.domain_geom == 'domedCyl'):
                domain_cyl = gmsh.model.occ.addCylinder(0, 0, -self.domain_height/2, 0, 0, self.domain_height, self.domain_radius)
                domain = [(tdim, domain_cyl)] # dim, tags
                pml_cyl = gmsh.model.occ.addCylinder(0, 0, -self.PML_height/2, 0, 0, self.PML_height, self.PML_radius)
                pml = [(tdim, pml_cyl)] # dim, tags
                if(self.dome_height>0): ## add a spheroid domed top and bottom with some specified extra height, that passes through the cylindrical 'corner' (to try to avoid waves being parallel to the PML)
                    domain_spheroid = gmsh.model.occ.addSphere(0, 0, 0, self.domain_radius+self.domain_spheroid_extraRadius)
                    gmsh.model.occ.dilate([(tdim, domain_spheroid)], 0, 0, 0, 1, 1, self.domain_a)
                    domain_extraheight_cyl = gmsh.model.occ.addCylinder(0, 0, -self.domain_height/2-self.dome_height, 0, 0, self.domain_height+self.dome_height*2, self.domain_radius)
                    domed_ceilings = gmsh.model.occ.intersect([(tdim, domain_spheroid)], [(tdim, domain_extraheight_cyl)])
                    domain = gmsh.model.occ.fuse([(tdim, domain_cyl)], domed_ceilings[0])[0] ## [0] to get  dimTags
                    
                    pml_spheroid = gmsh.model.occ.addSphere(0, 0, 0, self.PML_radius+self.PML_spheroid_extraRadius)
                    gmsh.model.occ.dilate([(tdim, pml_spheroid)], 0, 0, 0, 1, 1, self.PML_a)
                    pml_extraheight_cyl = gmsh.model.occ.addCylinder(0, 0, -self.PML_height/2-self.dome_height, 0, 0, self.PML_height+self.dome_height*2, self.PML_radius)
                    domed_ceilings = gmsh.model.occ.intersect([(tdim, pml_spheroid)], [(tdim, pml_extraheight_cyl)])
                    pml = gmsh.model.occ.fuse([(tdim, pml_cyl)], domed_ceilings[0])[0]
            elif(self.domain_geom == 'sphere'):
                domain = gmsh.model.occ.addSphere(0, 0, 0, self.domain_radius)
                pml = gmsh.model.occ.addSphere(0, 0, 0, self.PML_radius)
            
            # Create fragments and dimtags
            outDimTags, outDimTagsMap = gmsh.model.occ.fragment(pml, domain + matDimTags + defectDimTags + antennas_DimTags)
            removeDimTags = [x for x in [y[0] for y in outDimTagsMap[-self.N_antennas:]]]
            defectDimTags = [x[0] for x in outDimTagsMap[3:] if x[0] not in removeDimTags]
            matDimTags = [x for x in outDimTagsMap[2] if x not in defectDimTags]
            domainDimTags = [x for x in outDimTagsMap[1] if x not in removeDimTags+matDimTags+defectDimTags]
            pmlDimTags = [x for x in outDimTagsMap[0] if x not in domainDimTags+defectDimTags+matDimTags+removeDimTags]
            gmsh.model.occ.remove(removeDimTags)
            gmsh.model.occ.synchronize()
            
            # Make physical groups for domains and PML
            mat_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in matDimTags])
            defect_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in defectDimTags])
            domain_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in domainDimTags])
            pml_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in pmlDimTags])
            
            # Identify antenna surfaces and make physical groups (by checking the Center-of-Mass of each surface entity)
            pec_surface = []
            antenna_surface = []
            for boundary in gmsh.model.occ.getEntities(dim=fdim):
                CoM = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
                for n in range(len(inPECSurface)): ## iterate over all of these
                    if inPECSurface[n](CoM):
                        pec_surface.append(boundary[1])
                for n in range(len(inAntennaSurface)): ## iterate over all of these
                    if inAntennaSurface[n](CoM):
                        antenna_surface.append(boundary[1])
            pec_surface_marker = gmsh.model.addPhysicalGroup(fdim, pec_surface)
            antenna_surface_markers = [gmsh.model.addPhysicalGroup(fdim, [s]) for s in antenna_surface]
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(tdim)
            #gmsh.write(self.fname) ## Write the mesh to a file. I have never actually looked at this, so I've commented it out
            
            if(viewGMSH):
                gmsh.fltk.run()
                exit()
            
        else: ## some data is also needed for subordinate processes
            mat_marker = None
            defect_marker = None
            domain_marker = None
            pml_marker = None
            pec_surface_marker = None
            antenna_surface_markers = None
            
        self.mat_marker = self.comm.bcast(mat_marker, root=self.model_rank)
        self.defect_marker = self.comm.bcast(defect_marker, root=self.model_rank)
        self.domain_marker = self.comm.bcast(domain_marker, root=self.model_rank)
        self.pml_marker = self.comm.bcast(pml_marker, root=self.model_rank)
        self.pec_surface_marker = self.comm.bcast(pec_surface_marker, root=self.model_rank)
        self.antenna_surface_markers = self.comm.bcast(antenna_surface_markers, root=self.model_rank)
    
        
        self.mesh, self.subdomains, self.boundaries = dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm=self.comm, rank=self.model_rank, gdim=tdim)
        gmsh.finalize()
        
        self.ncells = self.mesh.topology.index_map(self.mesh.topology.dim).size_global ## all cells, rather than only those in this MPI process
        
        self.meshingTime = timer() - t1 ## Time it took to make the mesh. Given to mem-time estimator
        if(self.verbosity > 0 and self.comm.rank == self.model_rank): ## start by giving some estimated calculation times/memory costs
            nloc = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
            print(f'Mesh generated in {self.meshingTime:.2e} s - {self.ncells} global cells, {nloc} local cells')
            sys.stdout.flush()
            