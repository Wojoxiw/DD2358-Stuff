# encoding: utf-8
## this file makes the mesh

import numpy as np
import dolfinx
from mpi4py import MPI
import gmsh
import sys
import pyvista
from scipy.constants import c as c0

class scatt3DProblem():
    """Class to hold definitions and functions for simulating scattering or transmission of electromagnetic waves for a rotationally symmetric structure."""
    def __init__(self,
                 meshdata,            # Mesh and metadata
                 f0=10e9,             # Frequency of the problem
                 epsr_bkg=1,          # Permittivity of the background medium
                 mur_bkg=1,           # Permeability of the background medium
                 material_epsr=1+0j,  # Permittivity of object
                 material_mur=1+0j,   # Permeability of object
                 hull_epsr=1+0j,      # Permittivity of defect
                 hull_mur=1+0j,       # Permeability of defect
                 degree=1,            # Degree of finite elements
                 comm=MPI.COMM_WORLD, # MPI communicator
                 model_rank=0,        # Model rank for saving, plotting etc
                 MPInum = 1,          # Number of MPI processes
                 Nf = 1,              # Number of frequency-points to calculate
                 ghost_ff_facets=None, 
                 ):
        """Initialize the problem."""
        
        self.tdim = 3                             # Dimension of triangles/tetraedra. 3 for 3D
        self.fdim = tdim - 1                      # Dimension of facets
        
        self.MPInum = MPInum                      # Number of MPI processes (used for estimating computational costs)
        
        self.lambda0 = c0/f0                      # Vacuum wavelength, used to define lengths in the mesh
        self.k0 = 2*np.pi*f0/c0                   # Vacuum wavenumber
        self.Nf = Nf
        self.epsr_bkg = epsr_bkg
        self.mur_bkg = mur_bkg
        self.n_bkg = np.sqrt(epsr_bkg*mur_bkg)    # Background refractive index
        self.etar_bkg = np.sqrt(mur_bkg/epsr_bkg) # Background relative wave impedance
        self.material_epsr = material_epsr
        self.material_mur = material_mur
        self.hull_epsr = hull_epsr
        self.hull_mur = hull_mur
        self.degree = degree
        self.ghost_ff_facets = ghost_ff_facets

        # Set up mesh information
        self.mesh = meshdata.mesh
        self.subdomains = meshdata.subdomains
        self.boundaries = meshdata.boundaries
        self.PML = meshdata.PML

        self.freespace_marker = meshdata.subdomain_markers['freespace']
        self.material_marker = meshdata.subdomain_markers['material']
        self.transition_marker = meshdata.subdomain_markers['transition']
        self.hull_marker = meshdata.subdomain_markers['hull']
        self.pml_marker = meshdata.subdomain_markers['pml']
        self.pml_hull_overlap_marker = meshdata.subdomain_markers['pml_hull_overlap']

        self.pec_surface_marker = meshdata.boundary_markers['pec']
        self.antenna_surface_marker = meshdata.boundary_markers['antenna']
        self.farfield_surface_marker = meshdata.boundary_markers['farfield']
        self.pml_surface_marker = meshdata.boundary_markers['pml']
        self.axis_marker = meshdata.boundary_markers['axis']

        self.comm = comm
        self.model_rank = model_rank
        max_r_local = np.sqrt(self.mesh.geometry.x[:,0]**2 + self.mesh.geometry.x[:,1]**2).max()
        max_r_locals = self.comm.gather(max_r_local, root=model_rank)
        max_rho_local = self.mesh.geometry.x[:,0].max()
        max_rho_locals = self.comm.gather(max_rho_local, root=model_rank)
        if self.comm.rank == model_rank:
            max_r = np.max(max_r_locals)
            max_rho = np.max(max_rho_locals)
        else:
            max_r = None
            max_rho = None
        max_r = self.comm.bcast(max_r, root=model_rank)
        max_rho = self.comm.bcast(max_rho, root=model_rank)
        self.max_r = max_r
        self.max_rho = max_rho
        
        # Initialize function spaces, boundary conditions, PML
        self.InitializeFEM()
        self.InitializeFarfieldCells()
        self.InitializeMaterial()
        self.InitializePML()