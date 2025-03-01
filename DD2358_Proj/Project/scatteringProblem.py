# encoding: utf-8
## this file makes the mesh

import numpy as np
import dolfinx
import ufl
import basix
from mpi4py import MPI
import gmsh
import sys
import pyvista
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0
eta0 = np.sqrt(mu0/eps0)

class scatt3DProblem():
    """Class to hold definitions and functions for simulating scattering or transmission of electromagnetic waves for a rotationally symmetric structure."""
    def __init__(self,
                 refMeshdata, # Mesh and metadata for the reference case
                 DUTMeshdata, # Mesh and metadata for the DUT case - should just include defects into the object (this will change the mesh)
                 comm, # MPI communicator
                 verbosity = 0,   ## if > 0, I print more stuff
                 f0=10e9,             # Frequency of the problem
                 epsr_bkg=1,          # Permittivity of the background medium
                 mur_bkg=1,           # Permeability of the background medium
                 material_epsr=3.0*(1 - 0.01j),  # Permittivity of object
                 material_mur=1+0j,   # Permeability of object
                 defect_epsr=2.5*(1 - 0.01j),      # Permittivity of defect
                 defect_mur=1+0j,       # Permeability of defect
                 fem_degree=1,            # Degree of finite elements
                 model_rank=0,        # Rank of the master model - for saving, plotting, etc.
                 MPInum = 1,          # Number of MPI processes
                 freqs = [],          # Optionally, pass in the exact frequencies to calculate  
                 Nf = 1,              # Number of frequency-points to calculate
                 BW = 4e9,              # Bandwidth (equally spaced around f0)
                 dataFolder = 'data3D/', # Folder where data is stored
                 pol = 'vert', ## Polarization of the antenna excitation - can be either 'vert' or 'horiz' (untested)
                 computeImmediately = True, ## compute solutions at the end of initialization
                 computeReference = False, # Computes the reference simulation, where no defects are included
                 ):
        """Initialize the problem."""
        
        self.dataFolder = dataFolder
        self.MPInum = MPInum                      # Number of MPI processes (used for estimating computational costs)
        self.comm = comm
        self.model_rank = model_rank
        self.verbosity = verbosity
        self.computeReference = computeReference
        
        self.tdim = 3                             # Dimension of triangles/tetraedra. 3 for 3D
        self.fdim = self.tdim - 1                      # Dimension of facets
        
        self.lambda0 = c0/f0                      # Vacuum wavelength, used to define lengths in the mesh
        self.k0 = 2*np.pi*f0/c0                   # Vacuum wavenumber
        
        if(len(freqs) > 0): ## if given frequency points, use those
            Nf = len(freqs)
            self.fvec = freqs  # Vector of simulation frequencies
        else:
            self.Nf = Nf
            self.fvec = np.linspace(f0-BW/2, f0+BW/2, Nf)  # Vector of simulation frequencies
            
        self.epsr_bkg = epsr_bkg
        self.mur_bkg = mur_bkg
        self.material_epsr = material_epsr
        self.material_mur = material_mur
        self.defect_epsr = defect_epsr
        self.defect_mur = defect_mur
        self.fem_degree = fem_degree
        self.antenna_pol = pol

        # Set up mesh information
        self.refMeshdata = refMeshdata
        self.refMeshdata.mesh.topology.create_connectivity(self.tdim, self.tdim)
        
        # Initialize function spaces, boundary conditions, and PML
        self.InitializeFEM()
        self.InitializeMaterial()
        self.InitializePML()
        
        # Calculations solutions
        if(computeImmediately):
            self.ComputeSolutions()
        
    def InitializeFEM(self):
        # Set up some FEM function spaces and boundary condition stuff.
        curl_element = basix.ufl.element('N1curl', self.refMeshdata.mesh.basix_cell(), self.fem_degree)
        self.Vspace = dolfinx.fem.functionspace(self.refMeshdata.mesh, curl_element)
        
        # Create measures for subdomains and surfaces
        self.dx = ufl.Measure('dx', domain=self.refMeshdata.mesh, subdomain_data=self.refMeshdata.subdomains, metadata={'quadrature_degree': 5})
        self.dx_dom = self.dx((self.refMeshdata.domain_marker, self.refMeshdata.mat_marker, self.refMeshdata.defect_marker))
        self.dx_pml = self.dx(self.refMeshdata.pml_marker)
        self.ds = ufl.Measure('ds', domain=self.refMeshdata.mesh, subdomain_data=self.refMeshdata.boundaries)
        self.ds_antennas = [self.ds(m) for m in self.refMeshdata.antenna_surface_markers]
        self.ds_pec = self.ds(self.refMeshdata.pec_surface_marker)
        self.pec_dofs = dolfinx.fem.locate_dofs_topological(self.Vspace, entity_dim=self.fdim, entities=self.refMeshdata.boundaries.find(self.refMeshdata.pec_surface_marker))
        self.Ezero = dolfinx.fem.Function(self.Vspace)
        self.Ezero.x.array[:] = 0.0
        self.bc_pec = dolfinx.fem.dirichletbc(self.Ezero, self.pec_dofs)
        
    def InitializeMaterial(self):
        # Set up material parameters. Not chancing mur for now, need to edit this if doing so
        Wspace = dolfinx.fem.functionspace(self.refMeshdata.mesh, ("DG", 0))
        self.epsr = dolfinx.fem.Function(Wspace)
        self.mur = dolfinx.fem.Function(Wspace)
        self.epsr.x.array[:] = self.epsr_bkg
        self.mur.x.array[:] = self.mur_bkg
        mat_cells = self.refMeshdata.subdomains.find(self.refMeshdata.mat_marker)
        mat_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=self.tdim, entities=mat_cells)
        defect_cells = self.refMeshdata.subdomains.find(self.refMeshdata.defect_marker)
        defect_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=self.tdim, entities=defect_cells)
        self.epsr.x.array[mat_dofs] = self.material_epsr
        self.mur.x.array[mat_dofs] = self.material_mur
        self.epsr.x.array[defect_dofs] = self.material_epsr
        self.mur.x.array[defect_dofs] = self.material_mur
        self.epsr_array_ref = self.epsr.x.array.copy()
        self.epsr.x.array[defect_dofs] = self.defect_epsr
        self.mur.x.array[defect_dofs] = self.defect_mur
        self.epsr_array_dut = self.epsr.x.array.copy()
        
    def InitializePML(self):
        # Set up the PML
        def pml_stretch(y, x, k0, x_dom=0, x_pml=1, n=3, R0=1e-10):
            '''
            Calculates the PML stretching of a coordinate
            :param y: the coordinate to be stretched
            :param x: the coordinate the stretching is based on
            :param k0: wavenumber
            :param x_dom: size of domain
            :param x_pml: size of pml
            :param n: order
            :param R0: intended damping (based on relative strength of reflection?)
            '''
            return y*(1 - 1j*(n + 1)*np.log(1/R0)/(2*k0*np.abs(x_pml - x_dom))*((x - x_dom)/(x_pml - x_dom))**n)

        def pml_epsr_murinv(pml_coords):
            '''
            Transforms epsr, mur, using the given stretched coordinates (this implements the pml)
            :param pml_coords: the coordinates
            '''
            J = ufl.grad(pml_coords)
            A = ufl.inv(J)
            epsr_pml = ufl.det(J) * A * self.epsr * ufl.transpose(A)
            mur_pml = ufl.det(J) * A * self.mur * ufl.transpose(A)
            murinv_pml = ufl.inv(mur_pml)
            return epsr_pml, murinv_pml
        
        if(self.refMeshdata.domain_geom == 'domedCyl'): ## implement it for this geometry
            x, y, z = ufl.SpatialCoordinate(self.refMeshdata.mesh)
            r = ufl.real(ufl.sqrt(x**2 + y**2)) ## cylindrical radius. need to set this to real because I compare against it later
            self.refMeshdata.domain_height_spheroid = ufl.conditional(ufl.ge(r, self.refMeshdata.domain_radius), self.refMeshdata.domain_height/2, (self.refMeshdata.domain_height/2+self.refMeshdata.dome_height)*ufl.real(ufl.sqrt(1-(r/(self.refMeshdata.domain_radius+self.refMeshdata.domain_spheroid_extraRadius))**2)))   ##start the z-stretching at, at minmum, the height of the domain cylinder
            self.refMeshdata.PML_height_spheroid = (self.refMeshdata.PML_height/2+self.refMeshdata.dome_height)*ufl.sqrt(1-(r/(self.refMeshdata.PML_radius+self.refMeshdata.PML_spheroid_extraRadius))**2) ##should always be ge the pml cylinder's height
            x_stretched = pml_stretch(x, r, self.k0, x_dom=self.refMeshdata.domain_radius, x_pml=self.refMeshdata.PML_radius)
            y_stretched = pml_stretch(y, r, self.k0, x_dom=self.refMeshdata.domain_radius, x_pml=self.refMeshdata.PML_radius)
            z_stretched = pml_stretch(z, abs(z), self.k0, x_dom=self.refMeshdata.domain_height_spheroid, x_pml=self.refMeshdata.PML_height_spheroid) ## /2 since the height is from - to +
            x_pml = ufl.conditional(ufl.ge(abs(r), self.refMeshdata.domain_radius), x_stretched, x) ## stretch when outside radius of the domain
            y_pml = ufl.conditional(ufl.ge(abs(r), self.refMeshdata.domain_radius), y_stretched, y) ## stretch when outside radius of the domain
            z_pml = ufl.conditional(ufl.ge(abs(z), self.refMeshdata.domain_height_spheroid), z_stretched, z) ## stretch when outside the height of the cylinder of the domain (or oblate spheroid roof with factor a_dom/a_pml - should only be higher/lower inside the domain radially)
        pml_coords = ufl.as_vector((x_pml, y_pml, z_pml))
        self.epsr_pml, self.murinv_pml = pml_epsr_murinv(pml_coords)

        rho, z = ufl.SpatialCoordinate(self.mesh)
        if not self.PML.cylindrical: # Spherical PML
            r = ufl.sqrt(rho**2 + z**2)
            rho_stretched = pml_stretch(rho, r, self.k0, x_dom=self.PML.radius-self.PML.d, x_pml=self.PML.radius)
            z_stretched = pml_stretch(z, r, self.k0, x_dom=self.PML.radius-self.PML.d, x_pml=self.PML.radius)
            rho_pml = ufl.conditional(ufl.ge(abs(r), self.PML.radius-self.PML.d), rho_stretched, rho)
            z_pml = ufl.conditional(ufl.ge(abs(r), self.PML.radius-self.PML.d), z_stretched, z)
        else:
            rho_stretched = pml_stretch(rho, rho, self.k0, x_dom=self.PML.rho-self.PML.d, x_pml=self.PML.rho)
            zt_stretched = pml_stretch(z, z, self.k0, x_dom=self.PML.zt-self.PML.d, x_pml=self.PML.zt)
            zb_stretched = pml_stretch(z, z, self.k0, x_dom=self.PML.zb+self.PML.d, x_pml=self.PML.zb)
            rho_pml = ufl.conditional(ufl.ge(rho, self.PML.rho-self.PML.d), rho_stretched, rho)
            z_pml = ufl.conditional(ufl.ge(z, self.PML.zt-self.PML.d), zt_stretched, ufl.conditional(ufl.le(z, self.PML.zb+self.PML.d), zb_stretched, z))
        pml_coords = ufl.as_vector((rho_pml, z_pml))
        self.epsr_pml, self.murinv_pml = pml_epsr_murinv(pml_coords, rho)
        
    def ComputeSolutions(self):
        # Set up the excitation - on antenna faces
        def Eport(x):
            """
            Compute the normalized electric field distribution in all ports.
            :param x: some given position you want to find the field on
            """
            Ep = np.zeros((3, x.shape[1]), dtype=complex)
            for p in range(self.refMeshdata.N_antennas):
                center = self.refMeshdata.pos_antennas[p]
                phi = -self.refMeshdata.rot_antennas[p] # Note rotation by the negative of antenna rotation
                Rmat = np.array([[np.cos(phi), -np.sin(phi), 0],
                                 [np.sin(phi), np.cos(phi), 0],
                                 [0, 0, 1]]) ## rotation around z
                y = np.transpose(x.T - center)
                loc_x = np.dot(Rmat, y) ### position vector, [x, y, z] presumably, rotated to be in the coordinates the antenna was defined in
                if (self.antenna_pol == 'vert'): ## vertical (z-) pol, field varies along x
                    Ep_loc = np.vstack((0*loc_x[0], 0*loc_x[0], np.cos(self.refMeshdata.kc*loc_x[0])))/np.sqrt(self.refMeshdata.antenna_width/2)
                else: ## horizontal (x-) pol, field varies along z
                    Ep_loc = np.vstack((np.cos(self.refMeshdata.kc*loc_x[2])), 0*loc_x[2], 0*loc_x[2])/np.sqrt(self.refMeshdata.antenna_height/2)
                    
                #simple, inexact confinement conditions
                #Ep_loc[:,np.sqrt(loc_x[0]**2 + loc_x[1]**2) > antenna_width] = 0 ## no field outside of the antenna's width (circular)
                ##if I confine it to just the 'empty face' of the waveguide thing. After testing, this seems to make no difference to just selecting the entire antenna via a sphere, with the above line
                Ep_loc[:, np.abs(loc_x[0])  > self.refMeshdata.antenna_width*.54] = 0 ## no field outside of the antenna's width
                Ep_loc[:, np.abs(loc_x[1])  > self.refMeshdata.antenna_depth*.04] = 0 ## no field outside of the antenna's depth - origin should be on this face - it is a face so no depth
                #for both
                Ep_loc[:,np.abs(loc_x[2]) > self.refMeshdata.antenna_height*.54] = 0 ## no field outside of the antenna's height.. plus a small extra (no idea if that matters)
                
                Ep_global = np.dot(Rmat, Ep_loc)
                Ep = Ep + Ep_global
            return Ep
    
        Ep = dolfinx.fem.Function(self.Vspace)
        Ep.interpolate(lambda x: Eport(x))
        
        
        # Set up simulation
        E = ufl.TrialFunction(self.Vspace)
        v = ufl.TestFunction(self.Vspace)
        curl_E = ufl.curl(E)
        curl_v = ufl.curl(v)
        nvec = ufl.FacetNormal(self.refMeshdata.mesh)
        Zrel = dolfinx.fem.Constant(self.refMeshdata.mesh, 1j)
        k00 = dolfinx.fem.Constant(self.refMeshdata.mesh, 1j)
        a = [dolfinx.fem.Constant(self.refMeshdata.mesh, 1.0 + 0j) for n in range(self.refMeshdata.N_antennas)]
        F_antennas_str = ''
        for n in range(self.refMeshdata.N_antennas):
            F_antennas_str += f"""+ 1j*k00/Zrel*ufl.inner(ufl.cross(E, nvec), ufl.cross(v, nvec))*ds_antennas[{n}] - 1j*k00/Zrel*2*a[{n}]*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(v, nvec))*ds_antennas[{n}]"""
        F = ufl.inner(1/self.mur*curl_E, curl_v)*self.dx_dom \
            - ufl.inner(k00**2*self.epsr*E, v)*self.dx_dom \
            + ufl.inner(self.murinv_pml*curl_E, curl_v)*self.dx_pml \
            - ufl.inner(k00**2*self.epsr_pml*E, v)*self.dx_pml + eval(F_antennas_str)
        bcs = [self.bc_pec]
        lhs, rhs = ufl.lhs(F), ufl.rhs(F)
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"} ## try looking this up to see if some other options might be better (apparently it is hard to find iterative solvers that converge)
        problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs, petsc_options=petsc_options)
        
        def ComputeFields(self):
            '''
            Computes the fields
            '''
            S = np.zeros((self.Nf, self.refMeshdata.N_antennas, self.refMeshdata.N_antennas), dtype=complex)
            solutions = []
            for nf in range(self.Nf):
                print(f'Rank {self.comm.rank}: Frequency {nf+1} / {self.Nf}')
                sys.stdout.flush()
                k00.value = 2*np.pi*self.fvec[nf]/c0
                Zrel.value = k00.value/np.sqrt(k00.value**2 - self.refMeshdata.kc**2)
                sols = []
                for n in range(self.refMeshdata.N_antennas):
                    for m in range(self.refMeshdata.N_antennas):
                        a[m].value = 0.0
                    a[n].value = 1.0
                    E_h = problem.solve()
                    for m in range(self.refMeshdata.N_antennas):
                        factor = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(2*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(Ep, nvec))*self.ds_antennas[m]))
                        factors = self.comm.gather(factor, root=self.model_rank)
                        if self.comm.rank == self.model_rank:
                            factor = sum(factors)
                        else:
                            factor = None
                        factor = self.comm.bcast(factor, root=self.model_rank)
                        b = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.cross(E_h, nvec), ufl.cross(Ep, nvec))*self.ds_antennas[m] + Zrel/(1j*self.k0)*ufl.inner(ufl.curl(E_h), ufl.cross(Ep, nvec))*self.ds_antennas[m]))/factor
                        bs = self.comm.gather(b, root=self.model_rank)
                        if self.comm.rank == self.model_rank:
                            b = sum(bs)
                        else:
                            b = None
                        b = self.comm.bcast(b, root=self.model_rank)
                        S[nf,m,n] = b
                    sols.append(E_h.copy())
                solutions.append(sols)
            return S, solutions
        
        if(self.verbosity > 0):
            print(f'Rank {self.comm.rank}: Computing REF solutions')
            sys.stdout.flush()
        elif(self.comm.rank == self.model_rank):
            print(f'Computing REF solutions')
        self.epsr.x.array[:] = self.epsr_array_ref
        S_ref, solutions_ref = ComputeFields()
        if(not self.computeReference):
            if(self.verbosity > 0):
                print(f'Rank {self.comm.rank}: Computing DUT solutions')
                sys.stdout.flush()
            elif(self.comm.rank == self.model_rank):
                print(f'Computing DUT solutions')
            
            sys.stdout.flush()
            self.epsr.x.array[:] = self.epsr_array_dut
            S_dut, solutions_dut = ComputeFields()
        
        print(f'Rank {self.comm.rank}: Computing optimization vectors')
        sys.stdout.flush()
        
        
        
        
        
        