# encoding: utf-8
## this file computes the simulation

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
from matplotlib.collections import _MeshData
eta0 = np.sqrt(mu0/eps0)

#===============================================================================
# ##line profiling
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)
#===============================================================================

#===============================================================================
# ##memory profiling
# from memory_profiler import profile
#===============================================================================

class Scatt3DProblem():
    """Class to hold definitions and functions for simulating scattering or transmission of electromagnetic waves for a rotationally symmetric structure."""
    def __init__(self,
                 comm, # MPI communicator
                 refMeshdata, # Mesh and metadata for the reference case
                 DUTMeshdata = None, # Mesh and metadata for the DUT case - should just include defects into the object (this will change the mesh)
                 verbosity = 0,   ## if > 0, I print more stuff
                 f0=10e9,             # Frequency of the problem
                 epsr_bkg=1,          # Permittivity of the background medium
                 mur_bkg=1,           # Permeability of the background medium
                 material_epsr=3.0*(1 - 0.01j),  # Permittivity of object
                 material_mur=1+0j,   # Permeability of object
                 defect_epsr=5.5*(1 - 0.01j),      # Permittivity of defect
                 defect_mur=1+0j,       # Permeability of defect
                 fem_degree=1,            # Degree of finite elements
                 model_rank=0,        # Rank of the master model - for saving, plotting, etc.
                 MPInum = 1,          # Number of MPI processes
                 freqs = [],          # Optionally, pass in the exact frequencies to calculate  
                 Nf = 1,              # Number of frequency-points to calculate
                 BW = 4e9,              # Bandwidth (equally spaced around f0)
                 dataFolder = 'data3D/', # Folder where data is stored
                 name = 'testRun', # Name of this particular simulation
                 pol = 'vert', ## Polarization of the antenna excitation - can be either 'vert' or 'horiz' (untested)
                 computeImmediately = True, ## compute solutions at the end of initialization
                 computeRef = True, # If computing immediately, computes the reference simulation, where defects are not included
                 ErefEdut = False, # compute optimization vectors with Eref*Edut, a less-approximated version of the equation. Should provide better results, but can only be used in simulation
                 excitation = 'antennas', # if 'planewave', sends in a planewave from the +x-axis, otherwise antenna excitation as normal
                 PW_dir = np.array([-1, 0, 0]), ## incident direction of the plane-wave, if used above. Default is coming in from the x-axis
                 PW_pol = np.array([0, 0, 1]), ## incident polarization of the plane-wave, if used above. Default is along the z-axis
                 makeOptVects = True, ## if True, compute and saves the optimization vectors. Turn False if not needed
                 computeBoth = False, ## if True and computeImmediately is True, computes both ref and dut cases.
                 ):
        """Initialize the problem."""
        
        self.dataFolder = dataFolder
        self.name = name
        self.MPInum = MPInum                      # Number of MPI processes (used for estimating computational costs)
        self.comm = comm
        self.model_rank = model_rank
        self.verbosity = verbosity
        
        self.makeOptVects = makeOptVects
        self.ErefEdut = ErefEdut
        
        self.tdim = 3                             # Dimension of triangles/tetraedra. 3 for 3D
        self.fdim = self.tdim - 1                      # Dimension of facets
        
        self.lambda0 = c0/f0                      # Vacuum wavelength, used to define lengths in the mesh
        self.k0 = 2*np.pi*f0/c0                   # Vacuum wavenumber
        
        if(len(freqs) > 0): ## if given frequency points, use those
            self.Nf = len(freqs)
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
        self.excitation = excitation
        
        self.PW_dir = PW_dir
        self.PW_pol = PW_pol

        # Set up mesh information
        self.refMeshdata = refMeshdata
        self.refMeshdata.mesh.topology.create_connectivity(self.tdim, self.tdim)
        self.refMeshdata.mesh.topology.create_connectivity(self.fdim, self.tdim) ## required when there are no antennas, for some reason
        if(DUTMeshdata != None):
            self.DUTMeshdata = DUTMeshdata
            self.DUTMeshdata.mesh.topology.create_connectivity(self.tdim, self.tdim)
            self.DUTMeshdata.mesh.topology.create_connectivity(self.fdim, self.tdim) ## required when there are no antennas, for some reason
            
        # Calculate solutions
        if(computeImmediately):
            if(computeBoth): ## compute both cases, then opt vectors if asked for
                self.compute(True, makeOptVects=False)
                self.compute(False, makeOptVects=self.makeOptVects)
            else: ## just compute the ref case, and make opt vects if asked for
                self.compute(computeRef, makeOptVects=self.makeOptVects)
            
                
    
    #@profile
    def compute(self, computeRef=True, makeOptVects=True):
        '''
        Sets up and runs the simulation. All the setup is set to reflect the current mesh, reference or dut. Solutions are saved.
        :param computeRef: If True, computes on the reference mesh
        '''
        t1 = timer()
        if(computeRef):
            meshData = self.refMeshdata
        else:
            meshData = self.DUTMeshdata
        # Initialize function spaces, boundary conditions, and PML - for the reference mesh
        self.InitializeFEM(meshData)
        self.InitializeMaterial(meshData)
        self.CalculatePML(meshData, self.k0) ## this is recalculated for each frequency, in ComputeSolutions - run it here just to initialize variables (not sure if needed)
        mem_usage = memory_usage((self.ComputeSolutions, (meshData,), {'computeRef':computeRef,}), max_usage = True)/1000 ## track the memory usage here
        #self.ComputeSolutions(meshData, computeRef=True)
        self.calcTimes = timer()-t1 ## Time it took to solve the problem. Given to mem-time estimator 
        if(makeOptVects):
            self.makeOptVectors(meshData)
        
        if(self.verbosity > 2):
            print(f'Max. memory: {mem_usage:.3f} GiB -- '+f"{self.comm.rank=} {self.comm.size=}")
        mems = self.comm.gather(mem_usage, root=self.model_rank)
        if (self.comm.rank == self.model_rank):
            self.memCost = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
            if(self.verbosity>0):
                print(f'Total memory: {self.memCost:.3f} GiB ({mem_usage*self.MPInum:.3f} GiB for this process, MPInum={self.MPInum} times)')
                print(f'Computations for {self.name} completed in {self.calcTimes:.2e} s ({self.calcTimes/3600:.2e} hours)')
        sys.stdout.flush()
                
                
    def InitializeFEM(self, meshData):
        # Set up some FEM function spaces and boundary condition stuff.
        curl_element = basix.ufl.element('N1curl', meshData.mesh.basix_cell(), self.fem_degree)
        self.Vspace = dolfinx.fem.functionspace(meshData.mesh, curl_element)
        self.ScalarSpace = dolfinx.fem.functionspace(meshData.mesh, ('CG', self.fem_degree))
        self.Wspace = dolfinx.fem.functionspace(meshData.mesh, ("DG", 0))
        # Create measures for subdomains and surfaces
        self.dx = ufl.Measure('dx', domain=meshData.mesh, subdomain_data=meshData.subdomains, metadata={'quadrature_degree': 5})
        self.dx_dom = self.dx((meshData.domain_marker, meshData.mat_marker, meshData.defect_marker))
        self.dx_pml = self.dx(meshData.pml_marker)
        self.ds = ufl.Measure('ds', domain=meshData.mesh, subdomain_data=meshData.boundaries)
        self.dS = ufl.Measure('dS', domain=meshData.mesh, subdomain_data=meshData.boundaries) ## capital S for internal facets (shared between two cells?)
        self.ds_antennas = [self.ds(m) for m in meshData.antenna_surface_markers]
        self.ds_pec = self.ds(meshData.pec_surface_marker)
        self.Ezero = dolfinx.fem.Function(self.Vspace)
        self.Ezero.x.array[:] = 0.0
        self.pec_dofs = dolfinx.fem.locate_dofs_topological(self.Vspace, entity_dim=self.fdim, entities=meshData.boundaries.find(meshData.pec_surface_marker))
        self.bc_pec = dolfinx.fem.dirichletbc(self.Ezero, self.pec_dofs)
        if(meshData.FF_surface): ## if there is a farfield surface
            self.dS_farfield = self.dS(meshData.farfield_surface_marker)
            cells = []
            ff_facets = meshData.boundaries.find(meshData.farfield_surface_marker)
            facets_to_cells = meshData.mesh.topology.connectivity(self.fdim, self.tdim)
            for facet in ff_facets:
                for cell in facets_to_cells.links(facet):
                    if cell not in cells:
                        cells.append(cell)
            cells.sort()
            self.farfield_cells = np.array(cells)
        
    def InitializeMaterial(self, meshData):
        # Set up material parameters. Not chancing mur for now, need to edit this if doing so
        self.epsr = dolfinx.fem.Function(self.Wspace)
        self.mur = dolfinx.fem.Function(self.Wspace)
        self.epsr.x.array[:] = self.epsr_bkg
        self.mur.x.array[:] = self.mur_bkg
        mat_cells = meshData.subdomains.find(meshData.mat_marker)
        self.mat_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=mat_cells)
        defect_cells = meshData.subdomains.find(meshData.defect_marker)
        self.defect_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=defect_cells)
        pml_cells = meshData.subdomains.find(meshData.pml_marker)
        self.pml_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=pml_cells)
        domain_cells = meshData.subdomains.find(meshData.domain_marker)
        self.domain_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=domain_cells)
        self.epsr.x.array[self.mat_dofs] = self.material_epsr
        self.mur.x.array[self.mat_dofs] = self.material_mur
        self.epsr.x.array[self.defect_dofs] = self.material_epsr
        self.mur.x.array[self.defect_dofs] = self.material_mur
        self.epsr_array_ref = self.epsr.x.array.copy()
        self.epsr.x.array[self.defect_dofs] = self.defect_epsr
        self.mur.x.array[self.defect_dofs] = self.defect_mur
        self.epsr_array_dut = self.epsr.x.array.copy()
        
    def CalculatePML(self, meshData, k):
        '''
        Set up the PML - stretched coordinates to form a perfectly-matched layer which absorbs incoming (perpendicular) waves
         Since we calculate at many frequencies, recalculate this for each freq. (Could also precalc it for each freq...)
        :param k: Frequency used for coordinate stretching.
        '''
        # Set up the PML
        def pml_stretch(y, x, k, x_dom=0, x_pml=1, n=3, R0=np.exp(-10)):
            '''
            Calculates the PML stretching of a coordinate
            :param y: the coordinate to be stretched
            :param x: the coordinate the stretching is based on
            :param k: wavenumber
            :param x_dom: size of domain
            :param x_pml: size of pml
            :param n: order
            :param R0: intended damping (based on relative strength of reflection?) According to 'THE_ELECTRICAL_ENGINEERING_HANDBOOKS', section 9.7: Through extensive numerical experimentation, Gedney
            (1996) and He (1997) found that, for a broad range of applications, an optimal choice for a 10-cell-thick, polynomial-graded PML is R(0) = e^-16. For a 5-cell-thick PML, R(0) = e^-8 is optimal.
            '''
            return y*(1 - 1j*(n + 1)*np.log(1/R0)/(2*k*np.abs(x_pml - x_dom))*((x - x_dom)/(x_pml - x_dom))**n)

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
        
        x, y, z = ufl.SpatialCoordinate(meshData.mesh)
        if(meshData.domain_geom == 'domedCyl'): ## implement it for this geometry
            r = ufl.real(ufl.sqrt(x**2 + y**2)) ## cylindrical radius. need to set this to real because I compare against it later
            domain_height_spheroid = ufl.conditional(ufl.ge(r, meshData.domain_radius), meshData.domain_height/2, (meshData.domain_height/2+meshData.dome_height)*ufl.real(ufl.sqrt(1-(r/(meshData.domain_radius+meshData.domain_spheroid_extraRadius))**2)))   ##start the z-stretching at, at minmum, the height of the domain cylinder
            PML_height_spheroid = (meshData.PML_height/2+meshData.dome_height)*ufl.sqrt(1-(r/(meshData.PML_radius+meshData.PML_spheroid_extraRadius))**2) ##should always be ge the pml cylinder's height
            x_stretched = pml_stretch(x, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            y_stretched = pml_stretch(y, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            z_stretched = pml_stretch(z, abs(z), k, x_dom=domain_height_spheroid, x_pml=PML_height_spheroid) ## /2 since the height is from - to +
            x_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), x_stretched, x) ## stretch when outside radius of the domain
            y_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), y_stretched, y) ## stretch when outside radius of the domain
            z_pml = ufl.conditional(ufl.ge(abs(z), domain_height_spheroid), z_stretched, z) ## stretch when outside the height of the cylinder of the domain (or oblate spheroid roof with factor a_dom/a_pml - should only be higher/lower inside the domain radially)
        elif(meshData.domain_geom == 'sphere'):
            r = ufl.real(ufl.sqrt(x**2 + y**2 + z**2))
            x_stretched = pml_stretch(x, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            y_stretched = pml_stretch(y, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            z_stretched = pml_stretch(z, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            x_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), x_stretched, x) ## stretch when outside radius of the domain
            y_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), y_stretched, y) ## stretch when outside radius of the domain
            z_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), z_stretched, z) ## stretch when outside radius of the domain
        else:
            print('nonvalid meshData.domain_geom')
        pml_coords = ufl.as_vector((x_pml, y_pml, z_pml))
        self.epsr_pml, self.murinv_pml = pml_epsr_murinv(pml_coords)

    #@profile
    def ComputeSolutions(self, meshData, computeRef = True):
        '''
        Computes the solutions
        
        :param computeRef: if True, computes the reference case (this is always needed for reconstruction). if False, computes the DUT case (needed for simulation-only stuff).
        Since things need to be initialized for each mesh, that should be done first.
        '''
        def Eport(x): # Set up the excitation - on antenna faces
            """
            Compute the normalized electric field distribution in all ports.
            :param x: some given position you want to find the field on
            """
            Ep = np.zeros((3, x.shape[1]), dtype=complex)
            for p in range(meshData.N_antennas):
                center = meshData.pos_antennas[p]
                phi = -meshData.rot_antennas[p] # Note rotation by the negative of antenna rotation
                Rmat = np.array([[np.cos(phi), -np.sin(phi), 0],
                                 [np.sin(phi), np.cos(phi), 0],
                                 [0, 0, 1]]) ## rotation around z
                y = np.transpose(x.T - center)
                loc_x = np.dot(Rmat, y) ### position vector, [x, y, z] presumably, rotated to be in the coordinates the antenna was defined in
                if (self.antenna_pol == 'vert'): ## vertical (z-) pol, field varies along x
                    Ep_loc = np.vstack((0*loc_x[0], 0*loc_x[0], np.cos(meshData.kc*loc_x[0])))/np.sqrt(meshData.antenna_width/2)
                else: ## horizontal (x-) pol, field varies along z
                    Ep_loc = np.vstack((np.cos(meshData.kc*loc_x[2])), 0*loc_x[2], 0*loc_x[2])/np.sqrt(meshData.antenna_height/2)
                    
                #simple, inexact confinement conditions
                #Ep_loc[:,np.sqrt(loc_x[0]**2 + loc_x[1]**2) > antenna_width] = 0 ## no field outside of the antenna's width (circular)
                ##if I confine it to just the 'empty face' of the waveguide thing. After testing, this seems to make no difference to just selecting the entire antenna via a sphere, with the above line
                Ep_loc[:, np.abs(loc_x[0])  > meshData.antenna_width*.54] = 0 ## no field outside of the antenna's width
                Ep_loc[:, np.abs(loc_x[1])  > meshData.antenna_depth*.04] = 0 ## no field outside of the antenna's depth - origin should be on this face - it is a face so no depth
                #for both
                Ep_loc[:,np.abs(loc_x[2]) > meshData.antenna_height*.54] = 0 ## no field outside of the antenna's height.. plus a small extra (no idea if that matters)
                
                Ep_global = np.dot(Rmat, Ep_loc)
                Ep = Ep + Ep_global
            return Ep
    
        Ep = dolfinx.fem.Function(self.Vspace)
        Ep.interpolate(lambda x: Eport(x))
        Eb = dolfinx.fem.Function(self.Vspace) ## background/plane wave excitation
        
        # Set up simulation
        E = ufl.TrialFunction(self.Vspace)
        v = ufl.TestFunction(self.Vspace)
        curl_E = ufl.curl(E)
        curl_v = ufl.curl(v)
        nvec = ufl.FacetNormal(meshData.mesh)
        Zrel = dolfinx.fem.Constant(meshData.mesh, 1j)
        k00 = dolfinx.fem.Constant(meshData.mesh, 1j)
        a = [dolfinx.fem.Constant(meshData.mesh, 1.0 + 0j) for n in range(meshData.N_antennas)]
        F_antennas_str = '0' ## seems to give an error when evaluating an empty string
        for n in range(meshData.N_antennas):
            F_antennas_str += f"""+ 1j*k00/Zrel*ufl.inner(ufl.cross(E, nvec), ufl.cross(v, nvec))*self.ds_antennas[{n}] - 1j*k00/Zrel*2*a[{n}]*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(v, nvec))*self.ds_antennas[{n}]"""
        F = ufl.inner(1/self.mur*curl_E, curl_v)*self.dx_dom \
            - ufl.inner(k00**2*self.epsr*E, v)*self.dx_dom \
            + ufl.inner(self.murinv_pml*curl_E, curl_v)*self.dx_pml \
            - ufl.inner(k00**2*self.epsr_pml*E, v)*self.dx_pml \
            - ufl.inner(k00**2*(self.epsr - 1/self.mur*self.mur_bkg*self.epsr_bkg)*Eb, v)*self.dx_dom + eval(F_antennas_str) ## background field and antenna terms
        bcs = [self.bc_pec]
        lhs, rhs = ufl.lhs(F), ufl.rhs(F)
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"} ## try looking this up to see if some other options might be better (apparently it is hard to find iterative solvers that converge)
        problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs, petsc_options=petsc_options)
        
        def ComputeFields():
            '''
            Computes the fields. There are two cases: one with antennas, and one without (PW excitation)
            Returns solutions, a list of Es for each frequency and exciting antenna, and S (0 if no antennas), a list of S-parameters for each frequency, exciting antenna, and receiving antenna
            '''
            S = np.zeros((self.Nf, meshData.N_antennas, meshData.N_antennas), dtype=complex)
            solutions = []
            for nf in range(self.Nf):
                if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
                    print(f'Rank {self.comm.rank}: Frequency {nf+1} / {self.Nf}')
                    sys.stdout.flush()
                k0 = 2*np.pi*self.fvec[nf]/c0
                k00.value = k0
                Zrel.value = k00.value/np.sqrt(k00.value**2 - meshData.kc**2)
                self.CalculatePML(meshData, k0)  ## update PML to this freq.
                
                def planeWave(x):
                    '''
                    Set up the excitation for a background plane-wave. Uses the problem's PW parameters. Needs the frequency, so I do it inside the freq. loop
                    :param x: some given position you want to find the field on
                    '''
                    E_pw = np.zeros((3, x.shape[1]), dtype=complex)
                    if(self.excitation == 'planewave'): # only make an excitation if we actually want one
                        E_pw[0, :] = self.PW_pol[0] ## just use the same amplitude as the polarization has
                        E_pw[1, :] = self.PW_pol[1] ## just use the same amplitude as the polarization has
                        E_pw[2, :] = self.PW_pol[2] ## just use the same amplitude as the polarization has
                        k_pw = k0*self.PW_dir ## direction (should be given normalized)
                        E_pw[:] = E_pw[:]*np.exp(-1j*np.dot(k_pw, x))
                    return E_pw
                
                Eb.interpolate(planeWave)
                sols = []
                if(meshData.N_antennas == 0): ## if no antennas:
                    E_h = problem.solve()
                    sols.append(E_h.copy())
                else:
                    for n in range(meshData.N_antennas):
                        for m in range(meshData.N_antennas):
                            a[m].value = 0.0
                        a[n].value = 1.0
                        E_h = problem.solve()
                        for m in range(meshData.N_antennas):
                            factor = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(2*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(Ep, nvec))*self.ds_antennas[m]))
                            factors = self.comm.gather(factor, root=self.model_rank)
                            if self.comm.rank == self.model_rank:
                                factor = sum(factors)
                            else:
                                factor = None
                            factor = self.comm.bcast(factor, root=self.model_rank)
                            b = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.cross(E_h, nvec), ufl.cross(Ep, nvec))*self.ds_antennas[m] + Zrel/(1j*k0)*ufl.inner(ufl.curl(E_h), ufl.cross(Ep, nvec))*self.ds_antennas[m]))/factor
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
        
        if(computeRef):
            if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
                print(f'Rank {self.comm.rank}: Computing REF solutions')
            sys.stdout.flush()
                
            self.epsr.x.array[:] = self.epsr_array_ref
            self.S_ref, self.solutions_ref = ComputeFields()    
        else:
            if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
                print(f'Rank {self.comm.rank}: Computing DUT solutions')
            sys.stdout.flush()
            
            self.epsr.x.array[:] = self.epsr_array_dut
            self.S_dut, self.solutions_dut = ComputeFields()
            
    #@profile
    def makeOptVectors(self, meshData):
        '''
        Computes the optimization vectors from the E-fields and saves to .xdmf - this is done on the reference mesh.
        This function also saves various other parameters for later postprocessing
        '''
        
        if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
            print(f'Rank {self.comm.rank}: Computing optimization vectors')
            sys.stdout.flush()
        
        
        # Create function space for temporary interpolation
        q = dolfinx.fem.Function(self.Wspace)
        bb_tree = dolfinx.geometry.bb_tree(meshData.mesh, meshData.mesh.topology.dim)
        cell_volumes = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.conj(ufl.TestFunction(self.Wspace))*ufl.dx)).array
        def q_func(x, Em, En, k0):
            '''
            Calculates the 'optimization vector' at each position in the reference meshData. Since the DUT mesh is different,
            this requires interpolation to find the E-fields at each point
            :param x: positions/points
            :param Em: first E-field
            :param En: second E-field
            :param k0: wavenumber at this frequency
            '''
            cells = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(meshData.mesh, cell_candidates, x.T)
            for i, point in enumerate(x.T):
                if len(colliding_cells.links(i)) > 0:
                    cells.append(colliding_cells.links(i)[0])
            Em_vals = Em.eval(x.T, cells)
            En_vals = En.eval(x.T, cells)
            values = -1j*k0/eta0/2*(Em_vals[:,0]*En_vals[:,0] + Em_vals[:,1]*En_vals[:,1] + Em_vals[:,2]*En_vals[:,2])*cell_volumes
            return values
        
        xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'output-qs.xdmf', file_mode='w')
        xdmf.write_mesh(meshData.mesh)
        self.epsr.x.array[:] = cell_volumes
        xdmf.write_function(self.epsr, -3)
        self.epsr.x.array[:] = self.epsr_array_ref
        xdmf.write_function(self.epsr, -2)
        self.epsr.x.array[:] = self.epsr_array_dut
        xdmf.write_function(self.epsr, -1)
        b = np.zeros(self.Nf*meshData.N_antennas*meshData.N_antennas, dtype=complex)
        for nf in range(self.Nf):
            if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 2) and (meshData.N_antennas > 0) ):
                print(f'Rank {self.comm.rank}: Frequency {nf+1} / {self.Nf}')
                sys.stdout.flush()
            k0 = 2*np.pi*self.fvec[nf]/c0
            for m in range(meshData.N_antennas):
                Em_ref = self.solutions_ref[nf][m]
                for n in range(meshData.N_antennas):
                    if(self.ErefEdut): ## only using Eref*Eref right now. This should provide a superior reconstruction with fully simulated data, though
                        En = self.solutions_dut[nf][n] 
                    else:
                        En = self.solutions_ref[nf][n]
                    q.interpolate(functools.partial(q_func, Em=Em_ref, En=En, k0=k0))
                    # The function q is one row in the A-matrix, save it to file
                    xdmf.write_function(q, nf*meshData.N_antennas*meshData.N_antennas + m*meshData.N_antennas + n)
            if(meshData.N_antennas < 1): # if no antennas, still save
                q.interpolate(functools.partial(q_func, Em=self.solutions_ref[nf][0], En=self.solutions_ref[nf][0], k0=k0))
                xdmf.write_function(q, nf)
        xdmf.close()
        
        if (self.comm.rank == self.model_rank): # Save global values for further postprocessing
            if( hasattr(self, 'solutions_dut') and hasattr(self, 'solutions_ref')): ## need both computed - otherwise, do not save
                b = np.zeros(self.Nf*meshData.N_antennas*meshData.N_antennas, dtype=complex) ## the array of S-parameters
                for nf in range(self.Nf):
                    for m in range(meshData.N_antennas):
                        for n in range(meshData.N_antennas):
                            b[nf*meshData.N_antennas*meshData.N_antennas + m*meshData.N_antennas + n] = self.S_dut[nf, m, n] - self.S_ref[nf, n, m]
                np.savez(self.dataFolder+self.name+'output.npz', b=b, fvec=self.fvec, S_ref=self.S_ref, S_dut=self.S_dut, epsr_mat=self.material_epsr, epsr_defect=self.defect_epsr, N_antennas=meshData.N_antennas)     
    
    def saveEFieldsForAnim(self, Nframes = 50, removePML = True):
        '''
        Saves the E-field magnitudes for the final solution into .xdmf, for a number of different phase factors to create an animation in paraview
        Uses the reference mesh and fields. If removePML, set the values within the PML to 0 (can also be NaN, etc.)
        
        :param Nframes: Number of frames in the anim. Each frame is a different phase from 0 to 2*pi
        :param removePML: If True, sets all values in the PML to something different
        '''
        ## This is presumably an overdone method of finding these already-computed fields - I doubt this is needed
        meshData = self.refMeshdata # use the ref case
        
        E = dolfinx.fem.Function(self.ScalarSpace)
        bb_tree = dolfinx.geometry.bb_tree(meshData.mesh, meshData.mesh.topology.dim)
        def q_abs(x, Es, pol = 'z'): ## similar to the one in makeOptVectors
            cells = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(meshData.mesh, cell_candidates, x.T)
            for i, point in enumerate(x.T):
                if len(colliding_cells.links(i)) > 0:
                    cells.append(colliding_cells.links(i)[0])
            if(pol == 'z'): ## it is not simple to save the vector itself for some reason...
                E_vals = Es.eval(x.T, cells)[:, 2]
            elif(pol == 'x'):
                E_vals = Es.eval(x.T, cells)[:, 0]
            elif(pol == 'y'):
                E_vals = Es.eval(x.T, cells)[:, 1]
            return E_vals
        pols = ['x', 'y', 'z']
        sol = self.solutions_ref[0][0] ## fields for the first frequency
        pml_cells = meshData.subdomains.find(meshData.pml_marker)
        pml_dofs = dolfinx.fem.locate_dofs_topological(self.ScalarSpace, entity_dim=self.tdim, entities=pml_cells)
        for pol in pols:
            xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'outputPhaseAnimationE'+pol+'.xdmf', file_mode='w')
            E.interpolate(functools.partial(q_abs, Es=sol, pol=pol))
            xdmf.write_mesh(meshData.mesh)
            if(removePML):
                E.x.array[pml_dofs] = 0#np.nan ## use the ScalarSpace one
            for i in range(Nframes):
                E.x.array[:] = E.x.array*np.exp(1j*2*pi/Nframes)
                xdmf.write_function(E, i)
        xdmf.close()
        if(self.verbosity>0 & self.comm.rank == self.model_rank):
            print(self.name+' E-fields anim')
        
    def saveDofsView(self, meshData):
        '''
        Saves the dofs with different numbers for viewing in ParaView. This hangs on the cluster, for unknown reasons
        :param meshData: Whichever meshData to use
        '''
        self.InitializeFEM(meshData) ## so that this can be done before calculations
        self.InitializeMaterial(meshData) ## so that this can be done before calculations
        vals = dolfinx.fem.Function(self.Wspace)
        xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'Dofsview.xdmf', file_mode='w')
        xdmf.write_mesh(meshData.mesh)
        pec_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.fdim, entities=meshData.boundaries.find(meshData.pec_surface_marker)) ## to use Wspace instead of Vspace... maybe this matters
        vals.x.array[:] = np.nan
        vals.x.array[self.domain_dofs] = 0
        vals.x.array[self.pml_dofs] = -1
        vals.x.array[self.defect_dofs] = 3
        vals.x.array[self.mat_dofs] = 2
        vals.x.array[pec_dofs] = 5
        vals.x.array[self.farfield_cells] = 1
        xdmf.write_function(vals, 0)
        xdmf.close()
        if(self.verbosity>0 & self.comm.rank == self.model_rank):
            print(self.name+' DoFs view saved')
            
    def calcFarField(self, reference, angles = np.array([[90, 180], [90, 0]]), compareToMie = False, showPlots=False):
        '''
        Calculates the farfield at each frequency point at at given angles, using the farfield boundary in the mesh - must have mesh.FF_surface = True
        Returns an array of [E_theta, E_phi] at each angle, to the master process only
        :param reference: Whether this is being computed for the DUT case or the reference
        :param angles: List (or array) of theta and phi angles to calculate at [in degrees]. Incoming plane waves should be from (90, 0)
        :param compareToMie: If True, plots a comparison against predicted Mie scattering (assuming spherical object)
        :param showPlots: If True, plt.show(). Plots are still saved, though. This must be False for cluster use
        '''
        t1 = timer()
        if( (self.verbosity > 0 and self.comm.rank == self.model_rank)):
                print(f'Calculating farfield values...')
                sys.stdout.flush()
        if(reference):
            meshData = self.refMeshdata
            sols = self.solutions_ref
        else:
            meshData = self.DUTMeshdata
            sols = self.solutions_dut
            
            
        numAngles = np.shape(angles)[0]
        prefactor = dolfinx.fem.Constant(meshData.mesh, 0j)
        n = ufl.FacetNormal(meshData.mesh)('+')
        signfactor = ufl.sign(ufl.inner(n, ufl.SpatialCoordinate(meshData.mesh))) # Enforce outward pointing normal
        exp_kr = dolfinx.fem.Function(self.ScalarSpace)
        farfields = np.zeros((self.Nf, numAngles, 2), dtype=complex) ## for each frequency and angle, E_theta and E_phi
        for b in range(self.Nf):
            freq = self.fvec[b]
            k = 2*np.pi*freq/c0
            E = sols[b][0]('+')
            for i in range(numAngles):
                theta = angles[i,0]*pi/180 # convert to radians first
                phi = angles[i,1]*pi/180
                khat = ufl.as_vector([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) ## in cartesian coordinates 
                phiHat = ufl.as_vector([-np.sin(phi), np.cos(phi), 0])
                thetaHat = ufl.as_vector([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
                
                #eta0 = float(np.sqrt(self.mur_bkg/self.epsr_bkg)) # following Daniel's script, this should really be etar here
                eta0 = float(np.sqrt(mu0/eps0)) ## must convert to float first
                
                H = -1/(1j*k*eta0)*ufl.curl(E) ## or possibly B = 1/w k x E, 2*pi/freq*k*ufl.cross(khat, E)
                
                ## can only integrate scalars
                self.F_theta = signfactor*prefactor* ufl.inner(thetaHat, ufl.cross(khat, ( ufl.cross(E, n) + eta0*ufl.cross(khat, ufl.cross(n, H))) ))*exp_kr*self.dS_farfield
                self.F_phi = signfactor*prefactor* ufl.inner(phiHat, ufl.cross(khat, ( ufl.cross(E, n) + eta0*ufl.cross(khat, ufl.cross(n, H))) ))*exp_kr*self.dS_farfield
                
                #self.F_theta = 1*self.dS_farfield ## calculate area
                #self.F_phi = 1*self.dS_farfield ## calculate area
                
                khat = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)] ## so I can use it in evalFs as regular numbers
                def evalFs(): ## evaluates the farfield in some given direction khat
                    exp_kr.interpolate(lambda x: np.exp(1j*k*(khat[0]*x[0] + khat[1]*x[1] + khat[2]*x[2])), self.farfield_cells)
                    prefactor.value = 1j*k/(4*pi)
                    
                    F_theta = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(self.F_theta))
                    F_phi = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(self.F_phi))
                    
                    #return(F_thetaE, F_phiE, F_thetaH, F_phiH)
                    return np.array((F_theta, F_phi))
                
                farfieldpart = evalFs()
                farfieldparts = self.comm.gather(farfieldpart, root=self.model_rank)
                if(self.comm.rank == 0): ## assemble each part as it is made
                    farfields[b, i] = sum(farfieldparts)
        
        if(self.comm.rank == 0): ## plotting and returning
            if(self.verbosity > 1):
                print(f'Farfields calculated in {timer()-t1:.3f} s')
                                        
            if(compareToMie and self.Nf < 3): ## make some plots by angle if few freqs. (assuming here that we have many angles)
                for b in range(self.Nf):
                    fig = plt.figure()
                    ax1 = plt.subplot(1, 1, 1)
                    #print('theta',np.abs(farfields[b,:,0]))
                    #print('phi',np.abs(farfields[b,:,1]))
                    #print('intensity',np.abs(farfields[b,:,0])**2 + np.abs(farfields[b,:,1])**2)
                    #plt.plot(angles[:, 1], np.abs(farfields[b,:,0]), label = 'theta-pol')
                    #plt.plot(angles[:, 1], np.abs(farfields[b,:,1]), label = 'phi-pol')'
                    mag = np.abs(farfields[b,:,0])**2 + np.abs(farfields[b,:,1])**2
                    ax1.plot(angles[:, 1], mag, label = 'Integrated Intensity', linewidth = 2.5)
                    
                    #===========================================================
                    # ##Calculate Mie scattering
                    # import miepython ## this shouldn't need to be installed on the cluster (I can't figure out how to) so only import it here
                    # m = np.sqrt(self.material_epsr) ## complex index of refraction - if it is not PEC
                    # mie = np.zeros_like(angles[:, 1])
                    # for i in range(len(angles[:, 1])): ## get a miepython error if I use a vector of x, so:
                    #     lambdat = c0/freq
                    #     x = 2*pi*meshData.object_radius/lambdat
                    #     mie[i] = miepython.i_par(m, x, np.cos((angles[i, 1]*pi/180+pi)), norm='qsca')*pi*meshData.object_radius**2
                    # np.savetxt('mietest.out', mie)
                    #===========================================================
                    
                    mie = np.loadtxt('mietest.out') ## data for object_radius = 0.34, material_epsr=6
                    ax1.plot(angles[:, 1], mie, label = 'Miepython Intensity', linewidth = 2.5)
                    ax1.legend()
                    plt.savefig(self.dataFolder+self.name+'miecomp.png')
                    if(showPlots):
                        plt.show()
                    plt.clf()
        
            
            if(compareToMie and self.Nf > 2): ## do plots by frequency for forward+backward scattering
                ##Calculate Mie scattering
                m = np.sqrt(self.material_epsr) ## complex index of refraction - if it is not PEC
                mieForward = np.zeros_like(self.fvec)
                mieBackward = np.zeros_like(self.fvec)
                for i in range(len(self.fvec)): ## get a miepython error if I use a vector of x, so:
                    lambdat = c0/self.fvec[i]
                    x = 2*pi*meshData.object_radius/lambdat
                    mieForward[i] = miepython.i_par(m, x, np.cos(pi), norm='qsca') 
                    mieBackward[i] = miepython.i_par(m, x, np.cos(0), norm='qsca')
                
                for i in range(len(angles)):
                    plt.plot(self.fvec/1e9, np.abs(farfields[:, i, 0])**2 + np.abs(farfields[:, i, 1])**2, label = r'sim, $\theta=$'+f'{angles[i, 0]:.0f}, $\phi={angles[i, 1]:.0f}$')
                
                plt.xlabel('Frequency [GHz]')
                plt.ylabel('Intensity')
                plt.plot(self.fvec/1e9, mieForward, linestyle='--', label = r'Mie forward-scattering')
                plt.plot(self.fvec/1e9, mieBackward, linestyle='--', label = r'Mie backward-scattering')
                #plt.plot(self.fvec/1e9, 4*pi*meshData.FF_surface_radius**2*np.ones_like(self.fvec), label = r'theo, area of sphere') ## theoretical area of a sphere
                plt.legend()
                plt.grid()
                plt.tight_layout()
                if(showPlots):
                    plt.show()
            
            return farfields