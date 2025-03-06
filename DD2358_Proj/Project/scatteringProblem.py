# encoding: utf-8
## this file makes the mesh

import os ## presumably dont need this import here
os.environ["OMP_NUM_THREADS"] = "1" # seemingly needed for MPI speedup
from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import basix
import functools
from timeit import default_timer as timer
import psutil
from memory_profiler import memory_usage
import gmsh
import sys
import pyvista
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi
eta0 = np.sqrt(mu0/eps0)

##memory profiling
from memory_profiler import profile

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
                 defect_epsr=2.5*(1 - 0.01j),      # Permittivity of defect
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
                 ):
        """Initialize the problem."""
        
        self.dataFolder = dataFolder
        self.name = name
        self.MPInum = MPInum                      # Number of MPI processes (used for estimating computational costs)
        self.comm = comm
        self.model_rank = model_rank
        self.verbosity = verbosity
        
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

        # Set up mesh information
        self.refMeshdata = refMeshdata
        self.refMeshdata.mesh.topology.create_connectivity(self.tdim, self.tdim)
        if(DUTMeshdata != None):
            self.DUTMeshdata = DUTMeshdata
            self.DUTMeshdata.mesh.topology.create_connectivity(self.tdim, self.tdim)
        
        
        
        # Calculate solutions
        if(computeImmediately):
            self.compute(computeRef)
    
    def compute(self, computeRef=True):
        t1 = timer()
        if(computeRef):
            mesh = self.refMeshdata
        else:
            mesh = self.DUTMeshdata
        # Initialize function spaces, boundary conditions, and PML - for the reference mesh
        self.InitializeFEM(mesh)
        self.InitializeMaterial(mesh)
        self.CalculatePML(mesh, self.k0) ## this is recalculated for each frequency, in ComputeSolutions - run it here just to initialize variables (not sure if needed)
        mem_usage = memory_usage((self.ComputeSolutions, (mesh,), {'computeRef':True,}), max_usage = True)/1000 ## track the memory usage here
        #self.ComputeSolutions(mesh, computeRef=True)
        self.makeOptVectors(mesh)
        self.calcTimes = timer()-t1 ## Time it took to solve the problem. Given to mem-time estimator 
        
        if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 1) ):
            print(f'Max. memory: {mem_usage:.3f} GiB -- '+f"{self.comm.rank=} {self.comm.size=}")
        mems = self.comm.gather(mem_usage, root=self.model_rank)
        if (self.comm.rank == self.model_rank):
            self.memCost = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
            if(self.verbosity>0):
                print(f'Total memory: {self.memCost:.3f} GiB ({mem_usage*self.MPInum:.3f} GiB for this process, MPInum={self.MPInum} times)')
                print(f'Computations for {self.name} completed in {self.calcTimes:.2e} s ({self.calcTimes/3600:.2e} hours)')
        sys.stdout.flush()
                
                
    def InitializeFEM(self, mesh):
        # Set up some FEM function spaces and boundary condition stuff.
        curl_element = basix.ufl.element('N1curl', mesh.mesh.basix_cell(), self.fem_degree)
        self.Vspace = dolfinx.fem.functionspace(mesh.mesh, curl_element)
        
        # Create measures for subdomains and surfaces
        self.dx = ufl.Measure('dx', domain=mesh.mesh, subdomain_data=mesh.subdomains, metadata={'quadrature_degree': 5})
        self.dx_dom = self.dx((mesh.domain_marker, mesh.mat_marker, mesh.defect_marker))
        self.dx_pml = self.dx(mesh.pml_marker)
        self.ds = ufl.Measure('ds', domain=mesh.mesh, subdomain_data=mesh.boundaries)
        self.ds_antennas = [self.ds(m) for m in mesh.antenna_surface_markers]
        self.ds_pec = self.ds(mesh.pec_surface_marker)
        self.pec_dofs = dolfinx.fem.locate_dofs_topological(self.Vspace, entity_dim=self.fdim, entities=mesh.boundaries.find(mesh.pec_surface_marker))
        self.Ezero = dolfinx.fem.Function(self.Vspace)
        self.Ezero.x.array[:] = 0.0
        self.bc_pec = dolfinx.fem.dirichletbc(self.Ezero, self.pec_dofs)
        
    def InitializeMaterial(self, mesh):
        # Set up material parameters. Not chancing mur for now, need to edit this if doing so
        self.Wspace = dolfinx.fem.functionspace(mesh.mesh, ("DG", 0))
        self.epsr = dolfinx.fem.Function(self.Wspace)
        self.mur = dolfinx.fem.Function(self.Wspace)
        self.epsr.x.array[:] = self.epsr_bkg
        self.mur.x.array[:] = self.mur_bkg
        mat_cells = mesh.subdomains.find(mesh.mat_marker)
        mat_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=mat_cells)
        defect_cells = mesh.subdomains.find(mesh.defect_marker)
        defect_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=defect_cells)
        self.epsr.x.array[mat_dofs] = self.material_epsr
        self.mur.x.array[mat_dofs] = self.material_mur
        self.epsr.x.array[defect_dofs] = self.material_epsr
        self.mur.x.array[defect_dofs] = self.material_mur
        self.epsr_array_ref = self.epsr.x.array.copy()
        self.epsr.x.array[defect_dofs] = self.defect_epsr
        self.mur.x.array[defect_dofs] = self.defect_mur
        self.epsr_array_dut = self.epsr.x.array.copy()
        
    def CalculatePML(self, mesh, k):
        '''
        Set up the PML - stretched coordinates to form a perfectly-matched layer which absorbs incoming (perpendicular) waves
         Since we calculate at many frequencies, recalculate this for each freq. (Could also precalc it for each freq...)
        :param k: Frequency used for coordinate stretching.
        '''
        # Set up the PML
        def pml_stretch(y, x, k, x_dom=0, x_pml=1, n=3, R0=1e-10):
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
        
        if(mesh.domain_geom == 'domedCyl'): ## implement it for this geometry
            x, y, z = ufl.SpatialCoordinate(mesh.mesh)
            r = ufl.real(ufl.sqrt(x**2 + y**2)) ## cylindrical radius. need to set this to real because I compare against it later
            domain_height_spheroid = ufl.conditional(ufl.ge(r, mesh.domain_radius), mesh.domain_height/2, (mesh.domain_height/2+mesh.dome_height)*ufl.real(ufl.sqrt(1-(r/(mesh.domain_radius+mesh.domain_spheroid_extraRadius))**2)))   ##start the z-stretching at, at minmum, the height of the domain cylinder
            PML_height_spheroid = (mesh.PML_height/2+mesh.dome_height)*ufl.sqrt(1-(r/(mesh.PML_radius+mesh.PML_spheroid_extraRadius))**2) ##should always be ge the pml cylinder's height
            x_stretched = pml_stretch(x, r, k, x_dom=mesh.domain_radius, x_pml=mesh.PML_radius)
            y_stretched = pml_stretch(y, r, k, x_dom=mesh.domain_radius, x_pml=mesh.PML_radius)
            z_stretched = pml_stretch(z, abs(z), k, x_dom=domain_height_spheroid, x_pml=PML_height_spheroid) ## /2 since the height is from - to +
            x_pml = ufl.conditional(ufl.ge(abs(r), mesh.domain_radius), x_stretched, x) ## stretch when outside radius of the domain
            y_pml = ufl.conditional(ufl.ge(abs(r), mesh.domain_radius), y_stretched, y) ## stretch when outside radius of the domain
            z_pml = ufl.conditional(ufl.ge(abs(z), domain_height_spheroid), z_stretched, z) ## stretch when outside the height of the cylinder of the domain (or oblate spheroid roof with factor a_dom/a_pml - should only be higher/lower inside the domain radially)
        else:
            print('nonvalid mesh.domain_geom')
        pml_coords = ufl.as_vector((x_pml, y_pml, z_pml))
        self.epsr_pml, self.murinv_pml = pml_epsr_murinv(pml_coords)

    @profile
    def ComputeSolutions(self, mesh, computeRef = True):
        '''
        Computes the solutions
        
        :param computeRef: if True, computes the reference case (this is always needed for reconstruction). if False, computes the DUT case (needed for simulation-only stuff).
        Since things need to be initialized for each mesh, that should be done first.
        '''
        # Set up the excitation - on antenna faces
        def Eport(x):
            """
            Compute the normalized electric field distribution in all ports.
            :param x: some given position you want to find the field on
            """
            Ep = np.zeros((3, x.shape[1]), dtype=complex)
            for p in range(mesh.N_antennas):
                center = mesh.pos_antennas[p]
                phi = -mesh.rot_antennas[p] # Note rotation by the negative of antenna rotation
                Rmat = np.array([[np.cos(phi), -np.sin(phi), 0],
                                 [np.sin(phi), np.cos(phi), 0],
                                 [0, 0, 1]]) ## rotation around z
                y = np.transpose(x.T - center)
                loc_x = np.dot(Rmat, y) ### position vector, [x, y, z] presumably, rotated to be in the coordinates the antenna was defined in
                if (self.antenna_pol == 'vert'): ## vertical (z-) pol, field varies along x
                    Ep_loc = np.vstack((0*loc_x[0], 0*loc_x[0], np.cos(mesh.kc*loc_x[0])))/np.sqrt(mesh.antenna_width/2)
                else: ## horizontal (x-) pol, field varies along z
                    Ep_loc = np.vstack((np.cos(mesh.kc*loc_x[2])), 0*loc_x[2], 0*loc_x[2])/np.sqrt(mesh.antenna_height/2)
                    
                #simple, inexact confinement conditions
                #Ep_loc[:,np.sqrt(loc_x[0]**2 + loc_x[1]**2) > antenna_width] = 0 ## no field outside of the antenna's width (circular)
                ##if I confine it to just the 'empty face' of the waveguide thing. After testing, this seems to make no difference to just selecting the entire antenna via a sphere, with the above line
                Ep_loc[:, np.abs(loc_x[0])  > mesh.antenna_width*.54] = 0 ## no field outside of the antenna's width
                Ep_loc[:, np.abs(loc_x[1])  > mesh.antenna_depth*.04] = 0 ## no field outside of the antenna's depth - origin should be on this face - it is a face so no depth
                #for both
                Ep_loc[:,np.abs(loc_x[2]) > mesh.antenna_height*.54] = 0 ## no field outside of the antenna's height.. plus a small extra (no idea if that matters)
                
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
        nvec = ufl.FacetNormal(mesh.mesh)
        Zrel = dolfinx.fem.Constant(mesh.mesh, 1j)
        k00 = dolfinx.fem.Constant(mesh.mesh, 1j)
        a = [dolfinx.fem.Constant(mesh.mesh, 1.0 + 0j) for n in range(mesh.N_antennas)]
        F_antennas_str = ''
        for n in range(mesh.N_antennas):
            F_antennas_str += f"""+ 1j*k00/Zrel*ufl.inner(ufl.cross(E, nvec), ufl.cross(v, nvec))*self.ds_antennas[{n}] - 1j*k00/Zrel*2*a[{n}]*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(v, nvec))*self.ds_antennas[{n}]"""
        F = ufl.inner(1/self.mur*curl_E, curl_v)*self.dx_dom \
            - ufl.inner(k00**2*self.epsr*E, v)*self.dx_dom \
            + ufl.inner(self.murinv_pml*curl_E, curl_v)*self.dx_pml \
            - ufl.inner(k00**2*self.epsr_pml*E, v)*self.dx_pml + eval(F_antennas_str)
        bcs = [self.bc_pec]
        lhs, rhs = ufl.lhs(F), ufl.rhs(F)
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"} ## try looking this up to see if some other options might be better (apparently it is hard to find iterative solvers that converge)
        problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs, petsc_options=petsc_options)
        
        def ComputeFields():
            '''
            Computes the fields
            '''
            S = np.zeros((self.Nf, mesh.N_antennas, mesh.N_antennas), dtype=complex)
            solutions = []
            for nf in range(self.Nf):
                if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 1) ):
                    print(f'Rank {self.comm.rank}: Frequency {nf+1} / {self.Nf}')
                    sys.stdout.flush()
                k0 = 2*np.pi*self.fvec[nf]/c0
                k00.value = k0
                Zrel.value = k00.value/np.sqrt(k00.value**2 - mesh.kc**2)
                self.CalculatePML(mesh, k0)  ## update PML to this freq.
                sols = []
                for n in range(mesh.N_antennas):
                    for m in range(mesh.N_antennas):
                        a[m].value = 0.0
                    a[n].value = 1.0
                    E_h = problem.solve()
                    for m in range(mesh.N_antennas):
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
        
        if(computeRef):
            if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 1) ):
                print(f'Rank {self.comm.rank}: Computing REF solutions')
            sys.stdout.flush()
                
            self.epsr.x.array[:] = self.epsr_array_ref
            self.S_ref, self.solutions_ref = ComputeFields()    
        else:
            if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 1) ):
                print(f'Rank {self.comm.rank}: Computing DUT solutions')
            sys.stdout.flush()
            
            self.epsr.x.array[:] = self.epsr_array_dut
            self.S_dut, self.solutions_dut = ComputeFields()
            
    def makeOptVectors(self, mesh):
        '''
        Computes the optimization vectors from the E-fields and saves to .xdmf - this is done on the reference mesh
        '''
        
        if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 1) ):
            print(f'Rank {self.comm.rank}: Computing optimization vectors')
            sys.stdout.flush()
        
        
        # Create function space for temporary interpolation
        q = dolfinx.fem.Function(self.Wspace)
        bb_tree = dolfinx.geometry.bb_tree(mesh.mesh, mesh.mesh.topology.dim)
        cell_volumes = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.conj(ufl.TestFunction(self.Wspace))*ufl.dx)).array
        def q_func(x, Em, En, k0):
            '''
            Calculates the 'optimization vector' at each position in the reference mesh. Since the DUT mesh is different,
            this requires interpolation to find the E-fields at each point
            :param x: positions/points
            :param Em: first E-field
            :param En: second E-field
            :param k0: wavenumber at this frequency
            '''
            cells = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh.mesh, cell_candidates, x.T)
            for i, point in enumerate(x.T):
                if len(colliding_cells.links(i)) > 0:
                    cells.append(colliding_cells.links(i)[0])
            Em_vals = Em.eval(x.T, cells)
            En_vals = En.eval(x.T, cells)
            values = -1j*k0/eta0/2*(Em_vals[:,0]*En_vals[:,0] + Em_vals[:,1]*En_vals[:,1] + Em_vals[:,2]*En_vals[:,2])*cell_volumes
            return values
        
        xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'output-qs.xdmf', file_mode='w')
        xdmf.write_mesh(mesh.mesh)
        self.epsr.x.array[:] = cell_volumes
        xdmf.write_function(self.epsr, -3)
        self.epsr.x.array[:] = self.epsr_array_ref
        xdmf.write_function(self.epsr, -2)
        self.epsr.x.array[:] = self.epsr_array_dut
        xdmf.write_function(self.epsr, -1)
        for nf in range(self.Nf):
            if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 1) ):
                print(f'Rank {self.comm.rank}: Frequency {nf+1} / {self.Nf}')
                sys.stdout.flush()
            k0 = 2*np.pi*self.fvec[nf]/c0
            for m in range(mesh.N_antennas):
                Em_ref = self.solutions_ref[nf][m]
                for n in range(mesh.N_antennas):
                    if(self.ErefEdut): ## only using Eref*Eref right now. This should provide a superior reconstruction with fully simulated data, though
                        En = self.solutions_dut[nf][n] 
                    else:
                        En = self.solutions_ref[nf][n]
                    q.interpolate(functools.partial(q_func, Em=Em_ref, En=En, k0=k0))
                    # The function q is one row in the A-matrix, save it to file
                    xdmf.write_function(q, nf*mesh.N_antennas*mesh.N_antennas + m*mesh.N_antennas + n)
        xdmf.close()
    
    def saveEFieldsForAnim(self, Nframes = 50):
        '''
        Saves the E-field magnitudes for the final solution into .xdmf, for a number of different phase factors to create an animation in paraview
        Uses the reference mesh and fields.
        '''
        ## This is presumably an overdone method of finding these already-computed fields - I doubt this is needed
        E = dolfinx.fem.Function(self.Wspace)
        bb_tree = dolfinx.geometry.bb_tree(self.refMeshdata.mesh, self.refMeshdata.mesh.topology.dim)
        for pol in ['x', 'y', 'z']: ## save each pol in a different file
            def q_abs(x, E): ## similar to the one in makeOptVectors
                cells = []
                cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
                colliding_cells = dolfinx.geometry.compute_colliding_cells(self.refMeshdata.mesh, cell_candidates, x.T)
                for i, point in enumerate(x.T):
                    if len(colliding_cells.links(i)) > 0:
                        cells.append(colliding_cells.links(i)[0])
                E_vals = E.eval(x.T, cells)
                #values = np.sqrt((E_vals[:,0]*E_vals[:,0] + E_vals[:,1]*E_vals[:,1] + E_vals[:,2]*E_vals[:,2]))
                if(pol == 'z'): ## save z-pol
                    values = E_vals[:,2] ## this should be the z-component...
                elif(pol == 'x'): ## save z-pol
                    values = E_vals[:,0] ## this should be the x-component...
                elif(pol == 'y'): ## save z-pol
                    values = E_vals[:,1] ## this should be the y-component...
                return values
            E.interpolate(functools.partial(q_abs, E=self.solutions_ref[0][0])) ## fields for the first frequency and antenna
            xdmf2 = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'outputPhaseAnimationE'+pol+'.xdmf', file_mode='w')
            xdmf2.write_mesh(self.refMeshdata.mesh)
            for i in range(Nframes):
                E.x.array[:] = E.x.array*np.exp(1j*2*pi/Nframes)
                xdmf2.write_function(E, i)
            xdmf2.close()