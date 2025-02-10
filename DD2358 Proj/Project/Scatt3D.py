### Modification of scatt2d to handle 3d geometry
# Stripped down for simplicity for DD2358 course
#
# Daniel Sjoberg, 2024-12-13
# Alexandros Pallaris, after that

# use pytest
# use pylint
# use sphinx


from mpi4py import MPI
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc
import gmsh
from scipy.constants import pi, c as c0
from matplotlib import pyplot as plt
import pyvista as pv
import pyvistaqt as pvqt
import functools
import sys
from timeit import default_timer as timer
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import postprocess3DStuff as pp3D
from functools import wraps

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        print(f'@timefn: {fn.__name__} took {t2-t1} seconds ({(t2-t1)/3600:.3f} hours)')
        return result
    return measure_time

@timefn
def runScatt3d(runName, reference = False, folder = 'data3D', verbose=True, viewGMSH=False): ## so this can be run iteratively from other scripts - currently starting with a cylinder in the middle
    ## runName - this will be prepended to the various saved filenames
    # folder - folder to save data and stuff in
    # reference - runs without any defects contained in the mesh - possibly saves only what's needed for calculations, too
    ## verbose - prints some extra stuff about how many freqs have been calculated (things you dont want to see when iterating)
    # viewGMSH to just make the mesh, view it with fltk, then close
    
    
    startTime = timer()
    
    # MPI settings
    comm = MPI.COMM_WORLD
    model_rank = 0
    
    # Physical constants and geometry dimension
    mu0 = 4*np.pi*1e-7
    eps0 = 1/c0**2/mu0
    eta0 = np.sqrt(mu0/eps0)
    tdim = 3
    fdim = tdim - 1
    
    # Simulation variables
    f0 = 10e9                       # Design frequency
    f1 = 8e9                        # Start frequency
    f2 = 12e9                       # Stop frequency
    Nf = 1                          # Number of frequency points
    fvec = np.linspace(f1, f2, Nf)  # Vector of simulation frequencies
    lambda0 = c0/f0                 # Design wavelength
    k0 = 2*np.pi/lambda0            # Design wavenumber
    h = lambda0/12                  # Mesh size  (normally lambda0/20 with degree 1 fem is what we have used)
    fem_degree = 1                  # Degree of finite elements
    
    
    R_dom = 2*lambda0                 # Radius of domain
    d_pml = lambda0                    # Thickness of PML
    R_pml = R_dom + d_pml              # Outer radius of PML
    height_dom = 1.2*lambda0           # Height of domain - goes from -height/2 to height/2
    height_pml = height_dom + 2*d_pml  # Height of PML - goes from -height/2 to height/2
    d_spheroid = 0.4*lambda0           # max. extra thickness/height of the oblate spheroid added to the domain and pml to obtain a domed ceiling
    
    # Antennas - using a box where 1 surface is the antenna
    polarization = 'vert'           # Choose between 'vert' and 'horiz'
    antenna_width = 0.7625*lambda0  # Width of antenna apertures, 22.86 mm
    antenna_height = 0.3625*lambda0 # Height of antenna apertures
    antenna_depth = lambda0/10      # Depth of antenna box
    antenna_z_offset = 0 #lambda0/2    # raise the antennas in z to create asymmetry in z
    kc = np.pi/antenna_width        # Cutoff wavenumber of antenna
    N_antennas = 1                  # Number of antennas
    if(runName.startswith('sphereScatt')):
        print('Running a sphere scattering test')
        N_antennas = 1 ## just 1 antenna if testing S11 with a sphere
        Nf = 50 ## many freq points for a smoother graph
        fvec = np.linspace(f1, f2, Nf)
        R_sphere = 0.5*lambda0
    R_antennas = R_dom - .25*lambda0          # Radius at which antennas are placed - close to the edge, for maximum dist?
    phi_antennas = np.linspace(0, 2*np.pi, N_antennas + 1)[:-1]
    pos_antennas = np.array([[R_antennas*np.cos(phi), R_antennas*np.sin(phi), antenna_z_offset] for phi in phi_antennas])
    rot_antennas = phi_antennas + np.pi/2
    
    # Background, DUT, and defect
    epsr_bkg = 1.0
    mur_bkg = 1.0
    epsr_mat = 3.0*(1 - 0.01j)*0+1
    epsr_defect = 2.5*(1 - 0.01j)*0+1
    mur_mat = 1.0
    mur_defect = 1.0
    
    
    # Set up mesh using gmsh
    gmsh.initialize()
    if comm.rank == model_rank:
        if(verbose):
            estmem = pp3D.memoryTakenPlots( [height_pml*R_pml**2*pi/h**3] )
            esttime = pp3D.timeTakenPlots( [height_pml*R_pml**2*pi/h**3*Nf] )
            print('Variables created, generating mesh...')
        # Typical mesh size
        gmsh.option.setNumber('General.Verbosity', 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h) ## minimum mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h) ## max. mesh size
    
        # Create antennas
        antennas_DimTags = []
        x_antenna = np.zeros((N_antennas, 3))
        x_pec = np.zeros((N_antennas, 5, 3)) ### for each antenna, and PEC surface (of which there are 5), a position of that surface
        inAntennaSurface = []
        inPECSurface = []
        for n in range(N_antennas):
            box = gmsh.model.occ.addBox(-antenna_width/2, -antenna_depth, -antenna_height/2, antenna_width, antenna_depth, antenna_height) ## the antenna surface at (0, 0, 0)
            #rect = gmsh.model.occ.addRectangle(-antenna_width/2, -antenna_thickness, 0, antenna_width, antenna_thickness)
            gmsh.model.occ.rotate([(tdim, box)], 0, 0, 0, 0, 0, 1, rot_antennas[n])
            gmsh.model.occ.translate([(tdim, box)], pos_antennas[n,0], pos_antennas[n,1], pos_antennas[n,2])
            antennas_DimTags.append((tdim, box))
            x_antenna[n] = pos_antennas[n, :] ## the translation to the antenna's position
            Rmat = np.array([[np.cos(rot_antennas[n]), -np.sin(rot_antennas[n]), 0],
                             [np.sin(rot_antennas[n]), np.cos(rot_antennas[n]), 0],
                             [0, 0, 1]]) ## matrix for rotation about the z-axis
            x_pec[n, 0] = x_antenna[n] + np.dot(Rmat, np.array([0, -antenna_depth/2, -antenna_height/2])) ## bottom surface (in z)
            x_pec[n, 1] = x_antenna[n] + np.dot(Rmat, np.array([0, -antenna_depth/2,  antenna_height/2])) ## top surface (in z)
            x_pec[n, 2] = x_antenna[n] + np.dot(Rmat, np.array([-antenna_width/2, -antenna_depth/2, 0])) ## left surface (in x)
            x_pec[n, 3] = x_antenna[n] + np.dot(Rmat, np.array([antenna_width/2, -antenna_depth/2, 0])) ## right surface (in x)
            x_pec[n, 4] = x_antenna[n] + np.dot(Rmat, np.array([0, -antenna_depth, 0])) ## back surface (in y)
            inAntennaSurface.append(lambda x: np.allclose(x, x_antenna[n])) ## (0, 0, 0) - the antenna surface
            inPECSurface.append(lambda x: np.allclose(x, x_pec[n,0]) or np.allclose(x, x_pec[n,1]) or np.allclose(x, x_pec[n,2]) or np.allclose(x, x_pec[n,3]) or np.allclose(x, x_pec[n,4]))
    
        # Create object and defect
        sphereDimTags = [] ## will be empty unless it is a sphere run
        if(runName.startswith('sphereScatt')): ## just a sphere, in the centre
            disp =  -R_dom+R_sphere+lambda0*.05 #lambda0/1000 ## some tiny displacement so the center of mass isnt the same as the domain. Or a large displacement so it is far from the antenna
            matDimTags = []
            defectDimTags = []
            sphereActual = gmsh.model.occ.addSphere(disp,0,0, R_sphere) ## the actual sphere should not be in matDimTags
            sphereDimTags.append((tdim, sphereActual))
            ## hopefully I can just append the surface to this, now (its based on CoM - so presumably just the center of the sphere? - this does catch the center of the domain and PML too, though):
            inPECSurface.append(lambda x: np.allclose(x, np.array([disp,0,0])))
        elif(runName.startswith('noObject')): ## don't create any object
            matDimTags = []
            defectDimTags = []
        else:  # Just a cylinder inside a cylinder for now
            # Object variables
            R_cyl = .6*lambda0               # Radius of cylinder
            height_cyl = .5*lambda0            # Height of cylinder
            
            defect_radius = 0.15*lambda0
            
            cylinder = gmsh.model.occ.addCylinder(0,0,-height_cyl/2,0,0,height_cyl,R_cyl)
            matDimTags, matDimTagsMap = gmsh.model.occ.fuse([(tdim, cylinder)], [(tdim, cylinder)], removeTool = True)
    
            if (not reference):
                defect1 = gmsh.model.occ.addCylinder(0,0,-height_cyl/2*.8,0,0,height_cyl*.8, defect_radius)
                defectDimTags, defectDimTagsMap = gmsh.model.occ.fuse([(tdim, defect1)], [(tdim, defect1)], removeTool = True)
    
        # Create domain and PML region
        domain_cyl = gmsh.model.occ.addCylinder(0, 0, -height_dom/2, 0, 0, height_dom, R_dom)
        domain = [(tdim, domain_cyl)] # dim, tags
        pml_cyl = gmsh.model.occ.addCylinder(0, 0, -height_pml/2, 0, 0, height_pml, R_pml)
        pml = [(tdim, pml_cyl)] # dim, tags
        if(d_spheroid>0): ## add a spheroid domed top and bottom with some specified extra height, that passes through the cylindrical 'corner' (to try to avoid waves being parallel to the PML)
            
            R_extra_spheroid_dom = R_dom*(-1 + np.sqrt(1-( 1 - 1/(1- (height_dom/2/(height_dom/2+d_spheroid))**2 ) )) )       # how far beyond the domain the spheroid goes, radially. this will not actually be part of the domain. find using optimization
            a_dom = (height_dom/2+d_spheroid)/(R_dom+R_extra_spheroid_dom)
            domain_spheroid = gmsh.model.occ.addSphere(0, 0, 0, R_dom+R_extra_spheroid_dom)
            gmsh.model.occ.dilate([(tdim, domain_spheroid)], 0, 0, 0, 1, 1, a_dom)
            domain_extraheight_cyl = gmsh.model.occ.addCylinder(0, 0, -height_dom/2-d_spheroid, 0, 0, height_dom+d_spheroid*2, R_dom)
            domed_ceilings = gmsh.model.occ.intersect([(tdim, domain_spheroid)], [(tdim, domain_extraheight_cyl)])
            domain = gmsh.model.occ.fuse([(tdim, domain_cyl)], domed_ceilings[0])[0] ## [0] to get  dimTags
            
            R_extra_spheroid_pml = R_pml*(-1 + np.sqrt(1-( 1 - 1/(1- (height_pml/2/(height_pml/2+d_spheroid))**2 ) )))
            a_pml = (height_pml/2+d_spheroid)/(R_pml+R_extra_spheroid_pml)
            pml_spheroid = gmsh.model.occ.addSphere(0, 0, 0, R_pml+R_extra_spheroid_pml)
            gmsh.model.occ.dilate([(tdim, pml_spheroid)], 0, 0, 0, 1, 1, a_pml)
            pml_extraheight_cyl = gmsh.model.occ.addCylinder(0, 0, -height_pml/2-d_spheroid, 0, 0, height_pml+d_spheroid*2, R_pml)
            domed_ceilings = gmsh.model.occ.intersect([(tdim, pml_spheroid)], [(tdim, pml_extraheight_cyl)])
            pml = gmsh.model.occ.fuse([(tdim, pml_cyl)], domed_ceilings[0])[0]
    
        # Create fragments and dimtags
        outDimTags, outDimTagsMap = gmsh.model.occ.fragment(pml, domain + matDimTags + defectDimTags + sphereDimTags + antennas_DimTags)
        if(runName.startswith('sphereScatt')):
            removeDimTags = [x for x in [y[0] for y in outDimTagsMap[-(N_antennas+1):]]] ## also the sphere
        else:
            removeDimTags = [x for x in [y[0] for y in outDimTagsMap[-N_antennas:]]]
        if (reference): ## no defects
            defectDimTags = []
        else:
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
    
        # Identify antenna surfaces and make physical groups
        pec_surface = []
        antenna_surface = []
        for boundary in gmsh.model.occ.getEntities(dim=fdim):
            CoM = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            ##for sphere, not sure where the center of mass should be
            
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
        gmsh.write('tmp.msh')
        if(verbose):
            t2=timer()
            print('Mesh created in '+str(t2-startTime)+' s')
        if viewGMSH: ## show the mesh, then stop
            gmsh.fltk.run()
            exit()
    else:
        mat_marker = None
        defect_marker = None
        domain_marker = None
        pml_marker = None
        pec_surface_marker = None
        antenna_surface_markers = None
    mat_marker = comm.bcast(mat_marker, root=model_rank)
    defect_marker = comm.bcast(defect_marker, root=model_rank)
    domain_marker = comm.bcast(domain_marker, root=model_rank)
    pml_marker = comm.bcast(pml_marker, root=model_rank)
    pec_surface_marker = comm.bcast(pec_surface_marker, root=model_rank)
    antenna_surface_markers = comm.bcast(antenna_surface_markers, root=model_rank)
    
    mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm=comm, rank=model_rank, gdim=tdim)
    gmsh.finalize()
    
    mesh.topology.create_connectivity(tdim, tdim)
    
    # Set up FEM function spaces and boundary conditions.
    curl_element = basix.ufl.element('N1curl', mesh.basix_cell(), fem_degree)
    Vspace = dolfinx.fem.functionspace(mesh, curl_element)
    
    # Create measures for subdomains and surfaces
    dx = ufl.Measure('dx', domain=mesh, subdomain_data=subdomains, metadata={'quadrature_degree': 5})
    dx_dom = dx((domain_marker, mat_marker, defect_marker))
    dx_pml = dx(pml_marker)
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=boundaries)
    ds_antennas = [ds(m) for m in antenna_surface_markers]
    ds_pec = ds(pec_surface_marker)
    
    # Set up material parameters
    Wspace = dolfinx.fem.functionspace(mesh, ("DG", 0))
    epsr = dolfinx.fem.Function(Wspace)
    mur = dolfinx.fem.Function(Wspace)
    epsr.x.array[:] = epsr_bkg
    mur.x.array[:] = mur_bkg
    mat_cells = subdomains.find(mat_marker)
    mat_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=mat_cells)
    defect_cells = subdomains.find(defect_marker)
    defect_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=defect_cells)
    epsr.x.array[mat_dofs] = epsr_mat
    mur.x.array[mat_dofs] = mur_mat
    epsr.x.array[defect_dofs] = epsr_mat
    mur.x.array[defect_dofs] = mur_mat
    epsr_array_ref = epsr.x.array.copy()
    epsr.x.array[defect_dofs] = epsr_defect
    mur.x.array[defect_dofs] = mur_defect
    epsr_array_dut = epsr.x.array.copy()
    
    # Set up PML layer
    def pml_stretch(y, x, k0, x_dom=0, x_pml=1, n=3, R0=1e-10):
        return y*(1 - 1j*(n + 1)*np.log(1/R0)/(2*k0*np.abs(x_pml - x_dom))*((x - x_dom)/(x_pml - x_dom))**n)
    
    def pml_epsr_murinv(pml_coords):
        J = ufl.grad(pml_coords)
    
        # Transform the 2x2 Jacobian into a 3x3 matrix.
        #r_pml = ufl.sqrt(pml_coords[0]**2 + pml_coords[1]**2)
        #J = ufl.as_matrix(((J[0, 0], J[0, 1], 0),
        #                   (J[1, 0], J[1, 1], 0),
        #                   (0, 0, r_pml / r)))
        A = ufl.inv(J)
        epsr_pml = ufl.det(J) * A * epsr * ufl.transpose(A)
        mur_pml = ufl.det(J) * A * mur * ufl.transpose(A)
        murinv_pml = ufl.inv(mur_pml)
        return epsr_pml, murinv_pml
    
    x, y, z = ufl.SpatialCoordinate(mesh)
    r = ufl.real(ufl.sqrt(x**2 + y**2)) ## cylindrical radius. need to set this to real because I compare against it later
    height_dom_spheroid = ufl.conditional(ufl.ge(r, R_dom), height_dom/2, (height_dom/2+d_spheroid)*ufl.real(ufl.sqrt(1-(r/(R_dom+R_extra_spheroid_dom))**2)))   ##whatever the great height is, spheroid or domain
    height_pml_spheroid = (height_pml/2+d_spheroid)*ufl.sqrt(1-(r/(R_pml+R_extra_spheroid_pml))**2) ##should always be ge the pml cylinder's height
    x_stretched = pml_stretch(x, r, k0, x_dom=R_dom, x_pml=R_pml)
    y_stretched = pml_stretch(y, r, k0, x_dom=R_dom, x_pml=R_pml)
    z_stretched = pml_stretch(z, abs(z), k0, x_dom=height_dom_spheroid, x_pml=height_pml_spheroid) ## /2 since the height is from - to +
    x_pml = ufl.conditional(ufl.ge(abs(r), R_dom), x_stretched, x) ## stretch when outside radius of the domain
    y_pml = ufl.conditional(ufl.ge(abs(r), R_dom), y_stretched, y) ## stretch when outside radius of the domain
    z_pml = ufl.conditional(ufl.ge(abs(z), height_dom_spheroid), z_stretched, z) ## stretch when outside the height of the cylinder of the domain (or oblate spheroid roof with factor a_dom/a_pml - should only be higher/lower inside the domain)
    pml_coords = ufl.as_vector((x_pml, y_pml, z_pml))
    epsr_pml, murinv_pml = pml_epsr_murinv(pml_coords)
    
    
    
    # Excitation and boundary conditions
    def Eport(x, pol=polarization):
        """Compute the normalized electric field distribution in all ports."""
        Ep = np.zeros((3, x.shape[1]), dtype=complex)
        for p in range(N_antennas):
            center = pos_antennas[p]
            phi = -rot_antennas[p] # Note rotation by the negative of antenna rotation
            Rmat = np.array([[np.cos(phi), -np.sin(phi), 0],
                             [np.sin(phi), np.cos(phi), 0],
                             [0, 0, 1]]) ## rotation around z
            y = np.transpose(x.T - center)
            loc_x = np.dot(Rmat, y) ### position vector, [x, y, z] presumably, rotated to be in the coordinates the antenna was defined in
            if pol == 'vert': ## vertical (z-) pol, field varies along x
                Ep_loc = np.vstack((0*loc_x[0], 0*loc_x[0], np.cos(kc*loc_x[0])))/np.sqrt(antenna_width/2)
            else: ## horizontal (x-) pol, field varies along z
                Ep_loc = np.vstack((np.cos(kc*loc_x[2])), 0*loc_x[2], 0*loc_x[2])/np.sqrt(antenna_height/2)
                
            #simple, inexact confinement conditions
            #Ep_loc[:,np.sqrt(loc_x[0]**2 + loc_x[1]**2) > antenna_width] = 0 ## no field outside of the antenna's width (circular)
            ##if I confine it to just the 'empty face' of the waveguide thing. After testing, this seems to make no difference to just selecting the entire antenna via a sphere, with the above line
            Ep_loc[:, np.abs(loc_x[0])  > antenna_width*.54] = 0 ## no field outside of the antenna's width
            Ep_loc[:, np.abs(loc_x[1])  > antenna_depth*.04] = 0 ## no field outside of the antenna's depth - origin should be on this face - it is a face so no depth
            #for both
            Ep_loc[:,np.abs(loc_x[2]) > antenna_height*.54] = 0 ## no field outside of the antenna's height.. plus a small extra (no idea if that matters)
            
            Ep_global = np.dot(Rmat, Ep_loc)
            Ep = Ep + Ep_global
        return Ep
    Ep = dolfinx.fem.Function(Vspace)
    Ep.interpolate(lambda x: Eport(x))
    
    pec_dofs = dolfinx.fem.locate_dofs_topological(Vspace, entity_dim=fdim, entities=boundaries.find(pec_surface_marker))
    Ezero = dolfinx.fem.Function(Vspace)
    Ezero.x.array[:] = 0.0
    bc_pec = dolfinx.fem.dirichletbc(Ezero, pec_dofs)
    
    # Set up simulation
    E = ufl.TrialFunction(Vspace)
    v = ufl.TestFunction(Vspace)
    curl_E = ufl.curl(E)
    curl_v = ufl.curl(v)
    nvec = ufl.FacetNormal(mesh)
    Zrel = dolfinx.fem.Constant(mesh, 1j)
    k00 = dolfinx.fem.Constant(mesh, 1j)
    a = [dolfinx.fem.Constant(mesh, 1.0 + 0j) for n in range(N_antennas)]
    F_antennas_str = ''
    for n in range(N_antennas):
        F_antennas_str += f"""+ 1j*k00/Zrel*ufl.inner(ufl.cross(E, nvec), ufl.cross(v, nvec))*ds_antennas[{n}] - 1j*k00/Zrel*2*a[{n}]*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(v, nvec))*ds_antennas[{n}]"""
    F = ufl.inner(1/mur*curl_E, curl_v)*dx_dom \
        - ufl.inner(k00**2*epsr*E, v)*dx_dom \
        + ufl.inner(murinv_pml*curl_E, curl_v)*dx_pml \
        - ufl.inner(k00**2*epsr_pml*E, v)*dx_pml + eval(F_antennas_str)
    bcs = [bc_pec]
    lhs, rhs = ufl.lhs(F), ufl.rhs(F)
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    problem = dolfinx.fem.petsc.LinearProblem(
        lhs, rhs, bcs=bcs, petsc_options=petsc_options
    )
    
    def ComputeFields():
        S = np.zeros((Nf, N_antennas, N_antennas), dtype=complex)
        solutions = []
        for nf in range(Nf):
            print(f'Rank {comm.rank}: Frequency {nf+1} / {Nf}')
            sys.stdout.flush()
            k00.value = 2*np.pi*fvec[nf]/c0
            Zrel.value = k00.value/np.sqrt(k00.value**2 - kc**2)
            sols = []
            for n in range(N_antennas):
                for m in range(N_antennas):
                    a[m].value = 0.0
                a[n].value = 1.0
                E_h = problem.solve()
                for m in range(N_antennas):
                    factor = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(2*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(Ep, nvec))*ds_antennas[m]))
                    factors = comm.gather(factor, root=model_rank)
                    if comm.rank == model_rank:
                        factor = sum(factors)
                    else:
                        factor = None
                    factor = comm.bcast(factor, root=model_rank)
                    b = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.cross(E_h, nvec), ufl.cross(Ep, nvec))*ds_antennas[m] + Zrel/(1j*k0)*ufl.inner(ufl.curl(E_h), ufl.cross(Ep, nvec))*ds_antennas[m]))/factor
                    bs = comm.gather(b, root=model_rank)
                    if comm.rank == model_rank:
                        b = sum(bs)
                    else:
                        b = None
                    b = comm.bcast(b, root=model_rank)
                    S[nf,m,n] = b
                sols.append(E_h.copy())
            solutions.append(sols)
        return S, solutions
    
    tcomp1 = timer()
    print(f'Rank {comm.rank}: Computing REF solutions')
    sys.stdout.flush()
    epsr.x.array[:] = epsr_array_ref
    S_ref, solutions_ref = ComputeFields()
    if(not reference):
        print(f'Rank {comm.rank}: Computing DUT solutions')
        sys.stdout.flush()
        epsr.x.array[:] = epsr_array_dut
        S_dut, solutions_dut = ComputeFields()
    
    print(f'Rank {comm.rank}: Computing optimization vectors')
    sys.stdout.flush()
    Nepsr = len(epsr.x.array[:])
    if(not reference):
        b = np.zeros(Nf*N_antennas*N_antennas, dtype=complex)
    # Create function space for temporary interpolation
    q = dolfinx.fem.Function(Wspace)
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_volumes = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.conj(ufl.TestFunction(Wspace))*ufl.dx)).array
    def q_func(x, Em, En, k0, conjugate=False):
        cells = []
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
        for i, point in enumerate(x.T):
            if len(colliding_cells.links(i)) > 0:
                cells.append(colliding_cells.links(i)[0])
        Em_vals = Em.eval(x.T, cells)
        En_vals = En.eval(x.T, cells)
        if conjugate:
            En_vals = np.conjugate(En_vals)
        values = -1j*k0/eta0/2*(Em_vals[:,0]*En_vals[:,0] + Em_vals[:,1]*En_vals[:,1] + Em_vals[:,2]*En_vals[:,2])*cell_volumes
        return values
    
    xdmf = dolfinx.io.XDMFFile(comm=comm, filename=folder+'/'+runName+'output.xdmf', file_mode='w')
    xdmf.write_mesh(mesh)
    epsr.x.array[:] = cell_volumes
    xdmf.write_function(epsr, -3)
    epsr.x.array[:] = epsr_array_ref
    xdmf.write_function(epsr, -2)
    epsr.x.array[:] = epsr_array_dut
    xdmf.write_function(epsr, -1)
    for nf in range(Nf):
        print(f'Rank {comm.rank}: Frequency {nf+1} / {Nf}')
        sys.stdout.flush()
        k0 = 2*np.pi*fvec[nf]/c0
        for m in range(N_antennas):
            Em_ref = solutions_ref[nf][m]
            for n in range(N_antennas):
                #En_dut = solutions_dut[nf][n] ## only using Eref*Eref right now
                En_ref = solutions_ref[nf][n]
                # Case Eref*Eref
                if(not reference): ## ref must compute b later
                    b[nf*N_antennas*N_antennas + m*N_antennas + n] = S_dut[nf, m, n] - S_ref[nf, n, m]
                q.interpolate(functools.partial(q_func, Em=Em_ref, En=En_ref, k0=k0, conjugate=False))
                # The function q is one row in the A-matrix, save it to file
                xdmf.write_function(q, nf*N_antennas*N_antennas + m*N_antennas + n)
    xdmf.close()
    
    ###if trying to plot phase, save the final freq. point as different timestamps:
    if(True):### make this True to animate the final fields
        xdmf = dolfinx.io.XDMFFile(comm=comm, filename=folder+'/'+runName+'outputPhaseAnimation.xdmf', file_mode='w')
        xdmf.write_mesh(mesh)
        Nframes = 50
        for i in range(Nframes):
            epsr.x.array[:] = q.x.array*np.exp(1j*i*2*pi/Nframes)
            xdmf.write_function(epsr, i)
        xdmf.close()
    ###
    
    tcomp2 = timer()
    print('Computations completed in',tcomp2-tcomp1,'s,',(tcomp2-tcomp1)/3600,'hours')
    
    if comm.rank == model_rank: # Save global values for further postprocessing
        if(reference): ## less stuff to save
            np.savez(folder+'/'+runName+'output.npz', fvec=fvec, S_ref=S_ref, epsr_mat=epsr_mat, epsr_defect=epsr_defect)
        else:
            np.savez(folder+'/'+runName+'output.npz', b=b, fvec=fvec, S_ref=S_ref, S_dut=S_dut, epsr_mat=epsr_mat, epsr_defect=epsr_defect)
    
    ### could possibly try adapting the 2D video-plotting here

##MAIN STUFF
if __name__ == '__main__':
    #runName = '3dtest3' ## just testing - constantly overwritten
    #runName = 'sphereScattNoSphereFixed' ## one antenna, simple PEC sphere object - to calculate S11 and compare with theory. Here the sphere is set to epsr=1, no PEC, antenna_offset_z = lambda/2
    #runName = 'sphereScattTest' ## one antenna, simple PEC sphere object - to calculate S11 and compare with theory
    #runName = 'sphereScattTestFF' ## one antenna, simple PEC sphere object, now placed as far away as possible - to calculate S11 and compare with theory
    #runName = 'noObjectTest' ## one antenna, no object. Just to see the waves - h = l0/12
    #runName = 'noObjectTesthsmaller' ## one antenna, no object. small radius. Just to see the waves - h = l0/18 (this is near my memory limit)
    runName = 'noObjectTestDomedBoundaries' ## one antenna, no object. testing domed domain and pml
 
    folder = '/mnt/d/Microwave Imaging/data3D'
     
    runScatt3d(runName = runName, reference = True, folder = folder, viewGMSH = False)
