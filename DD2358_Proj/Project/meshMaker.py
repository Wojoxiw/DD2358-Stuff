# encoding: utf-8
## this file makes the mesh

import numpy as np
import dolfinx
from mpi4py import MPI
import gmsh
import sys
import pyvista
from scipy.constants import c as c0

f0 = 10e9                            # Design frequency
lambda0 = c0/f0                      # Design wavelength

# Use variables tdim and fdim to clarify when addressing a dimension
tdim = 2                             # Dimension of triangles/tetraedra
fdim = tdim - 1                      # Dimension of facets

class PerfectlyMatchedLayer():
    """Data structure for PML."""
    def __init__(self, d=None, radius=None, rho=None, zt=None, zb=None, n=3, cylindrical=False):
        self.d = d                     # Thickness of PML
        self.radius = radius           # Radius of spherical PML
        self.rho = rho                 # Cylindrical radius of cylindrical PML
        self.zt = zt                   # Top z value of cylindrical PML
        self.zb = zb                   # Bottom z value of cylindrical PML
        self.cylindrical = cylindrical # Whether to use cylindrical PML
        
class MeshData():
    """Data structure for the mesh and metadata."""
    def __init__(self, mesh=None, subdomains=None, boundaries=None,
            subdomain_markers={'freespace': -1,
                               'material': -1,
                               'pml': -1,
                               'pml_material_overlap': -1},
            boundary_markers={'pec': -1,
                              'antenna': -1,
                              'farfield': -1,
                              'pml': -1,
                              'axis': -1},
            PML=None,
            comm=MPI.COMM_WORLD,
            model_rank=0
    ):
        self.mesh = mesh                           # Mesh
        self.subdomains = subdomains               # Tagged subdomains
        self.boundaries = boundaries               # Tagged boundaries
        self.subdomain_markers = subdomain_markers # Dictionary of subdomain markers
        self.boundary_markers = boundary_markers   # Dictionary of boundary markers
        self.PML = PML                             # PML data
        self.comm = comm                           # MPI communicator
        self.model_rank=model_rank                 # Model rank
        

def CheckGhostFarfieldFacets(comm, model_rank, mesh, boundaries, farfield_surface_marker):
    """When running the script with a distributed mesh, it may happen that the mesh is divided exactly on one or several facets where we want to compute the far field. As the code is written, this leads to these ghost facets contributing twice to the far field integral, and there is currently no fix. Use this routine to check for the occurence of ghost far field facets, and tweak the simulation parameters if there are any."""
    ff_facets_local = boundaries.find(farfield_surface_marker)
    local_to_global = mesh.topology.index_map(fdim).local_to_global
    ff_facets_global = local_to_global(ff_facets_local)
    ff_facets_local_all = comm.gather(ff_facets_local, root=model_rank)
    ff_facets_global_all = comm.gather(ff_facets_global, root=model_rank)
    if comm.rank == model_rank:
        ghost_ff_facets = []
        for rank_i in range(comm.size-1):
            for idx_i, fff in enumerate(ff_facets_global_all[rank_i]):
                for rank_j in range(rank_i+1, comm.size):
                    if fff in ff_facets_global_all[rank_j]:
                        idx_j = np.where(ff_facets_global_all[rank_j]==fff)[0][0]
                        ghost_ff_facets.append((rank_i, rank_j, fff, ff_facets_local_all[rank_i][idx_i], ff_facets_local_all[rank_j][idx_j]))
    else:
        ghost_ff_facets = None
    ghost_ff_facets = comm.bcast(ghost_ff_facets, root=model_rank)
#    for rank_i, rank_j, gfff in ghost_ff_facets:
#        if comm.rank == rank_j:
#            if gfff in ff_facets:
#                foo = ff_facets.tolist()
#                foo.remove(gfff)
#                ff_facets = np.array(foo)
#    ff_boundary = dolfinx.mesh.meshtags(mesh, fdim, ff_facets, farfield_surface_marker)
    if len(ghost_ff_facets) > 0:
        if comm.rank == model_rank:
            print('Ghost farfield facets detected, farfield results may be inaccurate. Consider making small changes in location of far field surface, mesh size, or use different number of ranks.')
            print(f'ghost_ff_facets = {ghost_ff_facets}')
            sys.stdout.flush()
    return ghost_ff_facets

def PlotMeshPartition(comm, model_rank, mesh, ghost_ff_facets, boundaries, farfield_surface_marker):
    V = dolfinx.fem.functionspace(mesh, ('CG', 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.ones(x.shape[1])*comm.rank)
    mesh.topology.create_connectivity(fdim, 0)
    for rank_i, rank_j, gfff_global, gfff_local_i, gfff_local_j in ghost_ff_facets:
        if comm.rank == rank_i or comm.rank == rank_j:
#            fmap = mesh.topology.index_map(fdim)
#            local_to_global = fmap.local_to_global(range(fmap.size_local + fmap.num_ghosts))
#            gfff_local = np.where(local_to_global == gfff_global)[0][0]
            facets_to_nodes = mesh.topology.connectivity(fdim, 0)
            if comm.rank == rank_i:
                node_indices = facets_to_nodes.links(gfff_local_i)
            else:
                node_indices = facets_to_nodes.links(gfff_local_j)
            dof_indices = []
            nmap = mesh.topology.index_map(0)
            size_local = nmap.size_local
            for idx in node_indices:
                if idx >= size_local:
                    dof_indices.append(size_local + np.where(V.dofmap.index_map.ghosts == nmap.ghosts[idx - size_local])[0][0])
                else:
                    dof_indices.append(idx)
#            print(f'Rank {comm.rank}: {gfff_local}')
#            print(f'Rank {comm.rank}: {dof_indices}')
            u.x.array[dof_indices] = -1
    cells, cell_types, x = dolfinx.plot.vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(cells, cell_types, x)
    grid["rank"] = np.real(u.x.array)
    grids = comm.gather(grid, root=model_rank)
    if comm.rank == model_rank:
        plotter = pyvista.Plotter()
        for g in grids:
            plotter.add_mesh(g, show_edges=True)
        plotter.view_xy()
        plotter.add_axes()
        plotter.show()

def partitioner(comm, n, m, topo):
    dests = []
    offsets = [0]
    for i in range(topo.num_nodes):
        dests.append(comm.rank)
        offsets.append(len(dests))
    dests = np.array(dests, dtype=np.int32)
    offsets = np.array(offsets, dtype=np.int32)
    return dolfinx.cpp.graph.AdjacencyList_int32(dests)
    

def CreateMeshSphere(
        comm=MPI.COMM_WORLD,        # MPI communicator
        model_rank=0,               # Rank of modelling process
        radius_sphere=1.0,          # Radius of scattering sphere
        radius_farfield=1.5,        # Radius of farfield surface
        radius_domain=2.0,          # Radius of computational domain
        radius_pml=2.5,             # Outer radius of PML
        pec=False,                  # Whether to have PEC sphere or not
        PMLcylindrical=False,       # To have a PML in cylindrical coordinates
        h=0.1,                      # Typical mesh size
        verbosity=1,                # Verbosity of gmsh
        visualize=False,            # Whether to visualize the mesh
        filename='spheremesh.msh'   # Name of file to save mesh in
):
    """Create the mesh using gmsh."""

    # Set up PML data
    PML = PerfectlyMatchedLayer()
    PML.cylindrical = PMLcylindrical
    PML.d = radius_pml - radius_domain
    PML.zt = radius_pml
    PML.zb = -radius_pml
    PML.rho = radius_pml
    PML.radius = radius_pml
    FF_d = radius_pml - radius_farfield
    
    gmsh.initialize()
    if comm.rank == model_rank:
        # Typical mesh size
        gmsh.option.setNumber('General.Verbosity', verbosity)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        # Create sphere boundary and domain (if applicable)
        sphere_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_sphere, -1, -np.pi/2, np.pi/2)
        sphere_point1 = gmsh.model.occ.addPoint(0, -radius_sphere, 0)
        sphere_point2 = gmsh.model.occ.addPoint(0, radius_sphere, 0)
        if not pec:
            sphere_join = gmsh.model.occ.addLine(sphere_point1, sphere_point2)
            sphere_loop = gmsh.model.occ.addCurveLoop([sphere_boundary, sphere_join])
            sphere_domain = gmsh.model.occ.addPlaneSurface([sphere_loop])

        # Create free space domain (one on each side of the farfield boundary)
        if not PMLcylindrical: # Spherical PML
            inner_point1 = gmsh.model.occ.addPoint(0, -radius_farfield, 0)
            inner_point2 = gmsh.model.occ.addPoint(0, radius_farfield, 0)
            outer_point1 = gmsh.model.occ.addPoint(0, -radius_domain, 0)
            outer_point2 = gmsh.model.occ.addPoint(0, radius_domain, 0)
            farfield_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_farfield, -1, -np.pi/2, np.pi/2)
            domain_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_domain, -1, -np.pi/2, np.pi/2)
        else: # Cylindrical PML
            inner_point1 = gmsh.model.occ.addPoint(0, PML.zb+FF_d, 0)
            inner_point2 = gmsh.model.occ.addPoint(0, PML.zt-FF_d, 0)
            inner_point3 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zt-FF_d, 0)
            inner_point4 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zb+FF_d, 0)
            outer_point1 = gmsh.model.occ.addPoint(0, PML.zb+PML.d, 0)
            outer_point2 = gmsh.model.occ.addPoint(0, PML.zt-PML.d, 0)
            outer_point3 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zt-PML.d, 0)
            outer_point4 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zb+PML.d, 0)

            farfield_boundary = gmsh.model.occ.addBSpline([inner_point2, inner_point3, inner_point4, inner_point1], degree=1)
            domain_boundary = gmsh.model.occ.addBSpline([outer_point2, outer_point3, outer_point4, outer_point1], degree=1)

        outer_join1 = gmsh.model.occ.addLine(inner_point1, outer_point1)
        outer_join2 = gmsh.model.occ.addLine(inner_point2, outer_point2)
        inner_join1 = gmsh.model.occ.addLine(inner_point1, sphere_point1)
        inner_join2 = gmsh.model.occ.addLine(inner_point2, sphere_point2)

        outer_loop = gmsh.model.occ.addCurveLoop([farfield_boundary, outer_join2, domain_boundary, outer_join1])
        inner_loop = gmsh.model.occ.addCurveLoop([farfield_boundary, inner_join1, sphere_boundary, inner_join2])

        outer_domain = gmsh.model.occ.addPlaneSurface([outer_loop])
        inner_domain = gmsh.model.occ.addPlaneSurface([inner_loop])
            
        # Create PML domain
        if not PMLcylindrical: # Spherical PML
            pml_point1 = gmsh.model.occ.addPoint(0, -radius_pml, 0)
            pml_point2 = gmsh.model.occ.addPoint(0, radius_pml, 0)
            pml_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_pml, -1, -np.pi/2, np.pi/2)
        else: # Cylindrical PML
            pml_point1 = gmsh.model.occ.addPoint(0, PML.zb, 0)
            pml_point2 = gmsh.model.occ.addPoint(0, PML.zt, 0)
            pml_point3 = gmsh.model.occ.addPoint(PML.rho, PML.zt, 0)
            pml_point4 = gmsh.model.occ.addPoint(PML.rho, PML.zb, 0)
            pml_boundary = gmsh.model.occ.addBSpline([pml_point2, pml_point3, pml_point4, pml_point1], degree=1)
            
        pml_join1 = gmsh.model.occ.addLine(pml_point1, outer_point1)
        pml_join2 = gmsh.model.occ.addLine(pml_point2, outer_point2)
        pml_loop = gmsh.model.occ.addCurveLoop([pml_boundary, pml_join2, domain_boundary, pml_join1])
        pml_domain = gmsh.model.occ.addPlaneSurface([pml_loop])
            
        gmsh.model.occ.synchronize()

        # Create physical groups for domains
        if pec:
            sphere_marker = -1
        else:
            sphere_marker = gmsh.model.addPhysicalGroup(tdim, [sphere_domain])
        material_marker = sphere_marker
        freespace_marker = gmsh.model.addPhysicalGroup(tdim, [inner_domain, outer_domain])
        pml_marker = gmsh.model.addPhysicalGroup(tdim, [pml_domain])

        subdomain_markers = {'freespace': freespace_marker,
                             'material': material_marker,
                             'transition': -1,
                             'hull': -1,
                             'pml': pml_marker,
                             'pml_hull_overlap': -1}

        # Create physical groups for sphere surface and far field boundary
        sphere_surface_marker = gmsh.model.addPhysicalGroup(fdim, [sphere_boundary])
        if pec:
            pec_surface_marker = sphere_surface_marker
        else:
            pec_surface_marker = -1
        antenna_surface_marker = -1 # No antenna surface in this mesh
        farfield_surface_marker = gmsh.model.addPhysicalGroup(fdim, [farfield_boundary])
        pml_surface_marker = gmsh.model.addPhysicalGroup(fdim, [pml_boundary])
        if pec:
            axis_marker = gmsh.model.addPhysicalGroup(fdim, [pml_join1, pml_join2, outer_join1, outer_join2, inner_join1, inner_join2])
        else:
            axis_marker = gmsh.model.addPhysicalGroup(fdim, [pml_join1, pml_join2, outer_join1, outer_join2, inner_join1, inner_join2, sphere_join])

        gmsh.model.occ.synchronize()
        boundary_markers = {'pec': pec_surface_marker,
                            'antenna': antenna_surface_marker,
                            'farfield': farfield_surface_marker,
                            'pml': pml_surface_marker,
                            'axis': axis_marker}
    
        # Generate mesh
        gmsh.model.mesh.generate(tdim)
        gmsh.model.mesh.removeDuplicateNodes() # Some nodes seem to be duplicated

        # Save mesh for retrieval of function spaces later on
        gmsh.write(filename)

        if visualize:
            gmsh.fltk.run()
    else: # Some data generated by the meshing that is needed on all ranks
        subdomain_markers = None
        boundary_markers = None
    subdomain_markers = comm.bcast(subdomain_markers, root=model_rank)
    boundary_markers = comm.bcast(boundary_markers, root=model_rank)

    mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm=comm, rank=model_rank, gdim=tdim)
    gmsh.finalize()

    meshdata = MeshData(mesh=mesh, subdomains=subdomains, boundaries=boundaries, subdomain_markers=subdomain_markers, boundary_markers=boundary_markers, PML=PML, comm=comm, model_rank=model_rank)
    return meshdata

if __name__ == '__main__':
    # Create and visualize the mesh if run from the prompt
    if False:
        meshdata = CreateMeshOgive(visualize=True, h=0.1*lambda0, PMLcylindrical=True, PMLpenetrate=False, Antenna=True, AntennaMetalBase=False, t=lambda0/4, Htransition=1*lambda0, hfine=0.01*lambda0)
    else:
        pec = True
        meshdata = CreateMeshSphere(pec=pec, visualize=True, PMLcylindrical=True, h=0.1)

    if True:
        # Visualize mesh and dofs as handled by dolfinx
        import pyvista as pv

        mesh = meshdata.mesh
        subdomains = meshdata.subdomains
        boundaries = meshdata.boundaries
        
        freespace_marker = meshdata.subdomain_markers['freespace']
        material_marker = meshdata.subdomain_markers['material']
        transition_marker = meshdata.subdomain_markers['transition']
        pml_marker = meshdata.subdomain_markers['pml']
        pml_hull_overlap_marker = meshdata.subdomain_markers['pml_hull_overlap']
        
        pec_surface_marker = meshdata.boundary_markers['pec']
        antenna_surface_marker = meshdata.boundary_markers['antenna']
        farfield_surface_marker = meshdata.boundary_markers['farfield']
        pml_surface_marker = meshdata.boundary_markers['pml']
        axis_marker = meshdata.boundary_markers['axis']
        
        W_DG = dolfinx.fem.functionspace(mesh, ("CG", 1))
        chi = dolfinx.fem.Function(W_DG)
        chi.x.array[:] = 0.0
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        mesh.topology.create_connectivity(mesh.topology.dim, 1)
        mesh.topology.create_connectivity(mesh.topology.dim, 2)
        if True: # Indicate regions
            freespace_cells = subdomains.find(freespace_marker)
            freespace_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=freespace_cells)
            if material_marker >= 0:
                material_cells = subdomains.find(material_marker)
                material_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=material_cells)
                chi.x.array[material_dofs] = 2.0
            if pml_hull_overlap_marker >= 0:
                pml_hull_overlap_cells = subdomains.find(pml_hull_overlap_marker)
                pml_hull_overlap_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=pml_hull_overlap_cells)
                chi.x.array[pml_hull_overlap_dofs] = 4.0
            pml_cells = subdomains.find(pml_marker)
            pml_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=pml_cells)
            chi.x.array[freespace_dofs] = 1.0
            chi.x.array[pml_dofs] = 3.0
        if True: # Indicate boundaries
            if pec_surface_marker >= 0:
                pec_surface_cells = boundaries.find(pec_surface_marker)
                pec_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=pec_surface_cells)
                chi.x.array[pec_surface_dofs] = 10.0
            if antenna_surface_marker >= 0:
                antenna_surface_cells = boundaries.find(antenna_surface_marker)
                antenna_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=antenna_surface_cells)
                chi.x.array[antenna_surface_dofs] = 20.0
            farfield_surface_cells = boundaries.find(farfield_surface_marker)
            farfield_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=farfield_surface_cells)
            pml_surface_cells = boundaries.find(pml_surface_marker)
            pml_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=pml_surface_cells)
            axis_cells = boundaries.find(axis_marker)
            axis_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=axis_cells)
            chi.x.array[farfield_surface_dofs] = 30.0
            chi.x.array[pml_surface_dofs] = 40.0
            chi.x.array[axis_dofs] = 50.0
        cells, cell_types, x = dolfinx.plot.vtk_mesh(mesh, tdim)
        grid = pv.UnstructuredGrid(cells, cell_types, x)
        grid["chi"] = np.real(chi.x.array)
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.add_axes()
        plotter.show()
