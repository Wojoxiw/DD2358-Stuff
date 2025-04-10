## working example of a script that freezes when solving across 2 processors in a cluster

from mpi4py import MPI
import dolfinx
import numpy as np
import sys

print(f"{MPI.COMM_WORLD.rank=} {MPI.COMM_WORLD.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
sys.stdout.flush()
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u_r = dolfinx.fem.Function(V, dtype=np.float64) 
u_r.interpolate(lambda x: x[0])
u_c = dolfinx.fem.Function(V, dtype=np.complex128)
u_c.interpolate(lambda x:0.5*x[0]**2 + 1j*x[1]**2)
#print(u_r.x.array.dtype)
#print(u_c.x.array.dtype)

from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector
#print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'


import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1 - 2j))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx


L2 = f * ufl.conj(v) * ufl.dx
#print(L)
#print(L2)

J = u_c**2 * ufl.dx
F = ufl.derivative(J, u_c, ufl.conj(v))
residual = assemble_vector(dolfinx.fem.form(F))
#print(residual.array)


mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
bc = dolfinx.fem.dirichletbc(u_c, boundary_dofs)
petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)

print('solving problem')
sys.stdout.flush()
uh = problem.solve()
print('problem solved')
sys.stdout.flush()

x = ufl.SpatialCoordinate(mesh)
u_ex = 0.5 * x[0]**2 + 1j*x[1]**2
L2_error = dolfinx.fem.form(ufl.dot(uh-u_ex, uh-u_ex) * ufl.dx(metadata={"quadrature_degree": 5}))
local_error = dolfinx.fem.assemble_scalar(L2_error)
global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))
max_error = mesh.comm.allreduce(np.max(np.abs(u_c.x.array-uh.x.array)))
#print(global_error, max_error)

print('done')

