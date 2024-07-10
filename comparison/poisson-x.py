import time
import sys
import numpy as np
tottime= 0




# SECTION - check which version is used
from importlib.util import find_spec
is_dolfinx      = find_spec("dolfinx") is not None
is_dolfin       = find_spec("dolfin") is not None

#NOTE - prepare Log
if is_dolfin: s = "dolfin"
elif is_dolfinx: s = "dolfinx"
logfile = open("output/"+s+'-log.txt', 'w')
def log(*args, **kwargs):        
        lists = [str(item) for item in args]+[str(item) for item in kwargs]
        logfile.write("".join(lists)+"\n")
        print(*args, **kwargs)

#NOTE - print versions
from importlib.metadata import version
if is_dolfinx:  log("Dolfinx installed, version ",        version("fenics-dolfinx"))
if is_dolfin:   log("Dolfin legacy installed, version ",  version("fenics-dolfin"))
# !SECTION




# SECTION - import statement
if is_dolfin:
    from fenics import *
    import matplotlib.pyplot as plt
    """
    --> import of full namespace
    + easy
    - very full namespace
    """
else:
    from mpi4py import MPI
    from petsc4py.PETSc import ScalarType  # type: ignore
    import ufl
    from dolfinx import fem, io, mesh, plot
    from dolfinx.fem.petsc import LinearProblem
    from ufl import ds, dx, grad, inner
    """
    --> only partial import of namespace
    + less cluttered namespace --> less error prone
    + more explicit parallelization
    - more work
    """
#!SECTION




# SECTION - Create mesh and define function space
start = time.time()

nx, ny = 50, 50
if is_dolfin:
    domain = UnitSquareMesh(nx, ny)
    V = FunctionSpace(domain, 'P', 1)
    """
    +-0 
    """

elif is_dolfinx:
    domain = mesh.create_unit_square(comm=MPI.COMM_WORLD,    nx = nx, ny= ny,    cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    """
    +-0
    """

tottime += time.time() - start
log("Elapsed time for creating mesh and function space ", time.time() - start , " total time elapsed: ", tottime)
#!SECTION




# SECTION - Define boundary condition
start = time.time()

if is_dolfin:
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    """
    + easy at first implementation
    - lots of implicit choices
    """

elif is_dolfinx:
    def u_D(x: np.ndarray)-> np.ndarray:
        return 1 + x[0]*x[0] + 2*x[1]*x[1]
    
    u_Dh = fem.Function(V)
    u_Dh.interpolate(u_D)

    # Detect dofs either topologically or geometrically

    # topologically: easier when boundary is given by external mesh workflow or FE do not have coordinate DOFs
    # facets = mesh.locate_entities_boundary(  domain,  dim=(domain.topology.dim - 1),  marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    # dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

    # geometrically
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    bc = fem.dirichletbc(u_Dh, dofs=dofs)

    """
    + very explicit
    + syntax close to actual data structures
    - much more complex
    """

tottime += time.time() - start
log("Elapsed time for creating boundary conditions ", time.time() - start, " total time elapsed: ", tottime)
#!SECTION




# SECTION - Define variational problem
start = time.time()

if is_dolfin:
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression('10 * exp( (-1.0)*( pow(x[0] - 0.5,2.0) + pow(x[1] - 0.5, 2.0)) / 0.02 )', degree=1) # This is C++ syntax
    g = Expression('sin(5 * x[0])', degree=2)

    """
    - inconsequent mix of python and C++ syntax
    """

elif is_dolfinx:
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    # .. explanation
    f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02) # This is python syntax
    g = ufl.sin(5 * x[0])

    """
    + pythonic
    + makes use of existing and popular numpy data structures
    """

LHS = inner(grad(u), grad(v))*dx
RHS =  inner(f, v) * dx + inner(g, v) * ds 

tottime += time.time() - start
log("Elapsed time for creating mesh and function space ", time.time() - start, " total time elapsed: ", tottime)
#!SECTION




# SECTION - Setting up linear problem
start = time.time()


if is_dolfin:
    uh = Function(V)
    problem = LinearVariationalProblem(LHS, RHS, uh, bc)
    solver = LinearVariationalSolver(problem)
    prm = LinearVariationalSolver.default_parameters()
    prm["linear_solver"] = "gmres"
    prm["preconditioner"] = "ilu"

elif is_dolfinx:
    uh = fem.Function(V)
    solver_parameters =  {"ksp_type": "gmres", "pc_type": "ilu"}
    problem = LinearProblem(LHS, RHS, bcs=[bc], u = uh, petsc_options=solver_parameters)

tottime += time.time() - start
log("Elapsed time for setting up the linear problem and solver ", time.time() - start, " total time elapsed: ", tottime)
#!SECTION




# SECTION - Compute solution
start = time.time()

if is_dolfin:
    # solve(LHS == RHS, uh, bc) # blackbox approach, not optimized
    solver.solve()

elif is_dolfinx:
    uh = problem.solve()

tottime += time.time() - start
log("Elapsed time for computing the solution", time.time() - start, " total time elapsed: ", tottime)
#!SECTION




# SECTION - Save solutions
start = time.time()

if is_dolfin:
    xmdffile = XDMFFile('output/poisson-dolfin.xdmf')
    xmdffile.write(uh)
    """
    - old formats, lots of them deprecated
    """
if is_dolfinx:
    with io.XDMFFile(domain.comm, "output/poisson-dolfinx.xdmf", "w") as file:
        file.write_mesh(domain)
        # file.write_meshtags(mesh.meshtags(domain,1, facets, 1), domain.geometry)
        file.write_function(uh)
    """
    + more flexibility wrt exact representation of FEs
    - more complexity, syntax depends strongly on output format
    """

tottime += time.time() - start
log("Elapsed time for saving the solution", time.time() - start, " total time elapsed: ", tottime)
#!SECTION

# SECTION - Compute error in L2 norm
start = time.time()

#NOTE - The boundary condition is not the exact solution. This is just for the purpose of comparison.
if is_dolfin:
    error_L2 = errornorm(u_D, uh, 'L2')
    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(domain)
    vertex_values_uh = uh.compute_vertex_values(domain)
    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_uh))
    # Print errors
    log('error_L2  =', error_L2)
    log('error_max =', error_max)
    """
    + pre-defined methods for error computation
    - very implicit
    """

elif is_dolfinx:

    def errornorm(uh: fem.Function, u_ex: fem.Function, norm ="L2", degree_raise: int =3)-> float:
        from ufl.core.expr import Expr
        FS = uh.function_space
        
        # Create higher order function space
        dim = FS.num_sub_spaces
        degree = FS.ufl_element().degree()
        family = FS.ufl_element().family()
        mesh = FS.mesh
        
        if degree_raise > 0:   
            W = fem.functionspace(mesh, fem.ElementMetaData(family, degree+ degree_raise) ) #v0.7     
        else:
            W = FS
        # W is the Function space in which the error will be computed
        # Interpolate approximate solution
        u_W = fem.Function(W)
        u_W.interpolate(uh)


        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_ex_W = fem.Function(W)
        if isinstance(u_ex, Expr):
            log(isinstance(u_ex, Expr))
            u_expr = fem.Expression(u_ex, W.element.interpolation_points())
            u_ex_W.interpolate(u_expr)
        else:
            u_ex_W.interpolate(u_ex)
        
        # Compute the error in the higher order function space
        e_W = fem.Function(W)
        e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
        
        # Integrate the error
        if norm == "L2":
            error = fem.form(inner(e_W, e_W) * dx)
            error_local = fem.assemble_scalar(error)
            error_global = np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))
        elif norm == "inf":
            error_global = np.linalg.norm(e_W.x.array[:], np.inf)

        return error_global 

    error_L2 = errornorm(uh, u_D, norm = "L2")
    error_max = errornorm(uh, u_D, norm = "inf")
    # Print errors
    log('error_L2  =', error_L2)
    log('error_max =', error_max)

    """
    + very explicit, forces you to check what you compute
    - very complex, high effort
    """

tottime += time.time() - start
log("Elapsed time for error computation ", time.time() - start, " total time elapsed: ", tottime)
#!SECTION




# Plot solution and mesh
if is_dolfin:
    plot(uh)
    plot(domain)

    # Hold plot
    plt.show()

    """
    + easy first visualization
    - matplotlib syntax
    - limited in visualization of unstructured meshes
    """

elif is_dolfinx:
    import pyvista
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    # warped = grid.warp_by_scalar()
    # plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("output/uh_poisson.png")
    else:
        plotter.show()
    
    """
    - more complex syntax
    + more flexibility
    """