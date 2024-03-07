import warnings
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, create_unit_cube, create_unit_interval, CellType
from dolfinx.fem import ElementMetaData, functionspace, Function
from dolfinx.io import XDMFFile, VTKFile, VTXWriter, FidesWriter

class TGV:
    """
    class for the velocity field of a standard taylor green vortex
    """
    def __init__(self, t =0.0, nu = 0.1):
        self.t=t
        self.nu=nu
    def __call__(self, x: np.ndarray):
        values = np.zeros((2, x.shape[1]))
        values[0] = -np.cos(x[0])*np.sin(x[1])*np.exp(-2.0*self.nu*self.t)
        values[1] = np.sin(x[0])*np.cos(x[1])*np.exp(-2.0*self.nu*self.t)
        return values
    
def f( x: np.ndarray):
        nu=3
        t=1
        values = np.zeros((2, x.shape[1]))
        values[0] = -np.cos(x[0])*np.sin(x[1])*np.exp(-2.0*nu*t)
        values[1] = np.sin(x[0])*np.cos(x[1])*np.exp(-2.0*nu*t)
        return values

def get_function(tdim: int, celltype: CellType, dim: int, family: str, order: int):
    n = 4
    if tdim == 1:
        warnings.warn("Ignoring Cell Type in 1D.")
        domain = create_unit_interval(MPI.COMM_WORLD, n)
    elif tdim == 2:
        domain = create_unit_square(MPI.COMM_WORLD, n,n, cell_type = celltype)
    elif tdim == 3:
        domain = create_unit_cube(MPI.COMM_WORLD, n,n,n, cell_type = celltype)
    else:
        raise ValueError("Only dimensions 1,2,3 available.")
    element = ElementMetaData(family, order, shape = (dim,))
    FS = functionspace(domain, element)
    u = Function(FS)
    u.x.array[:] = np.random.rand(np.shape(u.x.array[:])[0])

    return (u,domain)

#ANCHOR - Initializing function (Taylor Green Vortex) to compare outputs
tgv = TGV(t=0.0)


#SECTION - XDMF
#NOTE - XDMF only supports Lagrange Elements of order <=1
#SECTION - Context Manager
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)
tgv.t = 0
u.interpolate(tgv)
with XDMFFile(MPI.COMM_WORLD, 'outputs/output_stationary.xdmf', "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)
#!SECTION
    
#SECTION - Instationary case, stationary mesh
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)
xdmf = XDMFFile(MPI.COMM_WORLD, 'outputs/output_instationary.xdmf', "w")
xdmf.write_mesh(domain)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    xdmf.write_function(u, t)
xdmf.close()
#!SECTION
    
#TODO - time dependent mesh
    
#!SECTION

#SECTION - VTK
#NOTE - VTK supports arbitrary order Lagrange finite elements for the geometry description. XDMF is the preferred format for geometry order <= 2.
#SECTION - with Context Manager
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 4)
tgv.t = 0
u.interpolate(tgv)
with VTKFile(MPI.COMM_WORLD, "outputs/stationary.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(u)
#!SECTION

#SECTION - Instationary case, stationary mesh
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 4)
vtk = VTKFile(MPI.COMM_WORLD, "outputs/instationary.pvd", "w")
vtk.write_mesh(domain)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    vtk.write_function(u, t)
vtk.close()
#!SECTION    

#TODO - time dependent mesh

#!SECTION


    
#SECTION - VTX
#NOTE - VTX supports arbitrary order Lagrange finite elements for the geometry description and arbitrary order (discontinuous) Lagrange finite elements for Functions. All Functions for output must share the same mesh and have the same element type.
#NOTE - Paraview: only available for version 5.12 and higher
#SECTION - with Context Manager
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 4)
tgv.t = 0
u.interpolate(tgv)
with VTXWriter(MPI.COMM_WORLD, "outputs/stationary.bp", u) as vtx:
    vtx.write(0.0)
#!SECTION

# #SECTION - Instationary case, stationary mesh
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 4)
vtx = VTXWriter(MPI.COMM_WORLD, "outputs/instationary.bp", u)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    vtx.write(t)
vtx.close()
#!SECTION    

#TODO - time dependent mesh

#!SECTION


#SECTION - Fides
#NOTE - Fides (https://fides.readthedocs.io/) supports first order Lagrange finite elements for the geometry description and first order Lagrange finite elements for functions. All functions have to be of the same element family and same order.
#NOTE - Paraview: only available for version 5.12 and higher
#SECTION - with Context manager
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)
tgv.t = 0
u.interpolate(tgv)
with FidesWriter(MPI.COMM_WORLD, "outputs/stationary-fides.bp", u) as fides:
    fides.write(0.0)
#!SECTION

#SECTION - Instationary case, stationary mesh
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)
fides = FidesWriter(MPI.COMM_WORLD, "outputs/instationary-fides.bp", u)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    fides.write(t)
fides.close()
#!SECTION    

#TODO - time dependent mesh

#!SECTION 

