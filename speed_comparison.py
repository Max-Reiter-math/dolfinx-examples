import warnings
import time
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, create_unit_cube, create_unit_interval, CellType
from dolfinx.fem import ElementMetaData, functionspace, Function
from dolfinx.io import XDMFFile, VTKFile, VTXWriter, FidesWriter

# dolfinx version v0.7.3

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
time0 = time.time()
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)
xdmf = XDMFFile(MPI.COMM_WORLD, 'outputs/output_instationary.xdmf', "w")
xdmf.write_mesh(domain)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    xdmf.write_function(u, t)
xdmf.close()
time1 = time.time()

print("saving to an xdmf file takes ", time1-time0)
#!SECTION

#SECTION - VTK
#NOTE - VTK supports arbitrary order Lagrange finite elements for the geometry description. XDMF is the preferred format for geometry order <= 2.
time0 = time.time()
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)
vtk = VTKFile(MPI.COMM_WORLD, "outputs/instationary.pvd", "w")
vtk.write_mesh(domain)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    vtk.write_function(u, t)
vtk.close()
time1 = time.time()

print("saving to an vtk/pvd file takes ", time1-time0)
#!SECTION


    
#SECTION - VTX
#NOTE - VTX supports arbitrary order Lagrange finite elements for the geometry description and arbitrary order (discontinuous) Lagrange finite elements for Functions. All Functions for output must share the same mesh and have the same element type.
#NOTE - Paraview: only available for version 5.12 and higher
time0 = time.time()
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 4)
vtx = VTXWriter(MPI.COMM_WORLD, "outputs/instationary.bp", u)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    vtx.write(t)
vtx.close()
time1 = time.time()

print("saving with VTXWriter takes ", time1-time0)
#!SECTION


#SECTION - Fides
#NOTE - Fides (https://fides.readthedocs.io/) supports first order Lagrange finite elements for the geometry description and first order Lagrange finite elements for functions. All functions have to be of the same element family and same order.
#NOTE - Paraview: only available for version 5.12 and higher
time0 = time.time()
(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)
fides = FidesWriter(MPI.COMM_WORLD, "outputs/instationary-fides.bp", u)
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)
    fides.write(t)
fides.close()
time1 = time.time()

print("saving with FidesWriter takes ", time1-time0)
#!SECTION 

#NOTE - Results
# Sorted by lowest time taken
# 1.) FidesWriter
# 2.) XDMFWriter
# 3.) VTKWriter
# 4.) VTXWriter