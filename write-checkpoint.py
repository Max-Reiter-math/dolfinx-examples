import warnings
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, create_unit_cube, create_unit_interval, CellType
from dolfinx.fem import ElementMetaData, functionspace, Function
from dolfinx.io import XDMFFile, VTKFile, VTXWriter, FidesWriter
import adios4dolfinx

# dolfinx - 0.7.3
# adios4dolfinx - 0.7.3    

"""
Checkpointing examples with dolfinx adios4dolfinx.
"""

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


filename = Path("checkpoints/checkpoint-tgv.bp")

tgv = TGV()

(u,domain) = get_function(2, CellType.triangle, 2, "Lagrange", 1)

adios4dolfinx.write_mesh(domain, filename, engine = "BP4")

xdmf = XDMFFile(MPI.COMM_WORLD, 'checkpoints/checkpoint-tgv-original.xdmf', "w")
xdmf.write_mesh(domain)
for i in range(0,11,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)

    adios4dolfinx.write_function(u, filename, engine = "BP4", time=t) # This cannot be opened with the standard paraview AFAIK

    xdmf.write_function(u, t)
xdmf.close()
