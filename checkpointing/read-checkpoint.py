import warnings
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import GhostMode,create_unit_square, create_unit_cube, create_unit_interval, CellType
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
    
tgv = TGV()

filename = Path("checkpoints/checkpoint-tgv.bp")

domain = adios4dolfinx.read_mesh(MPI.COMM_WORLD, filename, engine = "BP4", ghost_mode = GhostMode.shared_facet)

element = ElementMetaData("Lagrange", 1, shape = (2,))
FS = functionspace(domain, element)
u = Function(FS)
u_read = Function(FS)

xdmf = XDMFFile(MPI.COMM_WORLD, 'checkpoints/checkpoint-tgv-read.xdmf', "w")
xdmf.write_mesh(domain)
for i in range(0,11,1):
    t=i/10
    tgv.t = t
    u.interpolate(tgv)

    adios4dolfinx.read_function(u_read, filename, engine = "BP4", time=t) 

    # Test if the functions are basically equal (i.e. if their defining arrays are)
    print("timestep: ",str(t))
    np.testing.assert_allclose(u.x.array, u_read.x.array, atol=1e-14)

    xdmf.write_function(u_read, t)
xdmf.close()
