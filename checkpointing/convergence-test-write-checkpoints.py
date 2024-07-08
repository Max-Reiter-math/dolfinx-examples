import warnings
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import GhostMode, create_unit_square, create_unit_cube, create_unit_interval, CellType
from dolfinx.fem import ElementMetaData, functionspace, Function, assemble_scalar
from dolfinx.io import XDMFFile, VTKFile, VTXWriter, FidesWriter
from ufl import inner, dx
import adios4dolfinx

# dolfinx - 0.7.3
# adios4dolfinx - 0.7.3    

"""
Testing postprocessed error computation for nested meshes with checkpoints (1/2):
In this file we interpolate the Taylor-Green-Vortex to a sequence of hierarchical meshes. This is then written to checkpoints.
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


#SECTION - Preliminaries
mesh_sequence = [2,4,6,8]
family = "CG"
order = 1
dim = 2
element = ElementMetaData(family, order, shape = (dim,))
tgv = TGV() # Taylor-Green-Vortex callable
#!SECTION

#SECTION - Interpolation and saving checkpoints
for n in mesh_sequence:
    filename = Path("checkpoints/tgv-"+str(n)+".bp")

    domain = create_unit_square(MPI.COMM_WORLD, n,n, cell_type = CellType.triangle)
    FS = functionspace(domain, element)
    u = Function(FS)
    
    adios4dolfinx.write_mesh(domain, filename, engine = "BP4")

    for i in range(0,11,1):
        t=i/10
        tgv.t = t
        u.interpolate(tgv)
        adios4dolfinx.write_function(u, filename, engine = "BP4", time=t) 
    print("Finished checkpoint for n = ",str(n))
#!SECTION
        

