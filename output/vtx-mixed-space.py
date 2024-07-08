import warnings
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, create_unit_cube, create_unit_interval, CellType
from dolfinx.fem import ElementMetaData, functionspace, Function
from dolfinx.io import XDMFFile, VTKFile, VTXWriter, FidesWriter
from ufl import FiniteElement, VectorElement, MixedElement
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
    def velocity(self, x: np.ndarray):
        values = np.zeros((2, x.shape[1]))
        values[0] = -np.cos(x[0])*np.sin(x[1])*np.exp(-2.0*self.nu*self.t)
        values[1] = np.sin(x[0])*np.cos(x[1])*np.exp(-2.0*self.nu*self.t)
        # values[2] = -0.25*(np.cos(2*x[0])+np.cos(2*x[1]))*np.exp(-4.0*self.nu*self.t)
        return values
    def pressure(self, x:np.ndarray):
        values = np.zeros((1, x.shape[1]))
        values[0] = -0.25*(np.cos(2*x[0])+np.cos(2*x[1]))*np.exp(-4.0*self.nu*self.t)
        return values

dim =2
n = 4 
domain = create_unit_square(MPI.COMM_WORLD, n,n, cell_type = CellType.triangle)
P2          = VectorElement("Lagrange", domain.ufl_cell(), 2, dim = dim)
P1          = FiniteElement("Lagrange", domain.ufl_cell(), 1)
me          = MixedElement(P2, P1)
TH          = functionspace(domain, me)    
V, mapV = TH.sub(0).collapse()
Q, mapQ = TH.sub(1).collapse()

u = Function(TH)

tgv = TGV()

v_out = Function(V)
    
# #SECTION - Instationary case, stationary mesh
vtx = VTXWriter(MPI.COMM_WORLD, "outputs/tgv-mixed-VTX.bp", v_out, engine="BP4")
for i in range(0,10,1):
    t=i/10
    tgv.t = t
    u.sub(0).interpolate(tgv.velocity)
    u.sub(1).interpolate(tgv.pressure)

    
    # Different options. Lets see what works
    v_out.interpolate(u.sub(0))
    # v_out.interpolate(u.sub(0).collapse())
    # v_split, p_split = u.split()
    # v_out.interpolate(v_split.collapse())
    print("interpolation successful")
    vtx.write(t)
    print("writing successful")
vtx.close()
#!SECTION  