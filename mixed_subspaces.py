import warnings
import numpy as np
from mpi4py import MPI
from ufl import MixedElement, FiniteElement, VectorElement
from dolfinx.mesh import create_unit_square, create_unit_cube, create_unit_interval, CellType
from dolfinx.fem import ElementMetaData, functionspace, Function
from dolfinx.io import XDMFFile, VTKFile, VTXWriter, FidesWriter

# dolfinx version v0.7.3

"""
In this file the correct use of the different output options of dolfinx with respect to mixed FEM spaces is explored.
"""

class TGV:
    """
    class for the velocity field of a standard taylor green vortex
    """
    def __init__(self, t =0.0, nu = 0.1):
        self.t=t
        self.nu=nu
    def __call__(self, x: np.ndarray):
        values = np.zeros((3, x.shape[1]))
        values[0] = -np.cos(x[0])*np.sin(x[1])*np.exp(-2.0*self.nu*self.t)
        values[1] = np.sin(x[0])*np.cos(x[1])*np.exp(-2.0*self.nu*self.t)
        values[2] = -0.25*(np.cos(2*x[0] )+np.cos(2*x[1]) )*np.exp(-4.0*self.nu*self.t)
        return values
    def velocity(self, x: np.ndarray):
        values = np.zeros((2, x.shape[1]))
        values[0] = -np.cos(x[0])*np.sin(x[1])*np.exp(-2.0*self.nu*self.t)
        values[1] = np.sin(x[0])*np.cos(x[1])*np.exp(-2.0*self.nu*self.t)
        return values
    def pressure(self, x: np.ndarray):
        values = np.zeros((1, x.shape[1]))
        values[0] = -0.25*(np.cos(2*x[0] )+np.cos(2*x[1]) )*np.exp(-4.0*self.nu*self.t)
        return values

#ANCHOR - Parameters
n = 4
order = 1

domain = create_unit_square(MPI.COMM_WORLD, n,n, cell_type = CellType.triangle)

#ANCHOR - Mixed FEM Space
vP         = VectorElement("Lagrange", domain.ufl_cell(), order, dim = 2)
P1          = FiniteElement("Lagrange", domain.ufl_cell(), 1)
element     = MixedElement(vP, P1) 
FS = functionspace(domain, element)
u = Function(FS)

#ANCHOR - Subspaces
V, mapV = FS.sub(0).collapse()
Q, mapQ = FS.sub(1).collapse()

#ANCHOR - Initializing function (Taylor Green Vortex) to compare outputs
tgv = TGV(t=0.0)
# u.interpolate(tgv) # This does not work for a mixed space
u.sub(0).interpolate(tgv.velocity)
u.sub(1).interpolate(tgv.pressure)


#SECTION - XDMF
#NOTE - XDMF only supports Lagrange Elements of order <=1
# """
with XDMFFile(MPI.COMM_WORLD, 'outputs/output_stationary.xdmf', "w") as xdmf:
    xdmf.write_mesh(domain)
    # xdmf.write_function(u) # PETSC ERROR
    xdmf.write_function(u.sub(0)) # this throws an error for order >= 2
    xdmf.write_function(u.sub(1))
# """
#!SECTION


#SECTION - VTK
#NOTE - VTK supports arbitrary order Lagrange finite elements for the geometry description. XDMF is the preferred format for geometry order <= 2.
"""
with VTKFile(MPI.COMM_WORLD, "outputs/stationary.pvd", "w") as vtk:
    vtk.write_mesh(domain)
    # vtk.write_function(u) # PETSC ERROR     
    # vtk.write_function(u.sub(0)) # RuntimeError: Cannot write sub-Functions to VTK file.
    # vtk.write_function([u.sub(0).collapse(),u.sub(1).collapse()]) # only works if both functions have the same order
    vtk.write_function(u.sub(0).collapse()) # works    
"""
#!SECTION


    
#SECTION - VTX
#NOTE - VTX supports arbitrary order Lagrange finite elements for the geometry description and arbitrary order (discontinuous) Lagrange finite elements for Functions. All Functions for output must share the same mesh and have the same element type.
#NOTE - Paraview: only available for version 5.12 and higher
"""
# with VTXWriter(MPI.COMM_WORLD, "outputs/stationary.bp", u, engine="BP4") as vtx: #RuntimeError: Mixed functions are not supported by VTXWriter.
# with VTXWriter(MPI.COMM_WORLD, "outputs/stationary.bp", u.sub(0), engine="BP4") as vtx: # Error: munmap_chunk(): invalid pointer
# with VTXWriter(MPI.COMM_WORLD, "outputs/stationary.bp", [u.sub(0).collapse(), u.sub(1).collapse()], engine="BP4") as vtx: # only works if both functions have the same order
with VTXWriter(MPI.COMM_WORLD, "outputs/stationary.bp", u.sub(0).collapse(), engine="BP4") as vtx: # works
    vtx.write(0.0)
"""
#!SECTION


#SECTION - Fides
#NOTE - Fides (https://fides.readthedocs.io/) supports first order Lagrange finite elements for the geometry description and first order Lagrange finite elements for functions. All functions have to be of the same element family and same order.
#NOTE - Paraview: Currently not available for windows installer.
"""
# with FidesWriter(MPI.COMM_WORLD, "outputs/stationary-fides.bp", u, engine="BP4") as fides: # RuntimeError: Mixed functions are not supported by FidesWriter
# with FidesWriter(MPI.COMM_WORLD, "outputs/stationary-fides.bp", u.sub(0), engine="BP4") as fides: # PETSC ERROR 
# with FidesWriter(MPI.COMM_WORLD, "outputs/stationary-fides.bp", [u.sub(0).collapse(),u.sub(1).collapse()], engine="BP4") as fides: # only works for order <= 1 and when all functions have the same basis elements
with FidesWriter(MPI.COMM_WORLD, "outputs/stationary-fides.bp", u.sub(0).collapse(), engine="BP4") as fides: # only works for order <= 1
    fides.write(0.0)
"""
#!SECTION 

