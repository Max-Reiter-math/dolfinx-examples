import warnings
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import CellType, create_unit_cube, create_unit_interval, create_unit_square
from dolfinx.io import VTXWriter
from dolfinx.fem import functionspace, ElementMetaData, Function

# dolfinx version v0.7.3

"""
In this file we consider examples of FEM spaces which work and do not wirk the VTX File format.
"""

#NOTE - We call the last input order on purpose because it depends on the CellType whether it is equivalent to the polynomial degree. For more see: https://defelement.com/

def get_function(tdim: int, celltype: CellType, dim: int, family: str, order: int):
    n = 2
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


#SECTION - Lagrange functions
#SECTION - topological dim = 1
# (u,domain) = get_function(1, CellType.interval, 3, "Lagrange", order = 1) # Success
# (u,domain) = get_function(1, CellType.interval, 3, "Lagrange", order = 2) # Success
#!SECTION
#SECTION - topological dim = 2
# (u,domain) = get_function(2, CellType.triangle, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(2, CellType.triangle, 3, "Lagrange", 2) # Success
# (u,domain) = get_function(2, CellType.quadrilateral, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(2, CellType.quadrilateral, 3, "Lagrange", 2) # Success
#!SECTION
#SECTION - topological dim = 3
#SECTION - Tetrahedra
# (u,domain) = get_function(3, CellType.tetrahedron, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(3, CellType.tetrahedron, 3, "Lagrange", 2) # Success
#!SECTION
#SECTION - Pyramids
# (u,domain) = get_function(3, CellType.pyramid, 3, "Lagrange", 1) # RuntimeError: Non-equispaced points on pyramids not supported yet.
#!SECTION
#SECTION - Prisms
# (u,domain) = get_function(3, CellType.prism, 3, "Lagrange", 1) # RuntimeError: Unknown cell type
# (u,domain) = get_function(3, CellType.prism, 3, "Lagrange", 2) # RuntimeError: Elements with different numbers of DOFs on subentities of the same dimension are not yet supported in FFCx.
#SECTION - Hexahedra
# (u,domain) = get_function(3, CellType.hexahedron, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(3, CellType.hexahedron, 3, "Lagrange", 2) # Success
#!SECTION
#!SECTION

#SECTION - Other Function Spaces, see e.g. https://docs.fenicsproject.org/ufl/2022.2.0/manual/form_language.html?highlight=lagrange#element-families
#TODO - Test DG0
# (u,domain) = get_function(2, CellType.triangle, 3, "DG", 0) # RuntimeError: VTK does not support cell-wise fields. See https://gitlab.kitware.com/vtk/vtk/-/issues/18458.
#NOTE - For DG0 instead compute in DG0 and interpolate in DG1 before saving
(u,domain) = get_function(2, CellType.triangle, 3, "DG", 1) # Success
# (u,domain) = get_function(3, CellType.tetrahedron, 3, "DG", 1) # Success
# (u,domain) = get_function(2, CellType.triangle, 3, "CG", 1) # Success
# (u,domain) = get_function(2, CellType.triangle, 3, "CR", 1) # Success
#!SECTION

with VTXWriter(MPI.COMM_WORLD, "outputs/VTX-test.bp", u, engine="BP4") as vtx:
    vtx.write(0.0)
print("Success")

#NOTE - VTX Files are very flexible with respect to the FEM space.

