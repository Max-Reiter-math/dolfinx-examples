import warnings
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import CellType, create_unit_cube, create_unit_interval, create_unit_square
from dolfinx.io import XDMFFile
from dolfinx.fem import functionspace, ElementMetaData, Function


# dolfinx v 0.7.0

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

#NOTE - For the XDMFFile format, functions are interpolated onto the mesh nodes, as a Nth order Lagrange function, where N is the order of the coordinate map. If the Function is a cell-wise constant, it is saved as a cell-wise constant.

#SECTION - Lagrange functions
#SECTION - topological dim = 1
# (u,domain) = get_function(1, CellType.interval, 3, "Lagrange", order = 1) # Success
# (u,domain) = get_function(1, CellType.interval, 3, "Lagrange", order = 2) # RuntimeError: Degree of output Function must be same as mesh degree. Maybe the Function needs to be interpolated?
#!SECTION
#SECTION - topological dim = 2
(u,domain) = get_function(2, CellType.triangle, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(2, CellType.triangle, 3, "Lagrange", 2) # RuntimeError: Degree of output Function must be same as mesh degree. Maybe the Function needs to be interpolated?
# (u,domain) = get_function(2, CellType.quadrilateral, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(2, CellType.quadrilateral, 3, "Lagrange", 2) # RuntimeError: Degree of output Function must be same as mesh degree. Maybe the Function needs to be interpolated?
#!SECTION
#SECTION - topological dim = 3
#SECTION - Tetrahedra
# (u,domain) = get_function(3, CellType.tetrahedron, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(3, CellType.tetrahedron, 3, "Lagrange", 2) # RuntimeError: Degree of output Function must be same as mesh degree. Maybe the Function needs to be interpolated?
#!SECTION
#SECTION - Pyramids
# (u,domain) = get_function(3, CellType.pyramid, 3, "Lagrange", 1) # RuntimeError: Non-equispaced points on pyramids not supported yet.
#!SECTION
#SECTION - Prisms
# (u,domain) = get_function(3, CellType.prism, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(3, CellType.prism, 3, "Lagrange", 2) # RuntimeError: Elements with different numbers of DOFs on subentities of the same dimension are not yet supported in FFCx.
#SECTION - Hexahedra
# (u,domain) = get_function(3, CellType.hexahedron, 3, "Lagrange", 1) # Success
# (u,domain) = get_function(3, CellType.hexahedron, 3, "Lagrange", 2) # RuntimeError: Degree of output Function must be same as mesh degree. Maybe the Function needs to be interpolated?
#!SECTION
#!SECTION

#SECTION - Other Function Spaces, see e.g. https://docs.fenicsproject.org/ufl/2022.2.0/manual/form_language.html?highlight=lagrange#element-families
# (u,domain) = get_function(2, CellType.triangle, 3, "DG", 1) # RuntimeError: Function and Mesh dof layouts do not match. Maybe the Function needs to be interpolated?
# (u,domain) = get_function(3, CellType.tetrahedron, 3, "DG", 1) # RuntimeError: Function and Mesh dof layouts do not match. Maybe the Function needs to be interpolated?
# (u,domain) = get_function(2, CellType.triangle, 3, "CG", 1) # Success, since CG = Lagrange
# (u,domain) = get_function(2, CellType.triangle, 3, "CR", 1) # RuntimeError: Function and Mesh dof layouts do not match. Maybe the Function needs to be interpolated?
#!SECTION

with XDMFFile(MPI.COMM_WORLD, 'outputs/output1.xdmf', "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)
print("Success")

#NOTE - XDMF Files only do well with Lagrange spaces of first order

