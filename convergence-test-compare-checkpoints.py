import warnings
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import GhostMode, create_unit_square, create_unit_cube, create_unit_interval, CellType
from dolfinx.fem import ElementMetaData, functionspace, Function, assemble_scalar, form, create_nonmatching_meshes_interpolation_data
from dolfinx.io import XDMFFile, VTKFile, VTXWriter, FidesWriter
from ufl import inner, dx
import adios4dolfinx

# dolfinx - 0.7.3
# adios4dolfinx - 0.7.3    

"""
Testing postprocessed error computation for nested meshes with checkpoints (2/2):
In this file we read the Taylor-Green-Vortex on a sequence of hierarchical meshes from checkpoints. Afterwards the errors are computed.
"""

def L2error(u1, u2):
    e = u1-u2
    var_form = inner(e,e)*dx
    return assemble_scalar(form(var_form))



#SECTION - Preliminaries
mesh_sequence = [2,4,6,8]
family = "CG"
order = 1
dim = 2
element = ElementMetaData(family, order, shape = (dim,))
#!SECTION
        
#SECTION - Reading checkpoints and error computation
#SECTION - Importing reference solution
n_ref = mesh_sequence[-1]
filename_ref = Path("checkpoints/tgv-"+str(n_ref)+".bp")

domain_ref = adios4dolfinx.read_mesh(MPI.COMM_WORLD, filename_ref, engine = "BP4", ghost_mode = GhostMode.shared_facet)
FS_ref = functionspace(domain_ref, element)
u_ref = Function(FS_ref)
#NOTE - Since the mesh sequence is hierarchical we have to interpolate the approximate solutions on the coarser meshes onto the FEM space on the finer meshes
u_approx = Function(FS_ref)
#!SECTION

for i in range(0,11,1):
    t=i/10
    print("---")
    print("time step: ", str(t))
    for n in mesh_sequence[:-1]:
        filename_approx = Path("checkpoints/tgv-"+str(n)+".bp")

        domain_approx = adios4dolfinx.read_mesh(MPI.COMM_WORLD, filename_approx, engine = "BP4", ghost_mode = GhostMode.shared_facet)
        FS_approx = functionspace(domain_approx, element)
        u_in = Function(FS_approx)

        print("comparison of ",str(n_ref)," and ", str(n))
        adios4dolfinx.read_function(u_ref, filename_ref, engine = "BP4", time=t) # reading reference solution
        adios4dolfinx.read_function(u_in, filename_approx, engine = "BP4", time=t) # reading approximate solution
        #NOTE - Since the mesh sequence is hierarchical we have to interpolate the approximate solutions on the coarser meshes onto the FEM space on the finer meshes
        data = create_nonmatching_meshes_interpolation_data(u_approx.function_space.mesh._cpp_object, u_approx.function_space.element, u_in.function_space.mesh._cpp_object, 0.0) # padding seems to be some tolerance variable for the collision detection. So we set it to 0.0 on standard
        u_approx.interpolate(u_in, nmm_interpolation_data=data)
        
        #SECTION - Error computation
        err = L2error(u_ref, u_approx)        
        print("error: ", err)
        #!SECTION

#!SECTION

