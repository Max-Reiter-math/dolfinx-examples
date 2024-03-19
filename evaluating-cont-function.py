from mpi4py import MPI
import numpy as np
from dolfinx import cpp, default_scalar_type
from dolfinx.mesh import create_unit_square, CellType
from dolfinx.io import VTXWriter
from dolfinx.fem import functionspace, Function
from dolfinx.geometry import compute_colliding_cells, bb_tree,compute_collisions_points
import ufl
from dolfinx.fem.petsc import LinearProblem

# dolfinx v0.7.3

"""
Evaluating a continuous function without making use of parallel computing.
Based on:
https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
For parallel computing see:
https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
https://fenicsproject.discourse.group/t/evaluating-finite-element-function/13462/10
"""

#SECTION - Preliminaries
def eval_continuous_function(f: Function,points: np.ndarray):
    print("Evaluating function at points ", points, " with shape ",points.shape)
    domain = f.function_space.mesh
    # Create a bounding box tree for use in collision detection. Padding seems to be some tolerance parameter.
    tree = bb_tree(domain, domain.topology.dim)
    print("Bounding Box Tree: ", tree, " with ", tree.num_bboxes, " boxes.")
    # Compute collisions between points and leaf bounding boxes. Bounding boxes can overlap, therefore points can collide with more than one box.
    cell_candidates = compute_collisions_points(tree, points.T)
    print("Cell candidates: ", cell_candidates)
    # From a mesh, find which cells collide with a set of points. 
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
    print("Colliding Cells: ", colliding_cells)
    #NOTE - For DG0 e.g. this is unique and it suffices to use cells = colliding_cells.array. 
    #NOTE - For CG nodal values are used on several local cells, so we have to choose one cell out of the colliding ones. However, the choice does not matter
    #NOTE - For DG1 and discontinuous methods of higher order, the evaluation at mesh nodes is ambiguous. This is why the cell for evaluation has to be chosen as well.
    # Choose one of the cells that contains the point
    cells = []
    for i, point in enumerate(points.T):
        print("For index ", i," and point ", point," we observe the colliding cells ", colliding_cells.links(i))
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
            print(" Appended cell ", cells[-1])
    # Now we can apply the eval function
    return f.eval(points.T,cells).T

def identity(x: np.ndarray):
    #NOTE - Array x has a third entry with z-components that are all equal to 0
    dim = 2
    values = np.zeros((dim, x.shape[1]))
    for i in range(dim):
        values[i] = x[i]
    return values 
#!SECTION

#SECTION - Initializing function
n = 2
domain = create_unit_square(MPI.COMM_WORLD, n, n, CellType.quadrilateral)
P1 = functionspace(domain, ("CG", 1, (2,)))
uh = Function(P1)
uh.interpolate(identity)
#!SECTION

# #SECTION - Save to VTX file
with VTXWriter(MPI.COMM_WORLD, "outputs/f-eval.bp", uh, engine="BP4") as vtx:
    vtx.write(0.0)
# #!SECTION

#SECTION - Testing the correctness of the evaluation
points = domain.geometry.x # this has shape (amount of nodes, 3)
# print(points.shape)
point_evaluation = eval_continuous_function(uh, points.T)

print("Checking correctness...")

points_in_2D = np.array([points.T[0],points.T[1]]).T # this has shape (num_points, 2)
# print(points_in_2D.shape)
print("Considering ", len(points_in_2D) ," points: ", points_in_2D)
print("with evaluations ", point_evaluation.T)
print("Correctness report: \n")
np.testing.assert_allclose(point_evaluation.T, points_in_2D)

#NOTE - This method is precise up to rtol = atol = 1e-15
