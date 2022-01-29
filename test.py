import numpy as np
from Cube import *

cube = Cube(2)
state = np.array([cube.up_face, cube.down_face, cube.front_face,
                        cube.back_face, cube.right_face, cube.left_face], 
                        dtype = np.uint8)
print(np.rot90(state, k = 1, axes = (1,2)))
