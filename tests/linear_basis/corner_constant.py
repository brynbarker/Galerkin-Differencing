#!/usr/bin/env python3

from linear_basis.mac_grid.corner_classes import CornerRefineSolver
import numpy as np

output = []

u = lambda x,y,z: 1
f = lambda x,y,z: 0

# set up solvers
cref = CornerRefineSolver(16,u,f)
cref.laplace()

err = cref.error()
output.append(err < 1e-12)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

