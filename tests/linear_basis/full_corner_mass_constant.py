#!/usr/bin/env python3

from linear_basis.mac_grid.full_corner_classes import FullCornerRefineSolver
import numpy as np

output = []

u = lambda x,y,z: 1

# set up solvers
cref = FullCornerRefineSolver(16,u)
cref.projection()

err = cref.error(proj=True)
output.append(err < 2e-12)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

