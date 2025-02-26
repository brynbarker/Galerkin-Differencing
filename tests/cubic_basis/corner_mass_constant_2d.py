#!/usr/bin/env python3

from cubic_basis.mac_grid.corner_classes_2d import CornerRefineSolver
import numpy as np

output = []

u = lambda x,y: 1

# set up solvers
cref = CornerRefineSolver(16,u)
cref.projection()

err = cref.error(proj=True)
output.append(err < 2e-12)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

