#!/usr/bin/env python3

from linear_basis.mac_grid.vert_classes import VerticalRefineSolver
import numpy as np

output = []

u = lambda x,y,z: 1

# set up solvers
vref = VerticalRefineSolver(8,u)
vref.projection()

err = vref.error(proj=True)
output.append(err < 1e-12)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

