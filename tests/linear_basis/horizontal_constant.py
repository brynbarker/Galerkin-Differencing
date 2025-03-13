#!/usr/bin/env python3

from linear_basis.mac_grid.horiz_classes import HorizontalRefineSolver
import numpy as np

output = []

u = lambda x,y,z: 1
f = lambda x,y,z: 0

# set up solvers
href = HorizontalRefineSolver(8,u,f)
href.laplace()

err = href.error()
output.append(err < 1e-12)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

