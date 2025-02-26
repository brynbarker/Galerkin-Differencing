#!/usr/bin/env python3

from linear_basis.mac_grid.horiz_classes_2d import HorizontalRefineSolver
import numpy as np

output = []

u = lambda x,y: 1

# set up solvers
href = HorizontalRefineSolver(16,u)
href.projection()

err = href.error(proj=True)
output.append(err < 1e-12)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

