#!/usr/bin/env python3

from 2D.linear_basis.mac_grid.corner_refine.corner_classes import CornerRefineSolver
import numpy as np

output = []

u = lambda x,y: np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

# set up solvers
prev = 1
for N in [8,16,32,64]:
	cref = CornerRefineSolver(N,u)
	cref.projection()

	err = cref.error(proj=True)
	if prev != 1:
		output.append(prev/4 > err)
	prev = err

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

