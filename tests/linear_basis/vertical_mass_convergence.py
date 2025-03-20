#!/usr/bin/env python3

from linear_basis.mac_grid.horiz_classes import VerticalRefineSolver
import numpy as np

output = []

u = lambda x,y,z: np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*np.sin(2*np.pi*z)

# set up solvers
prev = 1
for N in [8,16]:
	vref = VerticalRefineSolver(N,u)
	vref.projection()

	err = vref.error(proj=True)
	if prev != 1:
		output.append(prev/4 > err)
	prev = err

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

