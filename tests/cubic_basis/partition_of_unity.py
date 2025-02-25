#!/usr/bin/env python3

from 2D.cubic_basis.mac_grid.horiz_classes import HorizontalRefineSolver
from 2D.cubic_basis.mac_grid.vert_classes import VerticalRefineSolver
from 2D.cubic_basis.mac_grid.corner_classes import CornerRefineSolver
import numpy as np

output = []

u = lambda x,y: 1

# set up solvers
href = HorizontalRefineSolver(16,u)
vref = VerticalRefineSolver(16,u)
cref = CornerRefineSolver(16,u)

# setup solvers for pou
hsol = href.sol(weights = np.ones((len(href.mesh.dofs))))
vsol = vref.sol(weights = np.ones((len(vref.mesh.dofs))))
csol = cref.sol(weights = np.ones((len(cref.mesh.dofs))))


dom = np.linspace(0,1)
hvals,vvals,cvals = [],[],[]
for x in dom:
	for y in dom:
		hvals.append(abs(hsol(x,y)-1))
		vvals.append(abs(vsol(x,y)-1))
		cvals.append(abs(csol(x,y)-1))

# compute error
herr = np.linalg.norm(np.array(hvals))
verr = np.linalg.norm(np.array(vvals))
cerr = np.linalg.norm(np.array(cvals))

output.append(herr < 1e-12)
output.append(verr < 1e-12)
output.append(cerr < 1e-12)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

