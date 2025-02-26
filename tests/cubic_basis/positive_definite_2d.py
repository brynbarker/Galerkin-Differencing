#!/usr/bin/env python3

from cubic_basis.mac_grid.horiz_classes_2d import HorizontalRefineSolver
from cubic_basis.mac_grid.vert_classes_2d import VerticalRefineSolver
from cubic_basis.mac_grid.corner_classes_2d import CornerRefineSolver
import numpy as np

output = []

u = lambda x,y: 1
f = lambda x,y: 0

# set up solvers
href = HorizontalRefineSolver(32,u,f)
vref = VerticalRefineSolver(32,u,f)
cref = CornerRefineSolver(32,u,f)

href.projection()
vref.projection()
cref.projection()

hm = href.C.T@href.M@href.C
evals = np.linalg.eig(hm)[0]
output.append( max(evals)*min(evals) > 0)

vm = vref.C.T@vref.M@vref.C
evals = np.linalg.eig(vm)[0]
output.append( max(evals)*min(evals) > 0)

cm = cref.C.T@cref.M@cref.C
evals = np.linalg.eig(cm)[0]
output.append( max(evals)*min(evals) > 0)

href.laplace()
vref.laplace()
cref.laplace()

hk = href.C.T@href.K@href.C
evals = np.linalg.eig(hk)[0]
output.append( max(evals)*min(evals) > 0)

vk = vref.C.T@vref.K@vref.C
evals = np.linalg.eig(vk)[0]
output.append( max(evals)*min(evals) > 0)

ck = cref.C.T@cref.K@cref.C
evals = np.linalg.eig(ck)[0]
output.append( max(evals)*min(evals) > 0)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

