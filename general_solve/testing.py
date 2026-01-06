import numpy as np
import matplotlib.pyplot as plt
from paper_1.solver import Solver
from paper_1.simple_solve import LaplaceOperator, ProjectionOperator
from constraints import ConstraintOperator
from patch import Patch
from mesh import Mesh

# let's look at how dirichlet is being implemented
m = Mesh(16,2,'node','uniform')
cop = ConstraintOperator(m,dirichlet=True)

u = lambda x,y: 2
# cop._setup_dirichlet(u)

C = cop.spC
plt.matshow(C.todense())
plt.show()
#
#print(cop.dirichlet)
#for p in cop.patches:
#	for dof_id in p.dofs:
#		dof = p.dofs[dof_id]
#		m = '^' if cop.Id[dof_id] else 'o'
#		plt.plot(dof.x,dof.y,m)
#	plt.show()


u = lambda x,y: 1#np.sin(2*np.pi*x)
m = Solver(8,2,dofloc='node',rtype='uniform',u=u,dirichlet=True)