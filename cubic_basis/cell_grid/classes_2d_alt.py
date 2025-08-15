import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy import sparse

from cubic_basis.cell_grid.helpers_2d import *
from cubic_basis.cell_grid.shape_functions_2d import phi3_dxx

keys = [None,0,1]
id = 0
LOOKUP = {}
for xk in keys:
    temp = {}
    for yk in keys:
        temp[yk] = id
        id += 1
    LOOKUP[xk] = temp

from cubic_basis.cell_grid.classes_2d import Laplace, Projection

class LaplaceAlt(Laplace):
	def __init__(self,N,u,f,qpn=5):
		super().__init__(N,u,f,qpn)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)
		Cr, Cc, Cd = [],[],[]

		for j in range(2):
			c_side = np.array(self.mesh.interface_offset[j][:4])
			f_side = np.array(self.mesh.interface_offset[j][4:])

			v12,v32 = 9/16,-1/16
			v14,v34,v54,v74 = 105/128,35/128,-7/128,-5/128

			self.Id[f_side[1]] = 1
			self.C_full[f_side[1]] *= 0
			for ind,vhorz in enumerate([v32,v12,v12,v32]):
				for vvert, offset in zip([v74,v34,v14,v54],[2,1,0,-1]):
					self.C_full[f_side[1][::2],np.roll(c_side[ind],offset)] = vvert*vhorz/v12
					Cr += list((f_side[1][::2]).flatten())
					Cc += list(np.roll(c_side[ind],offset).flatten())
					Cd += [vvert*vhorz/v12]*(f_side[1][::2]).size
				for vvert, offset in zip([v54,v14,v34,v74],[1,0,-1,-2]):
					self.C_full[f_side[1][1::2],np.roll(c_side[ind],offset)] = vvert*vhorz/v12
					Cr += list((f_side[1][1::2]).flatten())
					Cc += list(np.roll(c_side[ind],offset).flatten())
					Cd += [vvert*vhorz/v12]*(f_side[1][1::2]).size
			for ind,vhorz in zip([0,2,3],[v32,v12,v32]):
				self.C_full[f_side[1],f_side[ind]] = -vhorz/v12
				Cr += list((f_side[1]).flatten())
				Cc += list((f_side[ind]).flatten())
				Cd += [-vhorz/v12]*(f_side[1]).size


		for level in range(2):
			# lower are true dofs, upper are ghosts
			b0,b1,B2,B3,T0,T1,t2,t3 = np.array(self.mesh.periodic[level]).reshape((8,-1))
			ghost_list = np.array([b0,b1,t2,t3])
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds, ds = [T0,T1,B2,B3],[b0,b1,t2,t3]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [0,1,2,3,-4,-3,-2,-1]:
					self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					self.C_full[d[ind],:] = self.C_full[D[ind],:]
			
		for dof_id in self.mesh.boundaries:
			Cr,Cc,Cd = inddel(Cr,Cc,Cd,dof_id)
			assert dof_id not in Cr
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]
		return

	def vis_constraints(self):
		return super().vis_constraints(alt=True)