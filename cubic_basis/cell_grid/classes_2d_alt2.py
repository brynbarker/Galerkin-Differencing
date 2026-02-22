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

from cubic_basis.cell_grid.classes_2d import Laplace

class LaplaceAlt2(Laplace):
	def __init__(self,N,u,f,qpn=5,alt=1):
		super().__init__(N,u,f,qpn)
		self.alt = alt

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		for j in range(2):
			c_side = np.array(self.mesh.interface_offset[j][:4])
			f_side = np.array(self.mesh.interface_offset[j][4:])

			v12,v32 = 9/16,-1/16
			v14,v34,v54,v74 = 105/128,35/128,-7/128,-5/128

			f_len, c_len = len(f_side[0]), len(c_side[-1])
			sys_len = f_len+c_len
			sys = np.eye(sys_len) ## sys section A and D
			rhs_len = 3*sys_len
			rhs = np.zeros((sys_len,rhs_len))

			for c_col,vhorz in enumerate([v32,v12,v12]):
				for f_row in range(f_len):
					c_row = int(f_row/2)
					if f_row%2:
						rhs[c_row,c_col*c_len+c_row] = -vhorz/v32 ## rhs section A
						vverts, offsets = [v74,v34,v14,v54],[2,1,0,-1]
						f_row_shifted = f_row-2
						if f_row_shifted < 0: f_row_shifted += f_len
					else:
						vverts, offsets = [v54,v14,v34,v74],[1,0,-1,-2]
						f_row_shifted = f_row+2
						if f_row_shifted >= f_len: f_row_shifted -= f_len

					sys[c_col,c_len+f_row] = -v12 ## sys section B
					sys[c_col,c_len+f_row_shifted] = -v32 ## sys section B
					
					for vvert, offset in zip(vverts,offsets):
						c_row_shifted = c_row-offset
						if c_row_shifted < 0: c_row_shifted+=c_len
						if c_row_shifted >= c_len: c_row_shifted-=c_len
						rhs[c_len+f_row,c_col*c_len+c_row_shifted] = vvert*vhorz/v32 ## rhs section C
						if c_col == 0:
							sys[c_len+f_row,c_row_shifted] = -vvert ## sys section C

			for f_col,vhorz in enumerate([v12,v12,v32]):
				for f_row in range(f_len):
					rhs[c_len+f_row,3*c_len+f_col*f_len+f_row] = -vhorz/v32 ## rhs section D
					if f_row%2:
						f_row_shifted = f_row-2
						if f_row_shifted < 0: f_row_shifted += f_len
					else:
						f_row_shifted = f_row+2
						if f_row_shifted >= f_len: f_row_shifted -= f_len
					c_row = int(f_row/2)
					rhs[c_row,3*c_len+f_col*f_len+f_row] = vhorz*v12/v32 ## rhs section B
					rhs[c_row,3*c_len+f_col*f_len+f_row_shifted] = vhorz ## rhs section B
					
			self.sys = sys
			self.rhs = rhs
					
			self.Id[f_side[0]] = 1
			self.Id[c_side[-1]] = 1
			self.C_full[f_side[0]] *= 0
			self.C_full[c_side[-1]] *= 0

			coefs = la.inv(sys) @ rhs
			self.coefs = coefs
			rhs_index_to_id = []
			for i in range(3):
				rhs_index_to_id += [c_ind for c_ind in c_side[i]]
			for i in range(3):
				rhs_index_to_id += [f_ind for f_ind in f_side[i+1]]

			for c_row,c_ind in enumerate(c_side[-1]):
				self.C_full[c_ind,rhs_index_to_id] += coefs[c_row]

			for f_row,f_ind in enumerate(f_side[0]):
				self.C_full[f_ind,rhs_index_to_id] += coefs[c_len+f_row]

		for level in range(2):
			# lower are true dofs, upper are ghosts
			b0,b1,B2,B3,T0,T1,t2,t3 = np.array(self.mesh.periodic[level]).reshape((8,-1))
			ghost_list = np.array([b0,b1,t2,t3])
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds, ds = [T0,T1,B2,B3],[b0,b1,t2,t3]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				self.C_full[:,D] += self.C_full[:,d]
				self.C_full[d,:] = self.C_full[D,:]
				# for ind in [0,1,2,3,-4,-3,-2,-1]:
				# 	self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
				# 	self.C_full[d[ind],:] = self.C_full[D[ind],:]
			
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]
		return

	def vis_constraints(self):
		return super().vis_constraints(alt=True)