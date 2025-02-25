import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from 2D.cubic_basis.mac_grid.classes import Node, Element, Mesh, Solver
from 2D.cubic_basis.mac_grid.helpers import vis_constraints
from 2D.cubic_basis.mac_grid.shape_functions import phi3


class HorizontalRefineMesh(Mesh):
	def __init__(self,N):
		super().__init__(N)

	def _make_coarse(self): # overwritten
		self.interface[0] = [[],[]]
		self.periodic[0] = []
		H = self.h*2
		xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
		ydom = np.linspace(0-3*H/2,1+3*H/2,self.N+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0<=x<.5) and (-H<y<1.):###
					strt = dof_id-1-xlen
					element = Element(e_id,j-1,i-1,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1
				
				if x==H:
					self.boundaries.append(dof_id)

				if y < 2*H or y > 1-2*H:
					self.periodic[0].append(dof_id)
				
				if (x==.5 or x==0):# and (0 < y):
					self.interface[0][x==.5].append(dof_id)

				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self): # overwritten
		self.interface[1] = [[],[]]
		self.periodic[1] = []
		H = self.h
		xdom = np.linspace(0.5-H,1.+H,self.N+3)
		ydom = np.linspace(0-3*H/2,1+3*H/2,2*self.N+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0.5<=x<1.) and (-H<y<1.):####
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if y < 2*H or y > 1.-2*H:
					self.periodic[1].append(dof_id)
				if (x == 0.5 or x==1.):# and (0 < y):
					self.interface[1][x==.5].append(dof_id)

				dof_id += 1

class HorizontalRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=HorizontalRefineMesh)

	def _setup_constraints(self): # overwritten
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)


		c_inter = self.mesh.interface[0]
		f_inter = self.mesh.interface[1]
		for j in range(2):
			self.Id[f_inter[j]] = 1
			self.C_full[f_inter[j]] *= 0

		v1, v3, v5, v7 = phi3(1/4,1), phi3(3/4,1), phi3(5/4,1), phi3(7/4,1)
		
		for j in range(2):
			self.C_full[f_inter[j][1:-1:2],c_inter[j][:-3]] = v5
			self.C_full[f_inter[j][1:-1:2],c_inter[j][1:-2]] = v1
			self.C_full[f_inter[j][1:-1:2],c_inter[j][2:-1]] = v3
			self.C_full[f_inter[j][1:-1:2],c_inter[j][3:]] = v7

			self.C_full[f_inter[j][2::2],c_inter[j][:-3]] = v7
			self.C_full[f_inter[j][2::2],c_inter[j][1:-2]] = v3
			self.C_full[f_inter[j][2::2],c_inter[j][2:-1]] = v1
			self.C_full[f_inter[j][2::2],c_inter[j][3:]] = v5

		for level in range(2):
			b0,b1,B2,B3,T0,T1,t2,t3 = np.array(self.mesh.periodic[level]).reshape((8,-1))
			ghost_list = np.hstack((b0,b1,t2,t3))
			self.mesh.periodic_ghost.append(ghost_list)
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds,ds = [T0,T1,B2,B3],[b0,b1,t2,t3]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [1,-2]:
					if level == 0:
						self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					if level ==1:
						self.C_full[d[ind],:] = self.C_full[D[ind],:]

		self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self,retfig=False):
		fig = vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[1],'horiz')
		if retfig: return fig

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('horiz')
		if retfig: return fig

	def xy_to_e(self,x,y): # over_written
		n_x_els = [self.N/2,self.N]
		
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		fine = True if x >= 0.5 else False
		x_ind = int((x-fine*.5)/((2-fine)*self.h))
		y_ind = int(y/((2-fine)*self.h)+.5)
		el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		return e


