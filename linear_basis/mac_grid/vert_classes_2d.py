import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from linear_basis.mac_grid.classes_2d import Node, Element, Mesh, Solver
from linear_basis.mac_grid.helpers_2d import vis_constraints


class VerticalRefineMesh(Mesh):
	def __init__(self,N):
		self.interface_f_dofs = [[],[]]
		super().__init__(N)

	def _make_coarse(self):
		self.interface[0] = [[],[]]
		self.periodic[0] = []
		H = self.h*2
		xdom = np.linspace(0,1,self.N+1)
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			interface_element = (i==0 or i==ylen-2)
			side = i==0
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (y<.5) and (x<1.):
					strt = dof_id
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if y==3*H/2:
					self.boundaries.append(dof_id)
				if x < H or x > 1.-H:
					self.periodic[0].append(dof_id)
				if (y > 0.5-H or y<H):
					self.interface[0][y>H].append(dof_id)

				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		self.interface[1] = [[],[]]
		self.periodic[1] = []
		H = self.h
		xdom = np.linspace(0,1,2*self.N+1)
		ydom = np.linspace(0.5-H/2,1.+H/2,self.N+2)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			interface_element = (i==0 or i==ylen-2)
			side = (i==0)
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (y<1.) and (x<1.):
					strt = dof_id
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if x < H or x > 1.-H:
					self.periodic[1].append(dof_id)
				if (y < 0.5 or y>1):
					self.interface[1][y<1].append(dof_id)
				if (i==1 or i==ylen-2):
					self.interface_f_dofs[i==1].append(dof_id)
 

				dof_id += 1

class VerticalRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=VerticalRefineMesh)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter = self.mesh.interface[0]
		f_inter = self.mesh.interface[1]
		f_dofs = self.mesh.interface_f_dofs
		for j in range(2):
			self.Id[f_inter[j]] = 1
			self.C_full[f_inter[j]] *= 0

		for j in range(2):
			c_a, c_b = np.array(c_inter[j]).reshape((2,-1))
			self.C_full[f_inter[j][::2],c_a] = 1
			self.C_full[f_inter[j][::2],c_b] = 1

			self.C_full[f_inter[j][1::2],c_a[:-1]] = 1/2
			self.C_full[f_inter[j][1::2],c_b[:-1]] = 1/2
			self.C_full[f_inter[j][1::2],c_a[1:]] = 1/2
			self.C_full[f_inter[j][1::2],c_b[1:]] = 1/2

			self.C_full[f_inter[j],f_dofs[j]] = -1

		# periodic
		for level in range(2):
			# lower case are ghosts, upper case are true dofs
			B0,t0 = np.array(self.mesh.periodic[level]).reshape((-1,2)).T
			ghost_list = np.array(t0)
			self.mesh.periodic_ghost.append(ghost_list)
			self.C_full[ghost_list] *= 0.
			for ind in [0,1,-2,-1]:
				self.C_full[:,B0[ind]] += self.C_full[:,t0[ind]]
				self.C_full[t0[ind],:] = self.C_full[B0[ind],:]
			self.Id[ghost_list] = 1.
			self.C_full[t0,B0] = 1.

		self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self,retfig=False):
		fig = vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[1],'vert')
		if retfig: return fig

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('vert')
		if retfig: return fig

	def xy_to_e(self,x,y):
		n_x_els = [self.N,2*self.N]
		
		x -= (x==1)*1e-14
		y -= (y==1)*1e-14
		fine = True if y >= 0.5 else False

		if fine:
			y_ind = int((y-.5)/self.h+1/2)
			x_ind = int(x/self.h)
		else:
			y_ind = int(y/2/self.h+1/2)
			x_ind = int(x/2/self.h)
		el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		return e

