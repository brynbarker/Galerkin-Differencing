import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from ../linear_mac_classes import Node, Element, Mesh, Solver
from ../linear_mac_helpers import vis_constraints


class VerticalRefineMesh(Mesh):
	def __init__(self,N):
		super().__init__(N)

	def _make_coarse(self):
		H = self.h*2
		xdom = np.linspace(0,1,self.N+1)
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			if (i == ylen-1):
				y -= H/4
			interface_element = (i == ylen-2)
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (y<.5) and (x<1.):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id += 1
					if interface_element: element.set_interface()

				if y<0:
					self.boundaries.append(dof_id)
				elif x < H or x > 1.-H:
					self.boundaries.append(dof_id)
					#self.periodic[0].append(dof_id)
				if (y > 0.5) and (0 <= x < 1):
					self.interface[0].append(dof_id)

				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		H = self.h
		xdom = np.linspace(0,1,2*self.N+1)
		ydom = np.linspace(0.5+H/2,1.+H/2,self.N+1)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (y<1.) and (x<1.):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					e_id += 1

				if y>1.:
					self.boundaries.append(dof_id)
				elif x < H or x > 1.-H:
					self.boundaries.append(dof_id)
					#self.periodic[1].append(dof_id)
				if (y < 0.5+H) and (0 <= x < 1):
					self.interface[1].append(dof_id)

				dof_id += 1

class VerticalRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=VerticalRefineMesh)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter,f_inter = self.mesh.interface
		self.Id[f_inter] = 1
		self.C_full[f_inter] *= 0

		# collocated are set to the coarse node
		self.C_full[f_inter[::2],c_inter[:]] = 1

		self.C_full[f_inter[1::2],np.roll(c_inter,-1)] = 1/2
		self.C_full[f_inter[1::2],c_inter] = 1/2

		# dirichlet
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self):
		if self.C is not None:
			vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[1],'vert')
		else:
			print('Constraints have not been set')

	def xy_to_e(self,x,y):
		n_x_els = [self.N,2*self.N]
		
		x -= (x==1)*1e-14
		y -= (y==1)*1e-14
		fine = True if y >= 0.5+self.h/2 else False

		if fine:
			y_ind = int((y-.5)/self.h-1/2)
			x_ind = int(x/self.h)
		else:
			y_ind = int(y/2/self.h+1/2)
			x_ind = int(x/2/self.h)
		el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		return e

