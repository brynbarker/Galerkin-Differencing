import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from ../linear_mac_shape_functions import *
from ../linear_mac_classes import Node, Element, Mesh, Solver
from ../linear_mac_helpers import vis_constraints

class CornerRefinementMesh(Mesh):
	def __init__(self,N):
		super().__init__(N)

	def _make_coarse(self):
		self._make_q0()
		self._make_q1()
		self._make_q2()

	def _make_fine(self):
		self._make_q3()

	def _make_q0(self): #coarse
		self.interface[0] = [[],[]]
		H = self.h*2
		xdom = np.linspace(0,.5,int(self.N/2)+1)
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			interface_element = (i == ylen-2)
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (y<.5) and (x<.5):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id += 1

				if y<0 or x < H:
					self.boundaries.append(dof_id)
				if (x == 0.5):
					self.interface[0][0].append(dof_id)
				if (y > 0.5):
					self.interface[0][1].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q1(self): #coarse
		self.interface[1] = [[],[]]
		H = self.h*2
		xdom = np.linspace(.5,1,int(self.N/2)+1)
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
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

				if y<0 or x>1-H:
					self.boundaries.append(dof_id)
				if (x == 0.5):
					self.interface[1][0].append(dof_id)
				if (y > 0.5) and (x > 0.5):
					self.interface[1][1].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q2(self):
		self.interface[2] = [[],[]]
		H = self.h*2
		xdom = np.linspace(0,0.5,int(self.N/2)+1)
		ydom = np.linspace(.5+H/2,1+H/2,int(self.N/2)+1)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (x<.5) and (y<1.):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id += 1

				if x < H or y > 1-H:
					self.boundaries.append(dof_id)
				if (x==.5) and (y>.5+H):
					self.interface[2][0].append(dof_id)
				if (y<.5+H):
					self.interface[2][1].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q3(self):
		self.interface[3] = [[],[]]
		H = self.h
		xdom = np.linspace(0.5,1.,self.N+1)
		ydom = np.linspace(.5+H/2,1+H/2,self.N+1)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (x<1.) and (y<1.):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					e_id += 1

				if x==1. or y>1-H:
					self.boundaries.append(dof_id)
				if (x==.5) and (y>.5+H):
					self.interface[3][0].append(dof_id)
				if (y<.5+H):
					self.interface[3][1].append(dof_id)

				dof_id += 1

		self.dof_count = dof_id
		self.el_count = e_id

class CornerRefineSolver:
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=CornerRefineMesh)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		q0_x, q0_y = self.mesh.interface[0]
		q1_x, q1_y = self.mesh.interface[1]
		q2_x, q2_y = self.mesh.interface[2]
		q3_x, q3_y = self.mesh.interface[3]

		# clear
		ghosts = [q1_x,q2_y,q3_x,q3_y]
		for g in ghosts:
			self.Id[g] = 1
			self.C_full[g] *= 0 

		# same level
		self.C_full[q1_x[:-1],q0_x[:-1]] = 1
		self.C_full[q2_y,q0_y] = 1

		# horizontal refinement (x = .5)
		self.C_full[q3_x[2::2],q2_x[:-1]] = 3/4
		self.C_full[q3_x[::2],q2_x] = 1/4

		self.C_full[q3_x[3::2],q2_x[:-1]] = 1/4
		self.C_full[q3_x[1::2],q2_x] = 3/4

		# horizontal refinement (y = .5)
		self.C_full[q3_y[2::2],q1_y] = 1

		self.C_full[q3_y[1::2],q1_y] = 1/2
		self.C_full[q3_y[3::2],q1_y[:-1]] = 1/2

		# corner
		self.C_full[q1_x[-1],q0_x[-2:]] = [1/4,3/4]
		self.C_full[q3_y[0],q0_x[-2:]] = [1/4,3/4]
		self.C_full[q3_y[1],q0_x[-2:]] = [1/8,3/8]
		self.C_full[q3_x[:2],q0_x[-1]] = [3/4,1/4]

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
			vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[3],'corner')
		else:
			print('Constraints have not been set')

	def xy_to_e(self,x,y):
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12

		if (x<=.5) and (y<= .5+self.h): #q0
			x_ind = int(x/2/self.h)
			y_ind = int(y/2/self.h+1/2)
			e_ind = y_ind*self.N/2+x_ind
		elif (x<=.5):# q2
			x_ind = int(x/2/self.h)
			y_ind = int((y-.5)/2/self.h-1/2)
			e_ind = self.mesh.n_els[1]+y_ind*self.N/2+x_ind
		elif (y<=.5+self.h/2): #q1
			x_ind = int((x-.5)/2/self.h)
			y_ind = int(y/2/self.h+1/2)
			e_ind = self.mesh.n_els[0]+y_ind*self.N/2+x_ind
		else: #q3
			x_ind = int((x-.5)/self.h)
			y_ind = int((y-.5)/self.h-1/2)
			e_ind = self.mesh.n_els[2]+y_ind*self.N+x_ind
			
		e = self.mesh.elements[int(e_ind)]
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		return e
