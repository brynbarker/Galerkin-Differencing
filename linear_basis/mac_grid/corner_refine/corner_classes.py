import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from linear_basis.mac_grid.classes import Node, Element, Mesh, Solver
from linear_basis.mac_grid.helpers import vis_constraints

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
		self.interface[0] = [[],[],[],[]]
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

				if x==H:
					self.boundaries.append(dof_id)
				#if y==H/2:
				#	self.boundaries.append(dof_id)
				if (x == 0.5 or x==0):
					self.interface[0][x>0].append(dof_id)
				if (y > 0.5 or y<H):
					self.interface[0][2+(y>.5)].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q1(self): #coarse
		self.interface[1] = [[],[],[],[]]
		H = self.h*2
		xdom = np.linspace(.5,1,int(self.N/2)+1)
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			if (i==0) or (i==ylen-1):
				y = y - y/abs(y)*H/4
			interface_element = (i==0) or (i==ylen-2)
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (y<.5) and (x<1.):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id += 1
					if interface_element: element.set_interface()

				if (x == 0.5 or x==1):
					self.interface[1][x==1].append(dof_id)
				#if y==H/2:
				#	self.boundaries.append(dof_id)
				if (y > 0.5 or y <0):
					self.interface[1][2+(y>0)].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q2(self):
		self.interface[2] = [[],[],[],[]]
		H = self.h*2
		xdom = np.linspace(0,0.5,int(self.N/2)+1)
		ydom = np.linspace(.5+H/2,1+H/2,int(self.N/2)+1)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (x<.5) and (y<1.-H):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id += 1

				if x == H:
					self.boundaries.append(dof_id)
				if (x==.5) or (x==0):
					self.interface[2][x>0].append(dof_id)
				if (y<.5+H or y>1-H):
					self.interface[2][2+(y>1-H)].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q3(self):
		self.interface[3] = [[],[],[],[]]
		H = self.h
		xdom = np.linspace(0.5,1.,self.N+1)
		ydom = np.linspace(.5+H/2,1-H/2,self.N)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (x<1.) and (y<1.-H):
					strt = dof_id#-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					e_id += 1

				if (y<.5+H or y>1-H):
					self.interface[3][2+(y>1-H)].append(dof_id)
				elif (x==.5) or (x==1):
					self.interface[3][x==1].append(dof_id)

				dof_id += 1

		self.dof_count = dof_id
		self.el_count = e_id

class CornerRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=CornerRefinementMesh)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		q0 = self.mesh.interface[0]
		q1 = self.mesh.interface[1]
		q2 = self.mesh.interface[2]
		q3 = self.mesh.interface[3]

		# clear
		ghosts = q1[0]+q1[1]+q2[2]+q2[3]+q3[0]+q3[1]+q3[2]+q3[3]
		for g in ghosts:
			self.Id[g] = 1
			self.C_full[g] *= 0 


		# horizontal refinement (x = .5)
		for i in range(2):
			self.C_full[q3[1-i][1::2],q2[i][:-2]] = 1/4
			self.C_full[q3[1-i][1::2],q2[i][1:-1]] = 3/4

			self.C_full[q3[1-i][::2],q2[i][1:-1]] = 1/4
			self.C_full[q3[1-i][::2],q2[i][:-2]] = 3/4

		# vertical refinement (y = .5)
		for i in range(2):
			self.C_full[q3[3-i][::2],q1[2+i]] = 1

			self.C_full[q3[3-i][1::2],q1[2+i][1:]] = 1/2
			self.C_full[q3[3-i][1::2],q1[2+i][:-1]] = 1/2


		# same level horizontally
		for i in range(2):
			self.C_full[q1[1-i][1:-1],q0[i][1:-1]] = 1

		# same level vertically 
		self.C_full[q2[2],q0[3]] = 1
		self.C_full[q2[3],q0[2]] = 1

		# corners
		for i in range(2):
			self.C_full[q1[i][0],q0[1-i][1]] = 1/4
			self.C_full[q1[i][0],q0[1-i][0]] = 3/4

			self.C_full[q1[i][-1],q0[1-i][-2]] = 1/4
			self.C_full[q1[i][-1],q0[1-i][-1]] = 3/4

		# distributing constraints
		for i in range(2):
			self.C_full[:,q0[i][-1]] += self.C_full[:,q2[i][0]]
			self.C_full[:,q0[i][0]] += self.C_full[:,q2[i][-2]]

		for i in range(2):
			self.C_full[:,q0[i][-1]] += 3/4*self.C_full[:,q1[1-i][-1]]
			self.C_full[:,q0[i][-2]] += 1/4*self.C_full[:,q1[1-i][-1]]

			self.C_full[:,q0[i][1]] += 1/4*self.C_full[:,q1[1-i][0]]
			self.C_full[:,q0[i][0]] += 3/4*self.C_full[:,q1[1-i][0]]

		self.internal_overlap = q1[0][1:-1]+q2[2]
		self.mesh.periodic_ghost = [q1[1][1:-1]+q2[3],[]]

		self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		# dirichlet
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self,retfig=False):
		fig = vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[3],'corner')
		if retfig: return fig
	
	def vis_mesh(self,retfig=True):
		fig = super().vis_mesh(corner=True,retfig=True)
		if retfig: return fig

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('corner')
		if retfig: return fig

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
