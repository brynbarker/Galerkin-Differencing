import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from helpers_phi1 import *


class Node:
	def __init__(self,ID,j,i,x,y,h):
		self.ID = ID
		self.j = j
		self.i = i
		self.x = x
		self.y = y
		self.h = h
		self.elements = {}

	def add_element(self,e):
		if e.ID not in self.elements.keys():
			self.elements[e.ID] = e

class Element:
	def __init__(self,ID,j,i,x,y,h):
		self.ID = ID
		self.j = j
		self.i = i
		self.x = x
		self.y = y
		self.h = h
		self.dof_ids = []
		self.dof_list = []
		self.fine = False
		self.interface = False
		self.half = False
		self.plot = [[x,x+h,x+h,x,x],
					 [y,y,y+h,y+h,y]]
		self.dom = [x,x+h,y,y+h]

		if y < 0 or y > 1-h:
			self.half = True
			self.dom = [x,x+h,max(0,y),max(0,y)+h/2]

	def add_dofs(self,strt,xlen):
		if len(self.dof_ids) != 0:
			return
		for ii in range(2):#4):
			for jj in range(2):#4):
				self.dof_ids.append(strt+xlen*ii+jj)
		return

	def update_dofs(self,dofs):
		if len(self.dof_list) != 0:
			return
		for dof_id in self.dof_ids:
			dof = dofs[dof_id]
			dof.add_element(self)
			self.dof_list.append(dof)
		return

	def set_fine(self):
		self.fine = True
		
	def set_interface(self):
		self.plot[1][2] -= self.h/4
		self.plot[1][3] -= self.h/4
		self.dom[-1] -= self.h/4
		self.interface = True

	#def set_interface_half(self):
	#	self.dom[2] = self.y+self.h/2
	#	self.half = True

class Mesh:
	def __init__(self,N):
		self.N = N # number of fine elements 
				   # from y=0.5 to y=1.0
		self.h = 0.5/N
		self.dofs = {}
		self.elements = []
		self.boundaries = []
		self.interface = [[],[]]
		self.periodic = [[],[]]
		
		self._make_coarse()
		self._make_fine()

		self._update_elements()

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
					if True:#(0<=x<=.5) and (0<=y<=1.):
						self.boundaries.append(dof_id)
				elif x < H or x > 1.-H:
					self.periodic[0].append(dof_id)
				
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

				if y>1.:# and (0 <= y < 1):#or y==0. or y==1:
					self.boundaries.append(dof_id)
				elif x < H or x > 1.-H:
					self.periodic[1].append(dof_id)
				if (y < 0.5+H) and (0 <= x < 1):
					self.interface[1].append(dof_id)
					#element.set_interface()
					# if y < 0.5+H: element.set_interface_half()

				dof_id += 1

	def _update_elements(self):
		for e in self.elements:
			e.update_dofs(self.dofs)

class Solver:
	def __init__(self,N,u,f=None,qpn=5):
		self.N = N
		self.ufunc = u
		self.ffunc = f #needs to be overwritten 
		self.qpn = qpn

		self.mesh = Mesh(N)
		self.h = self.mesh.h

		self.solved = False
		self.C = None
		self.Id = None

	def _build_force(self):
		num_dofs = len(self.mesh.dofs)
		self.F = np.zeros(num_dofs)

		id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

		for e in self.mesh.elements:
			y0 = e.h/2*(e.y<0)
			y1 = e.h-e.h/2*(e.y+e.h>1)
			for test_id,dof in enumerate(e.dof_list):

				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi1_2d_ref(x,y,e.h,test_ind)
				func = lambda x,y: phi_test(x,y) * self.ffunc(x+e.x,y+e.y)
				val = gauss(func,0,e.h,y0,y1,self.qpn)

				self.F[dof.ID] += val

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}
		
		k_coarse = local_stiffness(2*self.h,qpn=self.qpn)
		k_fine = local_stiffness(self.h,qpn=self.qpn)
		
		kh_coarse_top = local_stiffness(2*self.h,qpn=self.qpn,half=1)
		kh_coarse_bot = local_stiffness(2*self.h,qpn=self.qpn,half=0)
		
		kh_fine_top = local_stiffness(self.h,qpn=self.qpn,half=1)
		kh_fine_bot = local_stiffness(self.h,qpn=self.qpn,half=0)

		local_ks = [k_coarse,k_fine]
		half_ks = [[kh_coarse_top,kh_coarse_bot],
				   [kh_fine_top,kh_fine_bot]]

		for e in self.mesh.elements:
			local_k = local_ks[e.fine]
			if e.half:
				local_k = half_ks[e.fine][e.y<0]
			for test_id,dof in enumerate(e.dof_list):
				self.K[dof.ID,e.dof_ids] += local_k[test_id]


	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

		base_m = local_mass(self.h,qpn=self.qpn)
	
		top_m = local_mass(self.h,qpn=self.qpn,half=1)
		bot_m = local_mass(self.h,qpn=self.qpn,half=0)
		half_ms = [top_m,bot_m]

		interface_m = local_mass(self.h*2,qpn=self.qpn,I=True)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			for test_id,dof in enumerate(e.dof_list):
				if e.half:
					self.M[dof.ID,e.dof_ids] += half_ms[e.y<0][test_id] * scale
				elif e.interface:
					self.M[dof.ID,e.dof_ids] += interface_m[test_id]
				else:
					self.M[dof.ID,e.dof_ids] += base_m[test_id] * scale

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter,f_inter = self.mesh.interface
		self.Id[f_inter] = 1
		self.C[f_inter] *= 0

        # collocated are set to the coarse node
		self.C[f_inter[::2],c_inter[:]] = 1

		self.C[f_inter[1::2],np.roll(c_inter,-1)] = 1/2
		self.C[f_inter[1::2],c_inter] = 1/2

		# dirichlet
		for dof_id in self.mesh.boundaries:
			self.C[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		# periodic
		for level in range(2):
			# lower case are ghosts, upper case are true dofs
			B0,t0 = np.array(self.mesh.periodic[level]).reshape((-1,2)).T
			ghost_list = np.array(t0)#np.hstack((b0,t1))
			self.C[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			self.C[t0,B0] = 1.

			if level == 1:
				self.C[t0[0],:] = self.C[B0[0],:]

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C_rect = self.C[:,self.true_dofs]
	def solve(self):
		print('virtual not overwritten')

	def vis_constraints(self):
		if self.C is not None:
			vis_constraints(self.C,self.mesh.dofs)
		else:
			print('Constraints have not been set')

	def vis_mesh(self):
		for dof in self.mesh.dofs.values():
			plt.scatter(dof.x,dof.y)
		plt.show()

	def vis_dofs(self):
		frame = [[0,0,1,1,0,0,1],[.5,1,1,0,0,.5,.5]]
		data = []
		for dof in self.mesh.dofs.values():
			blocks = []
			dots = [[dof.x],[dof.y]]
			for e in dof.elements.values():
				blocks.append(e.plot)
			data.append([blocks,dots])

		return animate_2d([frame],data,16)

	def vis_elements(self):
		frame = [[0,0,1,1,0,0,1],[.5,1,1,0,0,.5,.5]]
		data = []
		for e in self.mesh.elements:
			blocks = [e.plot]
			dots = [[],[]]
			for dof in e.dof_list:
				dots[0].append(dof.x)
				dots[1].append(dof.y)
			data.append([blocks,dots])

		return animate_2d([frame],data,16)

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

	def sol(self, weights=None):

		if weights is None:
			assert self.solved
			weights = self.U

		def solution(x,y):
			e = self.xy_to_e(x,y)

			id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}
			val = 0
			for local_id, dof in enumerate(e.dof_list):
				local_ind = id_to_ind[local_id]
				val += weights[dof.ID]*phi1_2d_eval(x,y,dof.h,dof.x,dof.y,e.interface)
			
			return val
		return solution

	def error(self,qpn=5):
		uh = self.sol()
		l2_err = 0.
		for e in self.mesh.elements:
			func = lambda x,y: (self.ufunc(x,y)-uh(x,y))**2
			x0,x1,y0,y1 = e.dom
			val = gauss(func,x0,x1,y0,y1,qpn)
			l2_err += val
		return np.sqrt(l2_err)

class Laplace(Solver):
	def __init__(self,N,u,f,qpn=5):
		super().__init__(N,u,f,qpn)

	def solve(self):
		self._build_stiffness()
		self._build_force()
		self._setup_constraints()
		LHS = self.C_rect.T @ self.K @ self.C_rect
		RHS = self.C_rect.T @ (self.F - self.K @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C_rect @ x + self.dirichlet
		self.solved = True


class Projection(Solver):
	def __init__(self,N,u,qpn=5):
		super().__init__(N,u,u,qpn)

	def solve(self):
		self._build_mass()
		self._build_force()
		self._setup_constraints()
		LHS = self.C_rect.T @ self.M @ self.C_rect
		RHS = self.C_rect.T @ (self.F - self.M @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C_rect @ x + self.dirichlet
		self.solved = True
		return x


