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
		self.dom = [x,x+h,y,y+h]

		if y < 0 or y > 1-h:
			self.half = True
			self.dom = [x,x+h,max(0,y),max(0,y)+h/2]
		
		tmp0,tmp1 = self.dom[2:]
		self.plot = [[x,x+h,x+h,x,x],
					 [tmp0,tmp0,tmp1,tmp1,tmp0]]

	def add_dofs(self,strt,xlen):
		if len(self.dof_ids) != 0:
			return
		for ii in range(2):
			for jj in range(2):
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
		self.interface = True

class Mesh:
	def __init__(self,N):
		self.N = N # number of fine elements 
				   # from x=0.5 to x=1.0
		self.h = 0.5/N
		self.dofs = {}
		self.elements = []
		self.boundaries = []

		self.dof_count = 0
		self.el_count = 0
		
		self.n_els = []

		self.interface = {}
		
		self._make_coarse()
		self._make_fine()

		self._update_elements()

	def _make_coarse(self):
		raise ValueError('virtual needs to be overwritten')

	def _make_fine(self):
		raise ValueError('virtual needs to be overwritten')

	def _update_elements(self):
		for e in self.elements:
			e.update_dofs(self.dofs)

class Solver:
	def __init__(self,N,u,f=None,qpn=5,meshtype=Mesh):
		self.N = N
		self.ufunc = u
		self.ffunc = f #needs to be overwritten 
		self.qpn = qpn

		self.mesh = meshtype(N)
		self.h = self.mesh.h

		self.C = None
		self.Id = None

		self.U_lap = None
		self.U_proj = None

		self._solved = False
		self._use_halves = True

	def _build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		F = np.zeros(num_dofs)
		myfunc = self.ufunc if proj else self.ffunc

		id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

		for e in self.mesh.elements:
			y0,y1 = 0,e.h
			if self._use_halves:
				y0 = e.h/2*(e.y<0)
				y1 = e.h-e.h/2*(e.y+e.h>1)
			if e.interface: y1 *= 3/4
			for test_id,dof in enumerate(e.dof_list):

				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi1_2d_ref(x,y,e.h,test_ind,e.interface)
				func = lambda x,y: phi_test(x,y) * myfunc(x+e.x,y+e.y)
				val = gauss(func,0,e.h,y0,y1,self.qpn)

				F[dof.ID] += val

		if proj:
			self.F_proj = F
		else:
			self.F = F

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}
		
		base_k = local_stiffness(self.h,qpn=self.qpn)
	
		if self._use_halves:
			top_k = local_stiffness(self.h,qpn=self.qpn,half=0)
			half_ks = [top_k,top_k.copy()[::-1,::-1]]

		interface_k = local_stiffness(self.h*2,qpn=self.qpn,I=True)

		for e in self.mesh.elements:
			for test_id,dof in enumerate(e.dof_list):
				if self._use_halves and e.half:
					self.K[dof.ID,e.dof_ids] += half_ks[e.y<0][test_id]
				elif e.interface:
					self.K[dof.ID,e.dof_ids] += interface_k[test_id]
				else:
					self.K[dof.ID,e.dof_ids] += base_k[test_id]

	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

		base_m = local_mass(self.h,qpn=self.qpn)
	
		if self._use_halves:
			top_m = local_mass(self.h,qpn=self.qpn,half=0)
			half_ms = [top_m,top_m.copy()[::-1,::-1]]

		interface_m = local_mass(self.h*2,qpn=self.qpn,I=True)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			for test_id,dof in enumerate(e.dof_list):
				if self._use_halves and e.half:
					self.M[dof.ID,e.dof_ids] += half_ms[e.y<0][test_id] * scale
				elif e.interface:
					self.M[dof.ID,e.dof_ids] += interface_m[test_id]
				else:
					self.M[dof.ID,e.dof_ids] += base_m[test_id] * scale

	def projection(self):
		self._build_mass()
		self._build_force()
		self._setup_constraints()
		LHS = self.C_rect.T @ self.M @ self.C_rect
		RHS = self.C_rect.T @ (self.F_proj - self.M @ self.dirichlet)
		x_proj = la.solve(LHS,RHS)
		self.U_proj = self.C_rect @ x_proj + self.dirichlet
		self._solved = True
		return x_proj

	def laplace(self):
		if self.f is None:
			raise ValueError('f not set, call .add_force(func)')
		self._build_stiffness()
		self._build_force()
		self._setup_constraints()
		LHS = self.C_rect.T @ self.K @ self.C_rect
		RHS = self.C_rect.T @ (self.F - self.K @ self.dirichlet)
		x_lap = la.solve(LHS,RHS)
		self.U_lap = self.C_rect @ x_lap + self.dirichlet
		self._solved = True
		return x_lap

	def _setup_constraints(self):
		print('virtual not overwritten')

	def add_force(self,f):
		self.ffunc = f

	def add_field(self,u):
		self.ufunc = u

	def turn_off_halves(self):
		self._use_halves = False

	def turn_on_halves(self):
		self._use_halves = True

	def check_halves(self):
		print('use halves = '+str(self._use_halves))

	def vis_constraints(self):
		print('virtual not overwritten')
		if self.C is not None:
			vis_constraints(self.C_full,self.mesh.dofs)
		else:
			print('Constraints have not been set')

	def vis_mesh(self):
		for dof in self.mesh.dofs.values():
			plt.scatter(dof.x,dof.y)
		plt.show()

	def vis_dofs(self):
		data = []
		for dof in self.mesh.dofs.values():
			blocks = []
			dots = [[dof.x],[dof.y]]
			for e in dof.elements.values():
				blocks.append(e.plot)
			data.append([blocks,dots])

		return animate_2d(data,16)

	def vis_elements(self):
		data = []
		for e in self.mesh.elements:
			blocks = [e.plot]
			dots = [[],[]]
			for dof in e.dof_list:
				dots[0].append(dof.x)
				dots[1].append(dof.y)
			data.append([blocks,dots])

		return animate_2d(data,16)

	def xy_to_e(self,x,y):
		raise ValueError('virtual xy_to_e func not overwritten')

	def sol(self, weights=None, proj=False):

		if weights is None:
			assert self._solved
			if proj:
				assert self.U_proj is not None
				weights = self.U_proj
			else:
				assert self.U_lap is not None
				weights = self.U_lap

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
