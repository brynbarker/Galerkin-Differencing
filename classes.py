import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from helpers import *


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
		self.interface = False
		self.fine = False
		self.top = True
		self.plot = [[x,x+h,x+h,x,x],
					 [y,y,y+h,y+h,y]]

	def add_dofs(self,strt,xlen):
		if len(self.dof_ids) != 0:
			return
		for ii in range(4):
			for jj in range(4):
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

	def set_interface(self):
		self.interface = True

	def set_fine(self):
		self.fine = True

	def set_not_top(self):
		self.top = False

class Mesh:
	def __init__(self,N):
		self.N = N # number of fine elements 
				   # from x=0.5 to x=1.0
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
		xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
		ydom = np.linspace(0-H,1+H,self.N+3)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0<=x<.5) and (0<=y<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j-1,i-1,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id += 1

				if x==0.:
					if True:#(0<=x<=.5) and (0<=y<=1.):
						self.boundaries.append(dof_id)
				elif y <= H or y >= 1.-H:
					self.periodic[0].append(dof_id)
				
				if (0.5-H <= x) and (0 <= y < 1):
					if x == 0.5:
						self.interface[0].append(dof_id)

				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		H = self.h
		xdom = np.linspace(0.5-H,1.+H,self.N+3)
		ydom = np.linspace(0-H,1+H,2*self.N+3)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0.5<=x<1.) and (0<=y<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j-1,i-1,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					if (y/2/self.h)%1 == 0:
						element.set_not_top()
					e_id += 1

				if x==1.:# and (0 <= y < 1):#or y==0. or y==1:
					self.boundaries.append(dof_id)
				elif y <= H or y >= 1.-H:
					self.periodic[1].append(dof_id)
				if (x <= 0.5+2*H) and (0 <= y < 1):
					if False:#x != 0.5+H:
						self.interface[1].append(dof_id)
					if x == 0.5:
						self.interface[1].append(dof_id)
						element.set_interface()

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

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

		for e in self.mesh.elements:

			for test_id,dof in enumerate(e.dof_list):

				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi3_2d_ref(x,y,e.h,test_ind,e.interface,e.top)
				func = lambda x,y: phi_test(x,y) * self.ffunc(x,y)
				val = gauss(func,0,e.h,0,e.h,self.qpn)

				self.F[dof.ID] += val

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		
		k_coarse = local_stiffness(2*self.h,qpn=self.qpn)
		k_fine = local_stiffness(self.h)

		k_interface_0 = local_stiffness(self.h,interface=True,top=0,qpn=self.qpn)
		k_interface_1 = local_stiffness(self.h,interface=True,top=1,qpn=self.qpn)

		local_ks = [[k_coarse,k_fine],[k_interface_0,k_interface_1]]

		for e in self.mesh.elements:
			local_k = local_ks[e.interface][e.fine*e.top]
			for test_id,dof in enumerate(e.dof_list):
				self.K[dof.ID,e.dof_ids] += local_k[test_id]


	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		
		m_coarse = local_mass(2*self.h,qpn=self.qpn)
		m_fine = local_mass(self.h)

		m_interface_0 = local_mass(self.h,interface=True,top=0,qpn=self.qpn)
		m_interface_1 = local_mass(self.h,interface=True,top=1,qpn=self.qpn)

		local_ms = [[m_coarse,m_fine],[m_interface_0,m_interface_1]]

		for e in self.mesh.elements:
			local_m = local_ms[e.interface][e.fine*e.top]
			for test_id,dof in enumerate(e.dof_list):
				self.M[dof.ID,e.dof_ids] += local_m[test_id]

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros((num_dofs,num_dofs))
		self.C = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		A0,A1,A2,A3 = [phi3(j*self.h/4,self.h) for j in [1,3,5,7]]
		a0,a1 = [phi3_interface(j*self.h/2,self.h,0,False) for j in [1,3]]

		V = 1/(a0**2-a1**2) * np.array([[a0,-a1],[-a1,a0]]) @\
				np.array([[A3,A1-a1,A0-a0,A2,0],[0,A2,A0-a0,A1-a1,A3]])

		c_inter,f_inter = self.mesh.interface
		self.Id[f_inter,f_inter] = 1
		self.C[f_inter] *= 0

		self.C[f_inter[::2],c_inter[:]] = 1
		# self.C[f_inter[::2],c_inter[:]] = 1

		for ind in range(5):
			end = ind-2 if ind < 2 else None
			fend = None if ind < 3 else -3
			if ind == 0:
				self.C[f_inter[5:-1:4],c_inter[1:end:2]] = V[0][ind]
				self.C[f_inter[7::4],c_inter[1:end:2]] = V[1][ind]
			else:
				self.C[f_inter[1:fend:4],c_inter[ind-1:end:2]] = V[0][ind]
				self.C[f_inter[3:fend:4],c_inter[ind-1:end:2]] = V[1][ind]

		self.C[f_inter[1:4:2],[c_inter[-1]]*2] = V[:,0]
		self.C[f_inter[-3::2],[c_inter[0]]*2] = V[:,3]
		self.C[f_inter[-3::2],[c_inter[1]]*2] = V[:,4]

		#self.C[f_inter[0]] = -a0/a1*self.C[f_inter[2]]
		#self.C[f_inter[-1]] = -a0/a1*self.C[f_inter[-3]]

		#self.C[f_inter[0],c_inter[:4]] += np.array([A2,A0-a0,A1-a1,A3])/a1
		#self.C[f_inter[-1],c_inter[-4:]] += np.array([A3,A1-a1,A0-a0,A2])/a1
		

		for dof_id in self.mesh.boundaries:
			self.C[dof_id] *= 0
			self.Id[dof_id,dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		for level in range(2):
			# lower case are ghosts, upper case are true dofs
			b0,B1,B2,T0,t1,t2 = np.array(self.mesh.periodic[level]).reshape((6,-1))
			ghost_list = np.hstack((b0,np.hstack((t1,t2))))
			self.C[ghost_list] *= 0.
			self.Id[ghost_list,ghost_list] = 1.
			self.C[b0,T0] = 1.
			self.C[t1,B1] = 1.
			self.C[t2,B2] = 1.			  

			if level == 1:
				self.C[b0[1],:] = self.C[T0[1],:]
				self.C[t1[1],:] = self.C[B1[1],:]
				self.C[t2[1],:] = self.C[B2[1],:]

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
		frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
		data = []
		for dof in self.mesh.dofs.values():
			blocks = []
			dots = [[dof.x],[dof.y]]
			for e in dof.elements.values():
				blocks.append(e.plot)
			data.append([blocks,dots])

		return animate_2d([frame],data,16)

	def vis_elements(self):
		frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
		data = []
		for e in self.mesh.elements:
			blocks = [e.plot]
			dots = [[],[]]
			for dof in e.dof_list:
				dots[0].append(dof.x)
				dots[1].append(dof.y)
			data.append([blocks,dots])

		return animate_2d([frame],data,16)

	def sol(self, weights=None):
		n_x_els = [self.N/2,self.N]

		if weights is None:
			assert self.solved
			weights = self.U

		def solution(x,y):
			x -= (x==1)*1e-12
			y -= (y==1)*1e-12
			fine = True if x >= 0.5 else False
			x_ind = int((x-fine*.5)/((2-fine)*self.h))
			y_ind = int(y/((2-fine)*self.h))
			el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
			e = self.mesh.elements[int(el_ind)]
			#print(x,y,x_ind,y_ind,el_ind,e.plot)
			assert x >= min(e.plot[0]) and x <= max(e.plot[0])
			assert y >= min(e.plot[1]) and y <= max(e.plot[1])

			id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
			val = 0
			for local_id, dof in enumerate(e.dof_list):
				local_ind = id_to_ind[local_id]
				val += weights[dof.ID]*phi3_2d_eval(x,y,dof.h,dof.x,dof.y)
			
			return val
		return solution

class Laplace(Solver):
	def __init__(self,N,u,f,qpn=5):
		super().__init__(N,u,f,qpn)

	def solve(self):
		self._build_stiffness()
		self._build_force()
		self._setup_constraints()
		LHS = self.C.T @ self.K @ self.C + self.Id
		RHS = self.C.T @ (self.F - self.K @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C @ x + self.dirichlet
		self.solved = True


class Projection(Solver):
	def __init__(self,N,u,qpn=5):
		super().__init__(N,u,u,qpn)

	def solve(self):
		self._build_mass()
		self._build_force()
		self._setup_constraints()
		LHS = self.C.T @ self.M @ self.C + self.Id
		RHS = self.C.T @ (self.F - self.M @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C @ x + self.dirichlet
		self.solved = True
		return x


