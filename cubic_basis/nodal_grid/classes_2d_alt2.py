import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from cubic_basis.nodal_grid.helpers_2d import *
from cubic_basis.nodal_grid.shape_functions_2d import phi3_dxx


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
		self.Interface = False
		self.dom = [x,x+h,y,y+h]
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
		self.interface = [[],[]]
		self.periodic = [[],[]]
		self.interface_offset = [[],[],[],[],[],[],[],[]]
		
		self._make_coarse()
		self._make_fine()

		self._update_elements()

	def _make_coarse(self):
		H = self.h*2
		xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
		ydom = np.linspace(-H,1+H,self.N+3)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0<=x<.5) and (0<=y<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id += 1

				if x==0.:
					self.boundaries.append(dof_id)
				elif y < 2*H or y > 1.-2*H:
					self.periodic[0].append(dof_id)
				
				if (x == 0.5) and (0 <= y < 1):
					self.interface[0].append(dof_id)

				if (.5-2*H <= x) and (0 <= y <1):
					if x < .5-H: self.interface_offset[0].append(dof_id)
					elif x < .5: self.interface_offset[1].append(dof_id)
					elif x == .5: self.interface_offset[2].append(dof_id)
					else: self.interface_offset[3].append(dof_id)

				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		H = self.h
		xdom = np.linspace(0.5-H,1.+H,self.N+3)
		ydom = np.linspace(-H,1+H,2*self.N+3)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0.5<=x<1.) and (0<=y<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					e_id += 1

				if x==1.:# and (0 <= y < 1):#or y==0. or y==1:
					self.boundaries.append(dof_id)
				elif y < 2*H or y > 1.-2*H:
					self.periodic[1].append(dof_id)
				if (x == 0.5) and (0 <= y < 1):
					self.interface[1].append(dof_id)
				if (x <= .5+2*H) and (0 <= y <1):
					if x < .5: self.interface_offset[4].append(dof_id)
					elif x == .5: self.interface_offset[5].append(dof_id)
					elif x < .5+2*H: self.interface_offset[6].append(dof_id)
					else: self.interface_offset[7].append(dof_id)

				dof_id += 1

	def _update_elements(self):
		for e in self.elements:
			e.update_dofs(self.dofs)

class SolverAlt2:
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
				phi_test = lambda x,y: phi3_2d_ref(x,y,e.h,test_ind)
				func = lambda x,y: phi_test(x,y) * self.ffunc(x+e.x,y+e.y)
				val = gauss(func,0,e.h,0,e.h,self.qpn)

				self.F[dof.ID] += val

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		
		base_k = local_stiffness(self.h)

		for e in self.mesh.elements:
			for test_id,dof in enumerate(e.dof_list):
				self.K[dof.ID,e.dof_ids] += base_k[test_id]


	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		
		base_m = local_mass(self.h,qpn=self.qpn)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			for test_id,dof in enumerate(e.dof_list):
				self.M[dof.ID,e.dof_ids] += base_m[test_id] * scale


	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_side = np.array(self.mesh.interface_offset[:4])
		f_side = np.array(self.mesh.interface_offset[4:])
		#c_side = np.array(self.mesh.interface_offset[:4])
		#f_side = np.array(self.mesh.interface_offset[4:])
		#to_set = np.array([c_side[-1],f_side[0,::2]])
		#to_use = np.vstack((c_side[:3],f_side[2:,::2]))

		#self.Id[to_set.flatten()] = 1
		#self.C_full[to_set.flatten()] *= 0

		#mat = np.array([[-1/3,5/3,-5,16/3,-2/3],[-1/12,2/3,1/4,1/3,-1/6]])
		#for ind in range(5):
		#	self.C_full[to_set[0],to_use[ind]] = mat[0,ind]
		#	self.C_full[to_set[1],to_use[ind]] = mat[1,ind]

		c_inter,f_inter = self.mesh.interface
		self.Id[f_inter] = 1
		self.C_full[f_inter] *= 0

        # collocated are set to the coarse node
		self.C_full[f_inter[::2],c_inter[:]] = 1

		v1, v3 = phi3(1/2,1), phi3(3/2,1)

		for v, offset in zip([v3,v1,v1,v3],[1,0,-1,-2]):#ind,ID in enumerate(f_odd):
			self.C_full[f_inter[1::2],np.roll(c_inter,offset)] = v

		### add in fine square ghost
		self.Id[f_side[0]] = 1
		self.C_full[f_side[0]] *= 0
		for ind,v in enumerate([v3,v1,v1,v3]):#,[1,0,-1,-2]):#ind,ID in enumerate(f_odd):
			self.C_full[f_side[0,::2],c_side[ind]] = v
			for v2,offset in zip([v3,v1,v1,v3],[1,0,-1,-2]):
				self.C_full[f_side[0,1::2],np.roll(c_side[ind],offset)] = v*v2

		self.Id[c_side[-1]] = 1
		self.C_full[c_side[-1]] *= 0
		self.C_full[c_side[-1],f_side[-1,::2]] = 1
		for (cind,find) in zip(c_side[-1],f_side[-1,::2]):
			self.C_full[:,cind] += self.C_full[:,find]
    
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		for level in range(2):
			# lower are true dofs, upper are ghosts
			b0,B1,B2,T0,t1,t2 = np.array(self.mesh.periodic[level]).reshape((6,-1))
			ghost_list = np.array([b0,t1,t2])
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds, ds = [T0,B1,B2],[b0,t1,t2]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [0,1,2,3,-4,-3,-2,-1]:
					self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					self.C_full[d[ind],:] = self.C_full[D[ind],:]

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def solve(self):
		print('virtual not overwritten')

	def vis_constraints(self):
		if self.C is not None:
			vis_constraints(self.C,self.mesh.dofs)
		else:
			print('Constraints have not been set')

	def vis_mesh(self):
		markers = ['o','^']
		colors = ['grey','C1']
		for dof in self.mesh.dofs.values():
			m = markers[dof.h==self.h]
			c = colors[int(self.Id[dof.ID])]
			plt.scatter(dof.x,dof.y,marker=m,c=c)
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

	def xy_to_e(self,x,y):
		n_x_els = [self.N/2,self.N]
        
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		fine = True if x >= 0.5 else False
		x_ind = int((x-fine*.5)/((2-fine)*self.h))
		y_ind = int(y/((2-fine)*self.h))
		el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		return e

	def sol(self, weights=None):

		if weights is None:
			assert self.solved
			weights = self.U

		def solution(x,y):
			e = self.xy_to_e(x,y)

			id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
			val = 0
			for local_id, dof in enumerate(e.dof_list):
				local_ind = id_to_ind[local_id]
				val += weights[dof.ID]*phi3_2d_eval(x,y,dof.h,dof.x,dof.y)
			
			return val
		return solution

	def error(self,qpn=5):
		uh = self.sol()
		l2_err = 0.
		for e in self.mesh.elements:
			func = lambda x,y: (self.ufunc(x,y)-uh(x,y))**2
			val = gauss(func,e.x,e.x+e.h,e.y,e.y+e.h,qpn)
			l2_err += val
		return np.sqrt(l2_err)

class LaplaceAlt2(SolverAlt2):
	def __init__(self,N,u,f,qpn=5):
		super().__init__(N,u,f,qpn)
		self._setup_constraints()

	def solve(self):
		self._build_stiffness()
		self._build_force()
		LHS = self.C.T @ self.K @ self.C
		RHS = self.C.T @ (self.F - self.K @ self.dirichlet)
		#LHS = self.C_rect.T @ self.K @ self.C_rect
		#RHS = self.C_rect.T @ (self.F - self.K @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C @ x + self.dirichlet
		self.solved = True


class ProjectionAlt2(SolverAlt2):
	def __init__(self,N,u,qpn=5):
		super().__init__(N,u,u,qpn)

	def solve(self):
		self._build_mass()
		self._build_force()
		self._setup_constraints()
		LHS = self.C.T @ self.M @ self.C
		RHS = self.C.T @ (self.F - self.M @ self.dirichlet)
		#LHS = self.C_rect.T @ self.M @ self.C_rect
		#RHS = self.C_rect.T @ (self.F - self.M @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C @ x + self.dirichlet
		self.solved = True
		return x


