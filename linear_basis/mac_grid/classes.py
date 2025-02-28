import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy import sparse
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


from linear_basis.mac_grid.helpers import *


class Node:
	def __init__(self,ID,j,i,k,x,y,z,h):
		self.ID = ID
		self.j = j
		self.i = i
		self.k = k
		self.x = x
		self.y = y
		self.z = z
		self.h = h
		self.elements = {}

	def add_element(self,e):
		if e.ID not in self.elements.keys():
			self.elements[e.ID] = e

class Element:
	def __init__(self,ID,j,i,k,x,y,z,h):
		self.ID = ID
		self.j = j
		self.i = i
		self.k = k
		self.x = x
		self.y = y
		self.z = z
		self.h = h
		self.dof_ids = []
		self.dof_list = []
		self.fine = False
		self.interface = False
		self.sides = []
		self.side = None
		self.dom = [x,x+h,y,y+h,z,z+h]

	def add_dofs(self,strt,xlen,ylen):
		if len(self.dof_ids) != 0:
			return
		for kk in range(2):
			for ii in range(2):
				for jj in range(2):
					self.dof_ids.append(strt+xlen*(kk*ylen+ii)+jj)
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

	def set_interface(self,which):
		self.interface = True
		self.sides.append(which)

		tmp_dict = {2:[4,6],3:[5,7]}
		if len(self.sides) == 1:
			self.side = self.sides[0]
		else:
			mx,mn = max(self.sides),min(self.sides)
			self.side = tmp_dict[mx][mn]
		

		if which == 0:
			self.dom[3] = self.y+self.h/2
		elif which == 1:
			self.dom[2] = self.y+self.h/2
		elif which == 2:
			self.dom[5] = self.z+self.h/2
		else:
			self.dom[4] = self.z+self.h/2

class Mesh:
	def __init__(self,N):
		self.N = N # number of fine elements 
				   # from x=0.5 to x=1.0
		self.h = 0.5/N
		self.dofs = {}
		self.elements = []
		self.boundaries = []
		self.periodic = []
		self.periodic_ghost = []

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

		self._setup_constraints()
		self.spC = sparse.csc_matrix(self.C)

	def _build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		F = np.zeros(num_dofs)
		myfunc = self.ufunc if proj else self.ffunc

		id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}

		for e in self.mesh.elements:
			y0,y1 = e.dom[2]-e.y, e.dom[3]-e.y
			z0,z1 = e.dom[4]-e.z, e.dom[5]-e.z
			for test_id,dof in enumerate(e.dof_list):
				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y,z: phi1_3d_ref(x,y,z,e.h,test_ind)
				func = lambda x,y,z: phi_test(x,y,z) * myfunc(x+e.x,y+e.y,z+e.z)
				val = gauss(func,0,e.h,y0,y1,z0,z1,self.qpn)

				F[dof.ID] += val
		if proj:
			self.F_proj = F
		else:
			self.F = F

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}
		
		base_k = local_stiffness(self.h,qpn=self.qpn)
	
		_k0 = local_stiffness(self.h,qpn=self.qpn,y1=.5)
		_k1 = local_stiffness(self.h,qpn=self.qpn,y0=.5)
		_k2 = local_stiffness(self.h,qpn=self.qpn,z1=.5)
		_k3 = local_stiffness(self.h,qpn=self.qpn,z0=.5)
		_k4 = local_stiffness(self.h,qpn=self.qpn,y1=.5,z1=.5)
		_k5 = local_stiffness(self.h,qpn=self.qpn,y1=.5,z0=.5)
		_k6 = local_stiffness(self.h,qpn=self.qpn,y0=.5,z1=.5)
		_k7 = local_stiffness(self.h,qpn=self.qpn,y0=.5,z0=.5)
		interface_k = [_k0,_k1,_k2,_k3,_k4,_k5,_k6,_k7]

		for e in self.mesh.elements:
			for test_id,dof in enumerate(e.dof_list):
				if e.interface:
					self.K[dof.ID,e.dof_ids] += interface_k[e.side][test_id]
				else:
					self.K[dof.ID,e.dof_ids] += base_k[test_id]
		self.spK = sparse.csc_matrix(self.K)

	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}

		base_m = local_mass(self.h,qpn=self.qpn)
	
		_m0 = local_mass(self.h,qpn=self.qpn,y1=.5)
		_m1 = local_mass(self.h,qpn=self.qpn,y0=.5)
		_m2 = local_mass(self.h,qpn=self.qpn,z1=.5)
		_m3 = local_mass(self.h,qpn=self.qpn,z0=.5)
		_m4 = local_mass(self.h,qpn=self.qpn,y1=.5,z1=.5)
		_m5 = local_mass(self.h,qpn=self.qpn,y1=.5,z0=.5)
		_m6 = local_mass(self.h,qpn=self.qpn,y0=.5,z1=.5)
		_m7 = local_mass(self.h,qpn=self.qpn,y0=.5,z0=.5)
		interface_m = [_m0,_m1,_m2,_m3,_m4,_m5,_m6,_m7]

		for e in self.mesh.elements:
			scale = 1 if e.fine else 8
			for test_id,dof in enumerate(e.dof_list):
				if e.interface:
					self.M[dof.ID,e.dof_ids] += interface_m[e.side][test_id]*scale
				else:
					self.M[dof.ID,e.dof_ids] += base_m[test_id] * scale
		self.spM = sparse.csc_matrix(self.M)

	def projection(self):
		self._build_mass()
		self._build_force(proj=True)
		LHS = self.spC.T * self.spM * self.spC
		RHS = self.spC.T.dot(self.F_proj - self.spM.dot(self.dirichlet))
		x_proj,conv = sla.cg(LHS,RHS,rtol=1e-12)
		assert conv==0
		self.U_proj = self.spC.dot( x_proj) + self.dirichlet
		self._solved = True
		return x_proj

	def laplace(self):
		if self.ffunc is None:
			raise ValueError('f not set, call .add_force(func)')
		self._build_stiffness()
		self._build_force()
		LHS = self.spC.T * self.spK * self.spC
		RHS = self.spC.T.dot(self.F - self.spK.dot( self.dirichlet))
		x_lap,conv = sla.cg(LHS,RHS,rtol=1e-12)
		assert conv==0
		self.U_lap = self.spC.dot( x_lap) + self.dirichlet
		self._solved = True
		return x_lap

	def _setup_constraints(self):
		print('virtual not overwritten')

	def add_force(self,f):
		self.ffunc = f

	def add_field(self,u):
		self.ufunc = u

	def vis_constraints(self):
		if True:
			fig,ax = plt.subplots(2,2)
			for csy in [0,1]:
				for csz in [0,1]:
					cinds = cgrid[csy:ends[csy],csz:ends[csz]]
					for fsy in [0,1]:
						for fsz in [0,1]:
							finds = fgrid[fsy::2,fsz::2]
							print((csy,csz,fsy,fsz),cinds.size,finds.size)
							v = frac[csy==fsy]*frac[csz==fsz]
							self.C_full[finds,cinds] = v
							for (f,c) in zip(finds.flatten(),cinds.flatten()):
								ax[csy,csz].plot([self.mesh.dofs[f].y,self.mesh.dofs[c].y],[self.mesh.dofs[f].z,self.mesh.dofs[c].z],color=cols[v])
								ax[csy,csz].scatter([self.mesh.dofs[f].y,self.mesh.dofs[c].y],[self.mesh.dofs[f].z,self.mesh.dofs[c].z],color=cols[v])
			plt.show()
		print('virtual not overwritten')
		vis_constraints(self.C_full,self.mesh.dofs)

	def vis_periodic(self,gridtype):
		fig = vis_periodic(self.C_full,self.mesh.dofs,gridtype)
		return fig

	def vis_mesh(self,corner=False,retfig=False):

		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		mk = ['^','o']
		for ind,dof in enumerate(self.mesh.dofs.values()):
			m = mk[dof.h==self.mesh.h]
			c = 'C0' if ind in self.true_dofs else 'C1'
			cind = 2*(ind in self.true_dofs)+((ind in self.true_dofs)==(dof.h==self.mesh.h))
			c = 'C'+str(cind)
			alpha = 1 if ind in self.true_dofs else .5
			ax.scatter(dof.x,dof.y,dof.z,marker=m,color=c,alpha=alpha)

		plt.show()
		return

	def vis_dof_sol(self,proj=False,err=False):
		U = self.U_proj if proj else self.U_lap
		x0,y0,z0,c0 = [], [], [], []
		x1,y1,z1,c1 = [], [], [], []
		for dof in self.mesh.dofs.values():
			if dof.h == self.h:
				x1.append(dof.x)
				y1.append(dof.y)
				z1.append(dof.z)
				val = U[dof.ID]
				if err: val = abs(val-self.ufunc(dof.x,dof.y,dof.z))
				c1.append(val)


			else:
				x0.append(dof.x)
				y0.append(dof.y)
				z0.append(dof.z)
				val = U[dof.ID]
				if err: val = abs(val-self.ufunc(dof.x,dof.y,dof.z))
				c0.append(val)
		
		m = ['o' for v in c1]+['^' for v in c0]
		
		lo = min(c0+c1)
		hi = max(c0+c1)
		fig = plt.figure(figsize=plt.figaspect(1))
		ax = fig.add_subplot(projection='3d')
		plot1 = ax.scatter(x0,y0,z0,marker='^',c=c0,cmap='jet')#,vmin=vmin,vmax=vmax)
		fig.colorbar(plot1,location='left')
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		ax.set_zlim(-1.5*self.h,1+1.5*self.h)
		plt.show()

		fig = plt.figure(figsize=plt.figaspect(1))
		ax = fig.add_subplot(projection='3d')
		plot2 = ax.scatter(x1,y1,z1,marker='o',c=c1,cmap='jet')#,vmin=vmin,vmax=vmax)
		fig.colorbar(plot2,location='left')
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		ax.set_zlim(-1.5*self.h,1+1.5*self.h)
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

		def solution(x,y,z,e=None):
			if e is None:
				e = self.xy_to_e(x,y,z)

			id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}
			val = 0
			for local_id, dof in enumerate(e.dof_list):
				local_ind = id_to_ind[local_id]
				val += weights[dof.ID]*phi1_3d_eval(x,y,z,dof.h,dof.x,dof.y,dof.z)
			
			return val
		return solution

	def error(self,qpn=5,proj=False):
		uh = self.sol(proj=proj)
			
		l2_err = 0.
		for e in self.mesh.elements:
			func = lambda x,y,z: (self.ufunc(x,y,z)-uh(x,y,z,e))**2
			x0,x1,y0,y1,z0,z1 = e.dom
			val = gauss(func,x0,x1,y0,y1,z0,z1,qpn)
			l2_err += val
		return np.sqrt(l2_err)
