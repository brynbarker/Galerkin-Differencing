import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


from linear_basis.mac_grid.helpers import *


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
		self.plot = [[x,x+h,x+h,x,x],
					 [y,y,y+h,y+h,y]]

		#if y < 0 or y > 1-h:
		#	self.half = True
		#	self.dom = [x,x+h,max(0,y),max(0,y)+h/2]
		
		#tmp0,tmp1 = self.dom[2:]
		#self.plot = [[x,x+h,x+h,x,x],
		#			 [tmp0,tmp0,tmp1,tmp1,tmp0]]

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
		self.periodic = {}
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
		self._use_halves = True

		self._setup_constraints()

	def _build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		F = np.zeros(num_dofs)
		myfunc = self.ufunc if proj else self.ffunc

		id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

		for e in self.mesh.elements:
			y0,y1 = 0,e.h
			#if self._use_halves:
			#	y0 = e.h/2*(e.y<0)
			#	y1 = e.h-e.h/2*(e.y+e.h>1)
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
		self.ffunc = self.ufunc
		self._build_mass()
		self._build_force(proj=True)
		LHS = self.C.T @ self.M @ self.C
		RHS = self.C.T @ (self.F_proj - self.M @ self.dirichlet)
		x_proj = la.solve(LHS,RHS)
		self.U_proj = self.C @ x_proj + self.dirichlet
		self._solved = True
		return x_proj

	def laplace(self):
		if self.ffunc is None:
			raise ValueError('f not set, call .add_force(func)')
		self._build_stiffness()
		self._build_force()
		LHS = self.C.T @ self.K @ self.C
		RHS = self.C.T @ (self.F - self.K @ self.dirichlet)
		x_lap = la.solve(LHS,RHS)
		self.U_lap = self.C @ x_lap + self.dirichlet
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
		vis_constraints(self.C_full,self.mesh.dofs)

	def vis_periodic(self,gridtype):
		fig = vis_periodic(self.C_full,self.mesh.dofs,gridtype)
		return fig

	def vis_mesh(self):
		fig,ax = plt.subplots(1,figsize=(7,7))
		mk = ['^','o']
		for dof in self.mesh.dofs.values():
			m = mk[dof.h==self.mesh.h]
			if dof.i == 0 and dof.j==0 and dof.ID !=0:
				plt.scatter(dof.x,dof.y,marker=m,color='k',label='dof')
			plt.scatter(dof.x,dof.y,marker=m,color='k')

		fine_inter = self.mesh.interface[1][0]+self.mesh.interface[1][1]
		for i,i_id in enumerate(fine_inter):
			assert self.Id[i_id]
			dof = self.mesh.dofs[i_id]
			if i==0:
				plt.scatter(dof.x,dof.y,marker='o',color='C1',label='interface')
			plt.scatter(dof.x,dof.y,marker='o',color='C1')

		for level in range(2):
			for i,g_id in enumerate(self.mesh.periodic_ghost[level]):
				assert self.Id[g_id]
				dof = self.mesh.dofs[g_id]
				if i==0 and level==1:
					plt.scatter(dof.x,dof.y,marker=mk[level],color='C0',label='periodic')
				plt.scatter(dof.x,dof.y,marker=mk[level],color='C0')

		for i,b_id in enumerate(self.mesh.boundaries):
			assert self.Id[b_id]
			dof = self.mesh.dofs[b_id]
			if i==0:
				plt.scatter(dof.x,dof.y,marker='^',color='C2',label='dirichlet')
			plt.scatter(dof.x,dof.y,marker='^',color='C2')

		ax.xaxis.set_major_locator(MultipleLocator(2*self.h))
		ax.xaxis.set_minor_locator(MultipleLocator(self.h))
		
		ax.yaxis.set_major_locator(MultipleLocator(2*self.h))
		ax.yaxis.set_minor_locator(MultipleLocator(self.h))
		
		plt.plot([0,1,1,0,0],[0,0,1,1,0],'k:',linewidth=2)
		
		ax.xaxis.grid(True,'minor',linewidth=.5)
		ax.yaxis.grid(True,'minor',linewidth=.5)
		ax.xaxis.grid(True,'major',linewidth=1)
		ax.yaxis.grid(True,'major',linewidth=1)
		plt.xlim(-2*self.h,1+2*self.h)
		plt.ylim(-2*self.h,1+2*self.h)
		plt.xticks([0,.5,1])
		plt.yticks([0,.5,1])
		plt.legend()
		plt.show()
		return

		
		tmp = self.mesh.periodic[0]+self.mesh.periodic[1]
		m_dict = ['^','o']
		c_dict = ['C0','C2']
		fig,ax = plt.subplots(1,figsize=(5,5))
		for dof in self.mesh.dofs.values():
			m = m_dict[dof.h==self.h]
			try:	
				c = c_dict[self.Id[dof.ID] and dof.ID in tmp]
				if self.Id[dof.ID] and dof.ID not in tmp:
					c = 'C1'
			except:
				c = c_dict[dof.h==self.h]
			plt.scatter(dof.x,dof.y,marker=m,color=c)

		ax.xaxis.set_major_locator(MultipleLocator(2*self.h))
		ax.xaxis.set_minor_locator(MultipleLocator(self.h))
		
		ax.yaxis.set_major_locator(MultipleLocator(2*self.h))
		ax.yaxis.set_minor_locator(MultipleLocator(self.h))
		
		plt.plot([0,1,1,0,0],[0,0,1,1,0],'k:',linewidth=2)
		
		ax.xaxis.grid(True,'minor',linewidth=.5)
		ax.yaxis.grid(True,'minor',linewidth=.5)
		ax.xaxis.grid(True,'major',linewidth=1)
		ax.yaxis.grid(True,'major',linewidth=1)
		plt.xlim(-2*self.h,1+2*self.h)
		plt.ylim(-2*self.h,1+2*self.h)
		plt.show()

	def vis_dof_sol(self,proj=False,err=False):
		U = self.U_proj if proj else self.U_lap
		fig = plt.figure(figsize=(10,6))
		x0,y0,c0 = [], [], []
		x1,y1,c1 = [], [], []
		for dof in self.mesh.dofs.values():
			if dof.h == self.h:
				x1.append(dof.x)
				y1.append(dof.y)
				val = U[dof.ID]
				if err: val = abs(val-self.ufunc(dof.x,dof.y))
				c1.append(val)


			else:
				x0.append(dof.x)
				y0.append(dof.y)
				val = U[dof.ID]
				if err: val = abs(val-self.ufunc(dof.x,dof.y))
				c0.append(val)
		
		m = ['o' for v in c1]+['^' for v in c0]
		
		lo = min(c0+c1)
		hi = max(c0+c1)
		plt.subplot(121)
		plt.scatter(x0,y0,marker='^',vmin=lo,vmax=hi,c=c0,cmap='jet')#,vmin=vmin,vmax=vmax)
		plt.colorbar(location='left')
		plt.xlim(-.5*self.h,.5+.5*self.h)
		plt.ylim(-1.5*self.h,1+1.5*self.h)
		plt.subplot(122)
		plt.scatter(x1,y1,marker='o',vmin=lo,vmax=hi,c=c1,cmap='jet')#,vmin=vmin,vmax=vmax)
		plt.colorbar(location='right')
		plt.xlim(.5-.5*self.h,1+.5*self.h)
		plt.ylim(-1.5*self.h,1+1.5*self.h)
		plt.tight_layout()
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

	def error(self,qpn=5,proj=False):
		uh = self.sol(proj=proj)
		l2_err = 0.
		for e in self.mesh.elements:
			func = lambda x,y: (self.ufunc(x,y)-uh(x,y))**2
			x0,x1,y0,y1 = e.dom
			val = gauss(func,x0,x1,y0,y1,qpn)
			l2_err += val
		return np.sqrt(l2_err)