import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy import sparse
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.integrate import tplquad


from linear_basis.mac_grid.helpers import *
M_ref = np.array([[8., 4., 4., 2., 4., 2., 2., 1.],
                  [4., 8., 2., 4., 2., 4., 1., 2.],
                  [4., 2., 8., 4., 2., 1., 4., 2.],
                  [2., 4., 4., 8., 1., 2., 2., 4.],
                  [4., 2., 2., 1., 8., 4., 4., 2.],
                  [2., 4., 1., 2., 4., 8., 2., 4.],
                  [2., 1., 4., 2., 4., 2., 8., 4.],
                  [1., 2., 2., 4., 2., 4., 4., 8.]])/27/8
MY = np.array([[0.875, 0.875, 0.5  , 0.5  , 0.875, 0.875, 0.5  , 0.5  ],
				[0.875, 0.875, 0.5	, 0.5  , 0.875, 0.875, 0.5	, 0.5  ],
				[0.5  , 0.5  , 0.125, 0.125, 0.5  , 0.5  , 0.125, 0.125],
				[0.5  , 0.5  , 0.125, 0.125, 0.5  , 0.5  , 0.125, 0.125],
				[0.875, 0.875, 0.5	, 0.5  , 0.875, 0.875, 0.5	, 0.5  ],
				[0.875, 0.875, 0.5	, 0.5  , 0.875, 0.875, 0.5	, 0.5  ],
				[0.5  , 0.5  , 0.125, 0.125, 0.5  , 0.5  , 0.125, 0.125],
				[0.5  , 0.5  , 0.125, 0.125, 0.5  , 0.5  , 0.125, 0.125]])
MZ = np.array([[0.875, 0.875, 0.875, 0.875, 0.5  , 0.5	, 0.5  , 0.5  ],
				[0.875, 0.875, 0.875, 0.875, 0.5  , 0.5  , 0.5	, 0.5  ],
				[0.875, 0.875, 0.875, 0.875, 0.5  , 0.5  , 0.5	, 0.5  ],
				[0.875, 0.875, 0.875, 0.875, 0.5  , 0.5  , 0.5	, 0.5  ],
				[0.5  , 0.5  , 0.5	, 0.5  , 0.125, 0.125, 0.125, 0.125],
				[0.5  , 0.5  , 0.5	, 0.5  , 0.125, 0.125, 0.125, 0.125],
				[0.5  , 0.5  , 0.5	, 0.5  , 0.125, 0.125, 0.125, 0.125],
				[0.5  , 0.5  , 0.5	, 0.5  , 0.125, 0.125, 0.125, 0.125]])
K_ref = np.array([[ 4., -0., -0., -1., -0., -1., -1., -1.],
                  [-0.,  4., -1., -0., -1., -0., -1., -1.],
                  [-0., -1.,  4., -0., -1., -1., -0., -1.],
                  [-1., -0., -0.,  4., -1., -1., -1., -0.],
                  [-0., -1., -1., -1.,  4., -0., -0., -1.],
                  [-1., -0., -1., -1., -0.,  4., -1., -0.],
                  [-1., -1., -0., -1., -0., -1.,  4., -0.],
                  [-1., -1., -1., -0., -1., -0., -0.,  4.]])/12
KY = np.array([[.75, 1, 1, .5, 1, 1, .5, .5],
               [1, .75, .5, 1, 1, 1, .5, .5],
               [1, .5, .25, 1, .5, .5, 1, 0],
               [.5, 1, 1, .25, .5, .5, 0, 1],
               [1, 1, .5, .5, .75, 1, 1, .5],
               [1, 1, .5, .5, 1, .75, .5, 1],
               [.5, .5, 1, 0, 1, .5, .25, 1],
               [.5, .5, 0, 1, .5, 1, 1, .25]])
KY_add = np.array([[ 0., -1.,  0.,  0., -1.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.],
                   [-1.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                   [ 0., -1.,  0.,  0., -1.,  0.,  0.,  0.],
                   [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.]])/3/16

class Node:
	def __init__(self,ID,j,i,k,x,y,z,h,patch=None):
		self.ID = ID
		self.j = j
		self.i = i
		self.k = k
		self.x = x
		self.y = y
		self.z = z
		self.h = h
		self.elements = {}
		self.patch = patch

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
		self.quad_vals = compute_gauss(qpn)
		self.spM = None
		self.spK = None

	def _build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		F = np.zeros(num_dofs)
		myfunc = self.ufunc if proj else self.ffunc

		g_reg,g_interface,g_p,g_w = self.quad_vals

		for e in self.mesh.elements:
			y0,y1 = e.dom[2]-e.y, e.dom[3]-e.y
			z0,z1 = e.dom[4]-e.z, e.dom[5]-e.z
			func = lambda x,y,z: myfunc(x+e.x,y+e.y,z+e.z)
			f_vals = gauss_vals(func,0,e.h,y0,y1,z0,z1,self.qpn,g_p)
			for test_id,dof in enumerate(e.dof_list):
				if e.interface:
					phi_vals = g_interface[e.side][test_id]
				else:
					phi_vals = g_reg[test_id]
				v = super_quick_gauss(f_vals,phi_vals,0,e.h,y0,y1,z0,z1,self.qpn,g_w)
				F[dof.ID] += v
		if proj:
			self.F_proj = F
		else:
			self.F = F

	def _build_stiffness(self):
		if self.spK is not None: return
		num_dofs = len(self.mesh.dofs)
		Kr, Kc, Kd = [],[],[]

		id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}
		
		base_k = K_ref*self.h #local_stiffness(self.h,qpn=self.qpn)
	
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
			scale = 1 if e.fine else 2
			for test_id,dof in enumerate(e.dof_list):
				Kr += [dof.ID]*len(e.dof_ids)
				Kc += e.dof_ids
				if e.interface:
					Kd += list(interface_k[e.side][test_id]*scale)
				else:
					Kd += list(base_k[test_id]*scale)
		self.spK = sparse.coo_array((Kd,(Kr,Kc)),shape=(num_dofs,num_dofs)).tocsc()

	def _build_mass(self):
		if self.spM is not None: return
		num_dofs = len(self.mesh.dofs)
		Mr, Mc, Md = [],[],[]

		id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}

		base_m = M_ref * self.h**3  #local_mass(self.h,qpn=self.qpn)

		_m0 = base_m*MY  #local_mass(self.h,qpn=self.qpn,y1=.5)
		_m1 = base_m-_m0 #local_mass(self.h,qpn=self.qpn,y0=.5)
		_m2 = base_m*MZ  #local_mass(self.h,qpn=self.qpn,z1=.5)
		_m3 = base_m-_m2 #local_mass(self.h,qpn=self.qpn,z0=.5)
		_m4 = _m0*MZ	 #local_mass(self.h,qpn=self.qpn,y1=.5,z1=.5)
		_m5 = _m0-_m4	 #local_mass(self.h,qpn=self.qpn,y1=.5,z0=.5)
		_m6 = _m1*MZ	 #local_mass(self.h,qpn=self.qpn,y0=.5,z1=.5)
		_m7 = _m1-_m6	 #local_mass(self.h,qpn=self.qpn,y0=.5,z0=.5)
		interface_m = [_m0,_m1,_m2,_m3,_m4,_m5,_m6,_m7]

		for e in self.mesh.elements:
			scale = 1 if e.fine else 8
			for test_id,dof in enumerate(e.dof_list):
				Mr += [dof.ID]*len(e.dof_ids)
				Mc += e.dof_ids
				if e.interface:
					Md += list(interface_m[e.side][test_id]*scale)
				else:
					Md += list(base_m[test_id]*scale)
		#Mr,Mc,Md = shorten(Mr,Mc,Md)
		self.spM = sparse.coo_array((Md,(Mr,Mc)),shape=(num_dofs,num_dofs)).tocsc()
		#self.spM = sparse.csc_matrix((Md,(Mr,Mc)),shape=(num_dofs,num_dofs))
		#self.spM = sparse.csc_matrix(self.M)

	def projection(self):
		self._build_mass()
		self._build_force(proj=True)
		LHS = self.spC.T @ self.spM @ self.spC
		RHS = self.spC.T.dot(self.F_proj - self.spM.dot(self.dirichlet))
		x_proj,conv = sla.cg(LHS,RHS,rtol=1e-14)
		assert conv==0
		self.U_proj = self.spC.dot( x_proj) + self.dirichlet
		self._solved = True
		return x_proj

	def laplace(self):
		self._build_stiffness()
		if self.ffunc is None:
			raise ValueError('f not set, call .add_force(func)')
		self._build_force()
		LHS = self.spC.T @ self.spK @ self.spC
		RHS = self.spC.T.dot(self.F - self.spK.dot( self.dirichlet))
		x_lap,conv = sla.cg(LHS,RHS,rtol=1e-14)
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
		self._update_dirichlet()

	def _update_dirichlet(self):
		for dof_id in self.mesh.boundaries:
			dof = self.mesh.dofs[dof_id]
			x,y,z = dof.x,dof.y,dof.z
			self.dirichlet[dof_id] = self.ufunc(x,y,z)

	def vis_constraints(self):
		fig,ax = plt.subplots(2,3,figsize=(24,16))
		markers = np.array([['s','^'],['v','o']])
		cols = {1/8:'C0',3/8:'C1',-1:'C2',1/4:'C3',3/4:'C4'}
		flags = {1/8:False,3/8:False,-1:False,1/4:False,3/4:False}
		labs = {1/8:'1/8',3/8:'3/8',-1:'-1',1/4:'1/4',3/4:'3/4'}
		axshow = []
		for ind,b in enumerate(self.Id):
			if b:
				row = self.C_full[ind]
				dof = self.mesh.dofs[ind]
				x,y,z = dof.x,dof.y,dof.z
				axind = int(z>.75)
				if dof.h==self.h:
					for cind,val in enumerate(row):
						if abs(val)>1e-12:
							cdof = self.mesh.dofs[cind]
							cx,cy,cz = cdof.x,cdof.y,cdof.z
							if cdof.h!=dof.h or val==-1:
								if cx-x > .5: tx=cx-1
								elif x-cx>.5: tx=cx+1
								else: tx=cx
								if cy-y > .5: ty=cy-1
								elif y-cy>.5: ty=cy+1
								else: ty=cy
								if cz-z > .5: tz=cz-1
								elif z-cz>.5: tz=cz+1
								else: tz=cz
								
								if tx == x: ax2 = 1
								elif tx < x: ax2 = 0
								else: ax2 = 2
								m = markers[int(ty==cy),int(tz==cz)]
								if True:#for ax2 in ax2ind:
									ax[axind,ax2].scatter([y],[z],color='k',marker='o')
									ax[axind,ax2].scatter([ty],[tz],color='k',marker=m)
									if flags[val]==False:
										ax[axind,ax2].plot([y,ty],[z,tz],color=cols[val],label=labs[val])
										flags[val] = True
										if [axind,ax2] not in axshow:
											axshow.append([axind,ax2])
									else:
										ax[axind,ax2].plot([y,ty],[z,tz],color=cols[val])
							else:
								assert val==1
							
							
					
					
		for inds in axshow:
			ax[inds[0],inds[1]].legend()
		plt.show()
		return
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

	def vis_dof_sol(self,proj=False,err=False,fltr=False,fval=.9,dsp=False,myU=None,onlytrue=False):
		U = self.U_proj if proj else self.U_lap
		U = myU if myU is not None else U
		id0,x0,y0,z0,c0 = [],[], [], [], []
		id1,x1,y1,z1,c1 = [],[], [], [], []
		for dof in self.mesh.dofs.values():
			if onlytrue and dof.ID in self.true_dofs:
				if dof.h == self.h:
					id1.append(dof.ID)
					x1.append(dof.x)
					y1.append(dof.y)
					z1.append(dof.z)
					val = U[dof.ID]
					if err: val = abs(val-self.ufunc(dof.x,dof.y,dof.z))
					c1.append(val)


				else:
					id0.append(dof.ID)
					x0.append(dof.x)
					y0.append(dof.y)
					z0.append(dof.z)
					val = U[dof.ID]
					if err: val = abs(val-self.ufunc(dof.x,dof.y,dof.z))
					c0.append(val)
		
		m = ['o' for v in c1]+['^' for v in c0]
		
		if fltr:
			mx = max(c0)
			msk = np.array(c0)>fval*mx
			id0 = np.array(id0)[msk]
			x0 = np.array(x0)[msk]
			y0 = np.array(y0)[msk]
			z0 = np.array(z0)[msk]
			c0 = np.array(c0)[msk]
			vals = np.array([x0,y0,z0,c0]).T
			if dsp:print(vals)

			mx = max(c1)
			msk = np.array(c1)>fval*mx
			id1 = np.array(id1)[msk]
			x1 = np.array(x1)[msk]
			y1 = np.array(y1)[msk]
			z1 = np.array(z1)[msk]
			c1 = np.array(c1)[msk]
			vals = np.array([x1,y1,z1,c1]).T
			if dsp:print(vals)
			


		fig = plt.figure(figsize=plt.figaspect(1))
		ax = fig.add_subplot(projection='3d')
		plot1 = ax.scatter(x0,y0,z0,marker='^',c=c0,cmap='jet')
		fig.colorbar(plot1,location='left')
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		ax.set_zlim(-1.5*self.h,1+1.5*self.h)
		plt.show()

		fig = plt.figure(figsize=plt.figaspect(1))
		ax = fig.add_subplot(projection='3d')
		plot2 = ax.scatter(x1,y1,z1,marker='o',c=c1,cmap='jet')
		fig.colorbar(plot2,location='left')
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		ax.set_zlim(-1.5*self.h,1+1.5*self.h)
		plt.show()

		if fltr and dsp: return id0,id1

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

	def error(self,qpn=5,proj=False,weights=None):
		if weights is None:
			assert self._solved
			if proj:
				assert self.U_proj is not None
				weights = self.U_proj
			else:
				assert self.U_lap is not None
				weights = self.U_lap
		g_reg,g_interface,g_p,g_w = self.quad_vals
			
		l2_err = 0.
		for e in self.mesh.elements:
			uh_vals = 0
			for local_id,dof in enumerate(e.dof_list):
				if e.interface:
					vals = g_interface[e.side][local_id]
				else:
					vals = g_reg[local_id]
				uh_vals += weights[dof.ID]*vals
			x0,x1,y0,y1,z0,z1 = e.dom
			u_vals = gauss_vals(self.ufunc,x0,x1,y0,y1,z0,z1,self.qpn,g_p)
			v = super_quick_gauss_error(u_vals,uh_vals,x0,x1,y0,y1,z0,z1,self.qpn,g_w)
			l2_err += v
		return np.sqrt(l2_err)
