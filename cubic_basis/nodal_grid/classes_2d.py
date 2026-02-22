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
		self.interface = False
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
		self.interface = [[],[],[],[]]
		self.periodic = [[],[]]
		
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

				# if x==0.:
					# self.boundaries.append(dof_id)
				if y < 2*H or y > 1.-2*H:
					self.periodic[0].append(dof_id)
				
				if (x == 0.5) and (0 <= y < 1):
					self.interface[0].append(dof_id)
				if (x == 0) and (0 <= y < 1):
					self.interface[2].append(dof_id)

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

				# if x==1.:# and (0 <= y < 1):#or y==0. or y==1:
					# self.boundaries.append(dof_id)
				if y < 2*H or y > 1.-2*H:
					self.periodic[1].append(dof_id)
				if (x == 0.5) and (0 <= y < 1):
					self.interface[1].append(dof_id)
				if (x == 1) and (0 <= y < 1):
					self.interface[3].append(dof_id)

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
		self.G = None

	def _build_force(self):
		num_dofs = len(self.mesh.dofs)
		self.F = np.zeros(num_dofs+1)

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

		for e in self.mesh.elements:

			for test_id,dof in enumerate(e.dof_list):

				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi3_2d_ref(x,y,e.h,test_ind)
				func = lambda x,y: phi_test(x,y) * self.ffunc(x+e.x,y+e.y)
				val = gauss(func,0,e.h,0,e.h,self.qpn)

				self.F[dof.ID] -= val
	
	def _build_zero_mean(self):
		num_dofs = len(self.mesh.dofs)
		self.G = np.zeros(num_dofs)

		base_g = local_zero_mean(self.h,qpn=self.qpn)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			for test_id,dof in enumerate(e.dof_list):
				self.G[dof.ID] += base_g[test_id] * scale

	def _build_stiffness(self):
		if self.G is None:
			self._build_zero_mean()
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs+1,num_dofs+1))

		base_k = local_stiffness(self.h)

		for e in self.mesh.elements:
			for test_id,dof in enumerate(e.dof_list):
				self.K[dof.ID,e.dof_ids] += base_k[test_id]
		self.K[:-1,-1] = self.G
		self.K[-1,:-1] = self.G[:]


	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))
		
		base_m = local_mass(self.h,qpn=self.qpn)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			for test_id,dof in enumerate(e.dof_list):
				self.M[dof.ID,e.dof_ids] += base_m[test_id] * scale


	def _setup_constraints(self,alt=0):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs+1)
		self.C_full = np.eye(num_dofs+1)
		self.dirichlet = np.zeros(num_dofs+1)


		for k in range(2):
			c_inter,f_inter = self.mesh.interface[2*k:2*k+2]
			self.Id[f_inter] = 1
			self.C_full[f_inter] *= 0

  			# collocated are set to the coarse node
			self.C_full[f_inter[::2],c_inter[:]] = 1

			v1, v3 = phi3(1/2,1), phi3(3/2,1)

			for v, offset in zip([v3,v1,v1,v3],[1,0,-1,-2]):#ind,ID in enumerate(f_odd):
				self.C_full[f_inter[1::2],np.roll(c_inter,offset)] = v
			

    
		# for dof_id in self.mesh.boundaries:
		# 	self.C_full[dof_id] *= 0
		# 	self.Id[dof_id] = 1.
		# 	x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
		# 	self.dirichlet[dof_id] = self.ufunc(x,y)

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
		fig,ax = plt.subplots(1,figsize=(16,24))
		markers = np.array([['s','^'],['v','o']])
		cols = {-1/16:'C0',9/16:'C1',1:'C2',1/4:'C3',3/4:'C4'}
		flags = {-1/16:False,9/16:False,1:False,1/4:False,3/4:False}
		labs = {-1/16:'-1/16',9/16:'9/16',1:'1',1/4:'1/4',3/4:'3/4'}
		axshow = []
		for ind,b in enumerate(self.Id):
			if b:
				row = self.C_full[ind]
				dof = self.mesh.dofs[ind]
				x,y = dof.x,dof.y
				if dof.h==self.h:
					for cind,val in enumerate(row):
						if abs(val)>1e-12:
							cdof = self.mesh.dofs[cind]
							cx,cy = cdof.x,cdof.y
							if cdof.h!=dof.h or val==-1:
								if cx-x > .5: tx=cx-1
								elif x-cx>.5: tx=cx+1
								else: tx=cx
								if cy-y > .5: ty=cy-1
								elif y-cy>.5: ty=cy+1
								else: ty=cy

								if tx==x: tx-=self.h
								
								m = markers[int(ty==cy),int(tx==cx)]
								ax.scatter([x],[y],color='k',marker='o')
								ax.scatter([tx],[ty],color='k',marker=m)
								if flags[val]==False:
									ax.plot([x,tx],[y,ty],color=cols[val],label=labs[val])
									flags[val] = True
								else:
									ax.plot([x,tx],[y,ty],color=cols[val])
							else:
								assert val==1
							
							
					
					
		ax.legend()
		plt.show()
		return

	#def vis_constraints(self):
	#	if self.C is not None:
	#		vis_constraints(self.C,self.mesh.dofs)
	#	else:
	#		print('Constraints have not been set')
	def vis_dof_sol(self,proj=False,err=False,fltr=False,fval=.9,dsp=False,myU=None,onlytrue=False):
		U = self.U
		if myU is not None: U=myU
		id0,x0,y0,c0 = [],[], [], []
		id1,x1,y1,c1 = [],[], [], []
		for dof in self.mesh.dofs.values():
			#if onlytrue and dof.ID in self.true_dofs:
			if dof.ID in self.true_dofs:
				if dof.h == self.h:
					id1.append(dof.ID)
					x1.append(dof.x)
					y1.append(dof.y)
					val = U[dof.ID]
					if err: val = abs(val-self.ufunc(dof.x,dof.y))
					c1.append(val)


				else:
					id0.append(dof.ID)
					x0.append(dof.x)
					y0.append(dof.y)
					val = U[dof.ID]
					if err: val = abs(val-self.ufunc(dof.x,dof.y))
					c0.append(val)
		
		m = ['o' for v in c1]+['^' for v in c0]
		
		if fltr:
			mx = max(c0)
			msk = np.array(c0)>fval*mx
			id0 = np.array(id0)[msk]
			x0 = np.array(x0)[msk]
			y0 = np.array(y0)[msk]
			c0 = np.array(c0)[msk]
			vals = np.array([x0,y0,c0]).T
			if dsp:print(vals)

			mx = max(c1)
			msk = np.array(c1)>fval*mx
			id1 = np.array(id1)[msk]
			x1 = np.array(x1)[msk]
			y1 = np.array(y1)[msk]
			c1 = np.array(c1)[msk]
			vals = np.array([x1,y1,c1]).T
			if dsp:print(vals)
			


		fig,ax = plt.subplots(1,2,figsize=(10,5))
		plot1 = ax[0].scatter(x0,y0,marker='^',c=c0,cmap='jet')
		fig.colorbar(plot1,location='left')
		#ax[0].set_xlim(-1.5*self.h,.5+1.5*self.h)
		#ax[0].set_ylim(-1.5*self.h,1+1.5*self.h)
		#plt.show()

		plot2 = ax[1].scatter(x1,y1,marker='o',c=c1,cmap='jet')
		fig.colorbar(plot2,location='left')
		#ax[1].set_xlim(-1.5*self.h,1+1.5*self.h)
		#ax[1].set_ylim(-1.5*self.h,1+1.5*self.h)
		plt.show()

		if fltr and dsp: return id0,id1

	def vis_mesh(self,corner=False,retfig=False):

		fig = plt.figure()
		mk = ['^','o']
		for ind,dof in enumerate(self.mesh.dofs.values()):
			m = mk[dof.h==self.mesh.h]
			c = 'C0' if ind in self.true_dofs else 'C1'
			cind = 2*(ind in self.true_dofs)+((ind in self.true_dofs)==(dof.h==self.mesh.h))
			c = 'C'+str(cind)
			alpha = 1 if ind in self.true_dofs else .5
			plt.scatter(dof.x,dof.y,marker=m,color=c,alpha=alpha)

		plt.show()
		return

	def vis_dofs(self):
		frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
		fig,ax = plt.subplots(figsize=(10,10))
		ax.set_xlim(-3*self.h,1+3*self.h)
		ax.set_ylim(-3*self.h,1+3*self.h)

		size = 16#int((self.p+1)**2)
	
		line, = ax.plot(frame[0],frame[1],'lightgrey')
		blocks = []
		for _ in range(size):
			block, = ax.plot([],[])
			blocks.append(block)
		dot, = ax.plot([],[],'ko',linestyle='None')

		def update(n):
			dof = self.mesh.dofs[n]
			els = list(dof.elements.values())
			line.set_data(frame[0],frame[1])
			dot.set_data(dof.x,dof.y)
			for i in range(size):
				if i < len(dof.elements):
					e = els[i]
					blocks[i].set_data(e.plot[0],e.plot[1])
				else:
					blocks[i].set_data([],[])
			return [line,blocks,dot]
		interval = 400
		ani = FuncAnimation(fig, update, frames=len(self.mesh.dofs), interval=interval)
		plt.close()
		return HTML(ani.to_html5_video())
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
		fig,ax = plt.subplots(figsize=(10,10))
		ax.set_xlim(-3*self.h,1+3*self.h)
		ax.set_ylim(-3*self.h,1+3*self.h)
	
		line, = ax.plot(frame[0],frame[1],'lightgrey')
		eline, = ax.plot([],[])
		dot, = ax.plot([],[],'ko',linestyle='None')

		def update(n):
			e = self.mesh.elements[n]
			line.set_data(frame[0],frame[1])
			eline.set_data(e.plot[0],e.plot[1])
			xs,ys = [],[]
			for dof in e.dof_list:
				xs.append(dof.x)
				ys.append(dof.y)
			dot.set_data(xs,ys)
			return [line,eline,dot]
			#	if i < len(e.dof_list):
			#		blocks[i].set_data(e.dof_list[i].x,e.dof_list.y
			#		blocks[i].set_data(blocks_n[i][0],blocks_n[i][1])
			#	else:
			#		blocks[i].set_data([],[])
			#if yesdot: dot.set_data(dots_n[0],dots_n[1])
			#to_return = [line]+blocks
			#if yesdot: to_return += [dot]
			#return to_return
		interval = 400
		ani = FuncAnimation(fig, update, frames=len(self.mesh.elements), interval=interval)
		plt.close()
		return HTML(ani.to_html5_video())
		frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
		data = []
		for e in self.mesh.elements:
			center = [(e.dom[1]+e.dom[0])/2,(e.dom[-1]+e.dom[-2])/2]
			plt.plot(e.plot[0],e.plot[1],'grey')
			plt.plot(center[0],center[1],'k.')


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

class Laplace(Solver):
	def __init__(self,N,u,f,qpn=5):
		super().__init__(N,u,f,qpn)
		self._setup_constraints()

	def solve(self):
		self._build_stiffness()
		self._build_force()
		LHS = self.C.T @ self.K @ self.C
		RHS = self.C.T @ (self.F )#- self.K @ self.dirichlet)
		#LHS = self.C_rect.T @ self.K @ self.C_rect
		#RHS = self.C_rect.T @ (self.F - self.K @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C @ x #+ self.dirichlet
		self.solved = True


class Projection(Solver):
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


