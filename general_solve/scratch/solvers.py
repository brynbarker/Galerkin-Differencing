import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from general_solve.grids import Mesh,SideCenteredUGrid, SideCenteredVGrid, CellCenteredGrid, NodeCenteredGrid
from general_solve.helpers import local_stiffness, local_mass, gauss
from general_solve.shape_functions import phi1, phi3, phi1_2d_ref, phi3_2d_ref, phi1_2d_eval, phi3_2d_eval


class Solver:
	def __init__(self,N,p,u,f=None,qpn=5,meshtype=Mesh):
		self.N = N
		self.p = p
		self.ufunc = u
		self.ffunc = f #needs to be overwritten 
		self.qpn = qpn

		self.mesh = meshtype(N,p)
		self.h = self.mesh.h

		self.C = None
		self.Id = None

		self.U_lap = None
		self.U_proj = None

		self._solved = False

		self._set_shape_functions()

	def _set_shape_functions(self):
		if self.p==1:
			self.phi = phi1
			self.phi_ref = phi1_2d_ref
			self.phi_eval = phi1_2d_eval
			self.id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}
		if self.p==3:
			self.phi = phi3
			self.phi_ref = phi3_2d_ref
			self.phi_eval = phi3_2d_eval
			self.id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	def _build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		F = np.zeros(num_dofs)
		myfunc = self.ufunc if proj else self.ffunc

		for e in self.mesh.elements:

			for test_id,dof in enumerate(e.dof_list):

				test_ind = self.id_to_ind[test_id]
				phi_test = lambda x,y: self.phi_ref(x,y,e.h,test_ind)
				func = lambda x,y: phi_test(x,y) * myfunc(x+e.x,y+e.y)
				val = gauss(func,0,e.h,0,e.h,self.qpn)

				F[dof.ID] += val

		if proj:
			self.F_proj = F
		else:
			self.F = F

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		base_k = local_stiffness(self.p,self.h,qpn=self.qpn)

		for e in self.mesh.elements:
			for test_id,dof in enumerate(e.dof_list):
				self.K[dof.ID,e.dof_ids] += base_k[test_id]


	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		base_m = local_mass(self.p,self.h,qpn=self.qpn)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			for test_id,dof in enumerate(e.dof_list):
				self.M[dof.ID,e.dof_ids] += base_m[test_id]*scale

	def _setup_constraints(self):
		raise ValueError("needs to be overwritten")
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter,f_inter = self.mesh.interface
		self.Id[f_inter] = 1
		self.C_full[f_inter] *= 0

        # collocated are set to the coarse node
		self.C_full[f_inter[::2],c_inter[:]] = 1

		f_odd = f_inter[1::2]

		v0, v1 = phi1(1/2,1), phi1(-1/2,1)

		for v, offset in zip([v0,v1],[0,-1]):#ind,ID in enumerate(f_odd):
			self.C_full[f_inter[1::2],np.roll(c_inter,offset)] = v

		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		for level in range(2):
			# lower are true dofs, upper are ghosts
			b,t = np.array(self.mesh.periodic[level]).reshape((2,-1))
			ghost_list = t.copy()
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			self.C_full[t,b] = 1.

			if level == 1:
				self.C_full[t[0],:] = self.C_full[b[0],:]

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self):
		fig,ax = plt.subplots(1,figsize=(24,16))
		markers = np.array([['s','^'],['v','o']])
		cols = {1/8:'C0',3/8:'C1',-1:'C2',1/4:'C3',3/4:'C4'}
		flags = {1/8:False,3/8:False,-1:False,1/4:False,3/4:False}
		labs = {1/8:'1/8',3/8:'3/8',-1:'-1',1/4:'1/4',3/4:'3/4'}
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

	def vis_periodic(self,gridtype):
		fig = vis_periodic(self.C_full,self.mesh.dofs,gridtype)
		return fig

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

	def vis_dof_sol(self,proj=False,err=False,fltr=False,fval=.9,dsp=False,myU=None,onlytrue=False):
		U = self.U_proj if proj else self.U_lap
		U = myU if myU is not None else U
		id0,x0,y0,c0 = [],[], [], []
		id1,x1,y1,c1 = [],[], [], []
		for dof in self.mesh.dofs.values():
			if onlytrue and dof.ID in self.true_dofs:
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
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		#plt.show()

		plot2 = ax[1].scatter(x1,y1,marker='o',c=c1,cmap='jet')
		fig.colorbar(plot2,location='left')
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		ax.set_zlim(-1.5*self.h,1+1.5*self.h)
		plt.show()

		if fltr and dsp: return id0,id1


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
			center = [(e.dom[1]-e.dom[0])/2,(e.dom[-1]-e.dom[-2])/2]
			plt.plot(center[0],center[1],'k.')
			for c,dof in enumerate(e.dof_list):
				m = 'o'
				myx,myy = dof.x,dof.y
				if dof.x - center[0] > 3*e.h: myx -= 1
				elif center[0] - dof.x > 3*e.h: myx += 1
				if dof.y - center[1] > 3*e.h: myy -= 1
				elif center[1] - dof.y > 3*e.h: myy += 1
				if myx != dof.x or myy != dof.y: m='s'

				plt.plot([center[0],myx],[center[1],myy],'C'+str(c))
				plt.plot([myx],[myy],'C'+str(c),marker=m)

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

	def sol(self, weights=None,proj=False):

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

			val = 0
			for local_id, dof in enumerate(e.dof_list):
				local_ind = self.id_to_ind[local_id]
				val += weights[dof.ID]*self.phi_eval(x,y,dof.h,dof.x,dof.y)
			
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

	def laplace(self):
		if self.ffunc is None:
			raise ValueError('f not set, call .add_force(func)')
		self._build_stiffness()
		self._build_force()
		LHS = self.C.T @ self.K @ self.C
		RHS = self.C.T @ (self.F - self.K @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U_lap = self.C @ x + self.dirichlet
		self._solved = True
		return x

	def projection(self):
		self._build_mass()
		self._build_force(proj=True)
		LHS = self.C.T @ self.M @ self.C
		RHS = self.C.T @ (self.F - self.M @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U_proj = self.C @ x + self.dirichlet
		self._solved = True
		return x

class CellCenteredSolver(Solver):
	def __init__(self,N,p,u,f=None,qpn=5):
		super().__init__(N,p,u,f,qpn,meshtype=CellCenteredGrid)

	def _setup_constraints(self):
		return super()._setup_constraints()
	
class NodeCenteredSolver(Solver):
	def __init__(self,N,p,u,f=None,qpn=5):
		super().__init__(N,p,u,f,qpn,meshtype=NodeCenteredGrid)
	pass

class SideCenteredUSolver(Solver):
	def __init__(self,N,p,u,f=None,qpn=5):
		super().__init__(N,p,u,f,qpn,meshtype=SideCenteredUGrid)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)


		c_inter = self.mesh.interface[0]
		f_inter = self.mesh.interface[1]
		self.Id[f_inter] = 1
		self.C_full[f_inter] *= 0

		if self.p == 1:
			v1, v2 = phi1

		if self.p == 3:
			v1, v3, v5, v7 = phi3(1/4,1), phi3(3/4,1), phi3(5/4,1), phi3(7/4,1)
		
			self.C_full[f_inter[1:-1:2],c_inter[:-3]] = v5
			self.C_full[f_inter[1:-1:2],c_inter[1:-2]] = v1
			self.C_full[f_inter[1:-1:2],c_inter[2:-1]] = v3
			self.C_full[f_inter[1:-1:2],c_inter[3:]] = v7

			self.C_full[f_inter[2::2],c_inter[:-3]] = v7
			self.C_full[f_inter[2::2],c_inter[1:-2]] = v3
			self.C_full[f_inter[2::2],c_inter[2:-1]] = v1
			self.C_full[f_inter[2::2],c_inter[3:]] = v5

		for level in range(2):
			b0,b1,B2,B3,T0,T1,t2,t3 = np.array(self.mesh.periodic[level]).reshape((8,-1))
			ghost_list = np.hstack((b0,b1,t2,t3))
			self.mesh.periodic_ghost.append(ghost_list)
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds,ds = [T0,T1,B2,B3],[b0,b1,t2,t3]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [1,-2]:
					if level == 0:
						self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					if level ==1:
						self.C_full[d[ind],:] = self.C_full[D[ind],:]

		self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]
	pass

class SideCenteredVSolver(Solver):
	def __init__(self,N,p,u,f=None,qpn=5):
		super().__init__(N,p,u,f,qpn,meshtype=SideCenteredVGrid)
	pass