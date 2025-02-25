import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from 2D.cubic_basis.mac_grid.classes import Node, Element, Mesh, Solver
from 2D.cubic_basis.mac_grid.helpers import vis_constraints
from 2D.cubic_basis.mac_grid.shape_functions import phi3


class VerticalRefineMesh(Mesh):
	def __init__(self,N):
		super().__init__(N)

	def _make_coarse(self):
		self.interface[0] = [[],[]]
		self.periodic[0] = []
		H = self.h*2
		xdom = np.linspace(0-H,1+H,self.N+3)
		ydom = np.linspace(0-3*H/2,0.5+3*H/2,int(self.N/2)+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (-H<y<.5) and (0<=x<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j-1,i-1,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if x < 2*H or x > 1.-2*H:
					self.periodic[0].append(dof_id)
				if (.5+2*H>y > 0.5-2*H) or (-2*H<y<2*H):# and (0 <= x < 1):
					self.interface[0][y>2*H].append(dof_id)

				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		self.interface[1] = [[],[]]
		self.periodic[1] = []
		H = self.h
		xdom = np.linspace(0-H,1+H,2*self.N+3)
		ydom = np.linspace(0.5-3*H/2,1.+3*H/2,self.N+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (.5-H<y<1.) and (0<=x<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j-1,i-1,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if y==1-5*H/2:
					self.boundaries.append(dof_id)
				if x < 2*H or x > 1.-2*H:
					self.periodic[1].append(dof_id)
				if (.5-2*H<y < 0.5+2*H) or (1+2*H>y>1-2*H):# and (0 <= x < 1):
					self.interface[1][y<1-2*H].append(dof_id)

				dof_id += 1

class VerticalRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=VerticalRefineMesh)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)


		c_inter = self.mesh.interface[0]
		f_inter = self.mesh.interface[1]

		v1, v3 = phi3(1/2,1), phi3(3/2,1)
		for j in range(2):
			c0,c1,c2,c3 = np.array(c_inter[j]).reshape((4,-1))
			f0,f1,f2,f3 = np.array(f_inter[j]).reshape((4,-1))
			if j==0:
				f = f3.copy()
				flist = [f2,f1,f0]
			else:
				f = f0.copy()
				flist = [f1,f2,f3]

			self.Id[f] = 1
			self.C_full[f] *= 0

			for (v,c) in zip([v3,v1,v1,v3],[c0,c1,c2,c3]):
				self.C_full[f[2:-1:2],c[:-3]] = v*v3/v3
				self.C_full[f[2:-1:2],c[1:-2]] = v*v1/v3
				self.C_full[f[2:-1:2],c[2:-1]] = v*v1/v3
				self.C_full[f[2:-1:2],c[3:]] = v*v3/v3
			for (v,fdof) in zip([v1,v1,v3],flist):
				self.C_full[f[2:-1:2],fdof[2:-1:2]] = -v/v3

			for (v,c) in zip([v3,v1,v1,v3],[c0,c1,c2,c3]):
				self.C_full[f[1::2],c[1:-1]] = v/v3
			for (v,fdof) in zip([v1,v1,v3],flist):
				self.C_full[f[1::2],fdof[1::2]] = -v/v3

		# periodic
		for level in range(2):
			# lower case are ghosts, upper case are true dofs
			b0,B1,B2,T0,t1,t2 = np.array(self.mesh.periodic[level]).reshape((-1,6)).T
			ghost_list = np.array([b0,t1,t2])
			self.mesh.periodic_ghost.append(ghost_list)
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds,ds = [T0,B1,B2],[b0,t1,t2]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [0,1,2,3,-4,-3,-2,-1]:
					self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					self.C_full[d[ind],:] = self.C_full[D[ind],:]

		#self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self,retfig=False):
		fig = vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[1],'vert')
		if retfig: return fig

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('vert')
		if retfig: return fig

	def xy_to_e(self,x,y):
		n_x_els = [self.N,2*self.N]
		
		x -= (x==1)*1e-14
		y -= (y==1)*1e-14
		fine = True if y >= 0.5 else False

		if fine:
			y_ind = int((y-.5)/self.h+1/2)
			x_ind = int(x/self.h)
		else:
			y_ind = int(y/2/self.h+1/2)
			x_ind = int(x/2/self.h)
		el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		return e

