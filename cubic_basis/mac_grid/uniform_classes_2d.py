import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import sparse

from cubic_basis.mac_grid.classes_2d import Node, Element, Mesh, Solver
from cubic_basis.mac_grid.helpers_2d import gauss,local_stiffness
from cubic_basis.mac_grid.shape_functions_2d import phi3,phi3_2d_ref


class UniformMesh(Mesh):
	def __init__(self,N):
		super().__init__(N)
		self.h = self.h*2

	def _make_coarse(self): # overwritten
		self.interface[0] = [[],[]]
		self.periodic[0] = []
		H = 1/self.N
		xdom = np.linspace(0-H,1+H,self.N+3)
		ydom = np.linspace(0-3*H/2,1+3*H/2,self.N+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0<=x<1) and (-H<y<1.):###
					strt = dof_id-1-xlen
					element = Element(e_id,j-1,i-1,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1
				
				### UNDO THIS
				#if x==H:
				if j == 0:# or i == 0:
					self.boundaries.append(dof_id)


				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self): # overwritten
		return

class UniformSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=UniformMesh)

	def _build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		F = np.zeros(num_dofs)
		myfunc = self.ufunc if proj else self.ffunc

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

		for e in self.mesh.elements:
			y0,y1 = e.dom[2]-e.y, e.dom[3]-e.y
			for test_id,dof in enumerate(e.dof_list):
				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi3_2d_ref(x,y,e.h,test_ind)
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

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		
		base_k = local_stiffness(self.h,qpn=self.qpn)
	
		interface_k0 = local_stiffness(self.h,qpn=self.qpn,y1=.5)
		interface_k1 = local_stiffness(self.h,qpn=self.qpn,y0=.5)
		interface_k = [interface_k0,interface_k1]


		for e in self.mesh.elements:
			for test_id,dof in enumerate(e.dof_list):
				if e.interface:
					self.K[dof.ID,e.dof_ids] += interface_k[e.side][test_id]
				else:
					self.K[dof.ID,e.dof_ids] += base_k[test_id]
		self.spK = sparse.csc_matrix(self.K)

	def _setup_constraints(self): # overwritten
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)



		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self,retfig=False):
		fig = vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[1],'horiz')
		if retfig: return fig

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('horiz')
		if retfig: return fig

	def xy_to_e(self,x,y): # over_written
		
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		x_ind = int(x/self.h)
		y_ind = int(y/self.h+.5)
		el_ind = y_ind*self.N+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		return e


