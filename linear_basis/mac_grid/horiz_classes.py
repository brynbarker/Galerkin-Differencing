import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from linear_basis.mac_grid.classes import Node, Element, Mesh, Solver
from linear_basis.mac_grid.helpers import vis_constraints


class HorizontalRefineMesh(Mesh):
	def __init__(self,N):
		super().__init__(N)

	def _make_coarse(self): # overwritten
		self.interface[0] = [[],[]]
		H = self.h*2
		xdom = np.linspace(0,0.5,int(self.N/2)+1)
		ydom = np.linspace(0-H/2,1+H/2,self.N+2)
		zdom = np.linspace(0-H/2,1+H/2,self.N+2)

		xlen,ylen,zlen = len(xdom),len(ydom),len(zdom)

		dof_id,e_id = 0,0
		for k,z in enumerate(zdom):
			zinterface_element = (k==0 or k==zlen-2)
			zside = 2+(k==0)
			for i,y in enumerate(ydom):
				yinterface_element = (i==0 or i==ylen-2)
				yside = i==0
				for j,x in enumerate(xdom):
					self.dofs[dof_id] = Node(dof_id,j,i,k,x,y,z,H)

					if (0<=x<.5) and (y<1) and (z<1):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						e_id += 1
				
					if x==H:
						self.boundaries.append(dof_id)

					if y < 0 or y > 1 or z < 0 or z > 1:
						#self.boundaries.append(dof_id)
						yind = ylen-2 if i==0 else i
						yind = 1 if i==ylen-1 else yind

						zind = zlen-2 if k==0 else k
						zind = 1 if k==zlen-1 else zind
					
						fill_id = zind*(xlen*ylen)+yind*(xlen)+j
						
						self.periodic.append([dof_id,fill_id,x==0 or x==.5])

					if (x==.5 or x==0):
						self.interface[0][x==.5].append(dof_id)

					dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self): # overwritten
		self.interface[1] = [[],[]]

		H = self.h
		xdom = np.linspace(0.5,1.,self.N+1)
		ydom = np.linspace(0-H/2,1+H/2,2*self.N+2)
		zdom = np.linspace(0-H/2,1+H/2,2*self.N+2)

		xlen,ylen,zlen = len(xdom),len(ydom),len(zdom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for k,z in enumerate(zdom):
			zinterface_element = (k==0 or k==zlen-2)
			zside = 2+(k==0)
			for i,y in enumerate(ydom):
				yinterface_element = (i==0 or i==ylen-2)
				yside = i==0
				for j,x in enumerate(xdom):
					self.dofs[dof_id] = Node(dof_id,j,i,k,x,y,z,H)

					if (0.5<=x<1.) and (y<1) and (z<1):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						element.set_fine()
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						
						e_id += 1

					if (x == 0.5 or x==1.):
						self.interface[1][x==.5].append(dof_id)

					elif y < 0 or y > 1 or z < 0 or z > 1:
						yind = ylen-2 if i==0 else i
						yind = 1 if i==ylen-1 else yind

						zind = zlen-2 if k==0 else k
						zind = 1 if k==zlen-1 else zind
					
						fill_id = zind*(xlen*ylen)+yind*(xlen)+j+self.n_coarse_dofs
						
						self.periodic.append([dof_id,fill_id,False])

					dof_id += 1


class HorizontalRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=HorizontalRefineMesh)

	def _setup_constraints(self): # overwritten
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter = np.array(self.mesh.interface[0])
		f_inter = np.array(self.mesh.interface[1])
		for j in range(2):
			self.Id[f_inter[j]] = 1
			self.C_full[f_inter[j]] *= 0

			nc = int(np.sqrt(c_inter[j].size))
			nf = int(np.sqrt(f_inter[j].size))

			cgrid = c_inter[j].reshape((nc,nc))
			fgrid = f_inter[j].reshape((nf,nf))

			frac = [.75,.25]
			ends = [-1,None]
			for csy in [0,1]:
				for csz in [0,1]:
					cinds = cgrid[csy:ends[csy],csz:ends[csz]]
					for fsy in [0,1]:
						for fsz in [0,1]:
							finds = fgrid[fsy::2,fsz::2]
							v = frac[csy==fsy]*frac[csz==fsz]
							self.C_full[finds,cinds] = v

		dL,DL,maskL = np.array(self.mesh.periodic).T
		for (d,D,mask) in zip(dL,DL,maskL):
			self.Id[d] = 1
			self.C_full[d,:] = self.C_full[D,:]
			if mask:
				self.C_full[:,D] += self.C_full[:,d]
			self.C_full[:,d] *= 0

		self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			dof = self.mesh.dofs[dof_id]
			x,y,z = dof.x,dof.y,dof.z
			self.dirichlet[dof_id] = self.ufunc(x,y,z)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self,retfig=False):
		fig = vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[1],'horiz')
		if retfig: return fig

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('horiz')
		if retfig: return fig

	def xy_to_e(self,x,y,z): # over_written
		n_x_els = [self.N/2,self.N]
		n_y_els = [self.N,self.N*2]
		
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		z -= (z==1)*1e-12
		fine = True if x >= 0.5 else False
		x_ind = int((x-fine*.5)/((2-fine)*self.h))
		y_ind = int(y/((2-fine)*self.h)+.5)
		z_ind = int(z/((2-fine)*self.h)+.5)
		el_ind = fine*self.mesh.n_coarse_els+z_ind*(n_x_els[fine]*n_y_els[fine])+y_ind*n_x_els[fine]+x_ind
		e = self.mesh.elements[int(el_ind)]
		print((x,y,z),fine,(x_ind,y_ind,z_ind),(e.j,e.i,e.k),e.dom)
		assert x >= e.dom[0] and x <= e.dom[1]
		assert y >= e.dom[2] and y <= e.dom[3]
		assert z >= e.dom[4] and z <= e.dom[5]
		return e


