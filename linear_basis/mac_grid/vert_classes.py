import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from linear_basis.mac_grid.classes import Node, Element, Mesh, Solver
from linear_basis.mac_grid.helpers import vis_constraints


class VerticalRefineMesh(Mesh):
	def __init__(self,N):
		super().__init__(N)

	def _make_coarse(self): # overwritten
		self.interface[0] = [[],[]]
		H = self.h*2

		xdom = np.linspace(0,1,self.N+1)
		ydom = np.linspace(0-H/2,1+H/2,self.N+2)
		zdom = np.linspace(0-H/2,.5+H/2,int(self.N/2)+2)

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

					if (x<1) and (y<1) and (z<.5):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						e_id += 1
				
					#if z==5*H/2 or z==3*H/2:
					#	self.boundaries.append(dof_id)

					if x==1 or y < 0 or y > 1:
						#self.boundaries.append(dof_id)
						#xind = xlen-1 if j==0 else j
						#xind = 1 if j==xlen-1 else xind
						xind = 0 if j==xlen-1 else j

						yind = ylen-2 if i==0 else i
						yind = 1 if i==ylen-1 else yind
					
						fill_id = k*(xlen*ylen)+yind*(xlen)+xind
						
						self.periodic.append([dof_id,fill_id,z>.5-H or z<H])

					if (z>.5-H or z<H):
						self.interface[0][z>H].append(dof_id)

					dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self): # overwritten
		self.interface[1] = [[],[]]

		H = self.h
		xdom = np.linspace(0,1.,2*self.N+1)
		ydom = np.linspace(0-H/2,1+H/2,2*self.N+2)
		zdom = np.linspace(0.5-H/2,1+H/2,self.N+2)

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

					if (x<1.) and (y<1) and (z<1):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						element.set_fine()
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						
						e_id += 1

					if z==1-3*H:
						self.boundaries.append(dof_id)

					if (z < 0.5+H or z>1.-H):
						self.interface[1][z<1-H].append(dof_id)

					if x == 1 or y < 0 or y > 1:
						if z>.5 and z<1:
							xind = 0 if j==xlen-1 else j

							yind = ylen-2 if i==0 else i
							yind = 1 if i==ylen-1 else yind
					
							fill_id = k*(xlen*ylen)+yind*(xlen)+xind+self.n_coarse_dofs
							
							self.periodic.append([dof_id,fill_id,z<.5+H or z>1-H])

					dof_id += 1
		self.dof_count = dof_id


class VerticalRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=VerticalRefineMesh)

	def _setup_constraints(self): # overwritten
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter = np.array(self.mesh.interface[0])
		f_inter = np.array(self.mesh.interface[1])
		for j in range(2):

			cgrid = c_inter[j].reshape((2,self.N+2,self.N+1))
			fgrid = f_inter[j].reshape((2,2*self.N+2,2*self.N+1))


			self.Id[fgrid[1-j].flatten()] = 1
			self.C_full[fgrid[1-j].flatten()] *= 0
			
			self.C_full[fgrid[1-j],fgrid[j]] = -1
			for k in range(2):
				for i in range(2):
					for l in range(2):
						end = None if l else -1
						val = 3/4 if i==l else 1/4
						self.C_full[fgrid[1-j,i::2,::2],cgrid[k,l:end,:]] = val

						for m in range(2):
							endx = None if m else -1
							self.C_full[fgrid[1-j,i::2,1::2],cgrid[k,l:end,m:endx]] = val/2

		dL,DL,maskL = np.array(self.mesh.periodic).T
		for (d,D,mask) in zip(dL,DL,maskL):
			if self.Id[D]: 
				dof = self.mesh.dofs[D]
				print(dof.h==self.h,(dof.x,dof.y,dof.z),sep='\t')
			self.Id[d] = 1
			self.C_full[d,:] = self.C_full[D,:]
			if True:#mask:
				self.C_full[:,D] += self.C_full[:,d]
			self.C_full[:,d] *= 0

		self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		for dof_id in self.mesh.boundaries+list(dL):
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			dof = self.mesh.dofs[dof_id]
			x,y,z = dof.x,dof.y,dof.z
			self.dirichlet[dof_id] = self.ufunc(x,y,z)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('horiz')
		if retfig: return fig

	def xy_to_e(self,x,y,z): # over_written
		n_y_els = [self.N+1,self.N*2+1]
		
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		z -= (z==1)*1e-12
		fine = True if x >= 0.5 else False
		if fine:
			x_ind = int((x-.5)/self.h)
			y_ind = int(y/self.h+1/2)
			z_ind = int(z/self.h+1/2)
			el_ind = self.mesh.n_coarse_els+z_ind*(self.N*n_y_els[1])+y_ind*self.N+x_ind
		else:
			x_ind = int(x/2/self.h)
			y_ind = int(y/2/self.h+1/2)
			z_ind = int(z/2/self.h+1/2)
			el_ind = z_ind*(self.N/2*n_y_els[0])+y_ind*self.N/2+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert x >= e.dom[0] and x <= e.dom[1]
		assert y >= e.dom[2] and y <= e.dom[3]
		assert z >= e.dom[4] and z <= e.dom[5]
		return e


