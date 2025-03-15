import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import sparse

from linear_basis.mac_grid.classes import Node, Element, Mesh, Solver
from linear_basis.mac_grid.helpers import vis_constraints, indswap, inddel 


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
		self.dof_count = dof_id


class HorizontalRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=HorizontalRefineMesh)

	def _setup_constraints(self): # overwritten
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		Cr, Cc, Cd = [],[],[]

		c_inter = np.array(self.mesh.interface[0])
		f_inter = np.array(self.mesh.interface[1])
		for j in range(2):
			self.Id[f_inter[j]] = 1

			nc = int(np.sqrt(c_inter[j].size))
			nf = int(np.sqrt(f_inter[j].size))

			cgrid = c_inter[j].reshape((nc,nc))
			fgrid = f_inter[j].reshape((nf,nf))

			frac = [.25,.75]
			ends = [-1,None]
			for csy in [0,1]:
				for csz in [0,1]:
					cinds = cgrid[csy:ends[csy],csz:ends[csz]]
					for fsy in [0,1]:
						for fsz in [0,1]:
							finds = fgrid[fsy::2,fsz::2]
							v = frac[csy==fsy]*frac[csz==fsz]
							Cr += list(finds.flatten())
							Cc += list(cinds.flatten())
							Cd += [v]*finds.size

		dL,DL,maskL = np.array(self.mesh.periodic).T
		for (d,D,mask) in zip(dL,DL,maskL):
			if mask:
				Cc = indswap(Cc,d,D)

		for (d,D,mask) in zip(dL,DL,maskL):
			self.Id[d] = 1
			Cr.append(d)
			Cc.append(D)
			Cd.append(1.)

		for dof_id in self.mesh.boundaries:
			Cr,Cc,Cd = inddel(Cr,Cc,Cd,dof_id)
			assert dof_id not in Cr
			self.Id[dof_id] = 1.
			dof = self.mesh.dofs[dof_id]
			x,y,z = dof.x,dof.y,dof.z
			self.dirichlet[dof_id] = self.ufunc(x,y,z)

		self.true_dofs = list(np.where(self.Id==0)[0])

		for true_ind in self.true_dofs:
			if true_ind not in self.mesh.boundaries:
				Cr.append(true_ind)
				Cc.append(true_ind)
				Cd.append(1.)

		self.spC_full = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_dofs))
		c_data = {}
		for i,r in enumerate(self.spC_full.row):
			tup = (self.spC_full.col[i],self.spC_full.data[i])
			if r in c_data.keys():
				c_data[r].append(tup)
			else:
				c_data[r] = [tup]
		self.C_full = c_data

		Cc_array = np.array(Cc)
		masks = []
		for true_dof in self.true_dofs:
			masks.append(Cc_array==true_dof)
		for j,mask in enumerate(masks):
			Cc_array[mask] = j
		Cc = list(Cc_array)

		num_true = len(self.true_dofs)
		self.spC = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_true)).tocsc()

	def vis_periodic(self,retfig=False):
		fig,ax = plt.subplots(1,2,figsize=(16,7))
		markers = np.array([['s','<'],['^','o']])
		colors = {1/16:'C0',3/16:'C1',9/16:'C2'}
		flags = {1/16:False,3/16:False,9/16:False}
		labs = {1/16:'1/16',3/16:'3/16',9/16:'9/16'}
		for g_id in self.C_full:#c_data:
			if len(self.C_full[g_id]) == 1:
				g_dof = self.mesh.dofs[g_id]
				axind = int(self.h==g_dof.h)
				x,y,z = g_dof.x,g_dof.y,g_dof.z
				for t_id,val in self.C_full[g_id]:
					if g_id!=t_id:
						t_dof = self.mesh.dofs[t_id]
						assert t_dof.h==g_dof.h
						assert val == 1
						tx,ty,tz = t_dof.x,t_dof.y,t_dof.z
						assert tx==x
						m = markers[int(ty==y),int(tz==z)]
						if y==ty:
							yshft = self.h/10 if tz>z else -self.h/10
						else:
							yshft = 0
						if z==tz:
							zshft = self.h/10 if ty>y else -self.h/10
						else:
							zshft = 0
						ax[axind].scatter([y+yshft],[z+zshft],color='k',marker=m)
						ax[axind].scatter([ty+yshft],[tz+zshft],color='k',marker='o')
						ax[axind].plot([y+yshft,ty+yshft],[z+zshft,tz+zshft])#,color=colors[val])
		ax[0].set_title('x = 0')
		ax[0].set_ylabel('z')
		ax[0].set_xlabel('y')
		ax[1].set_title('x = .5')
		ax[1].set_ylabel('z')
		ax[1].set_xlabel('y')
		if retfig: return fig
		plt.show()
		return

	def vis_constraints(self):
		fig,ax = plt.subplots(1,2,figsize=(16,7))
		markers = np.array([['s','^'],['v','o']])
		colors = {1/16:'C0',3/16:'C1',9/16:'C2'}
		flags = {1/16:False,3/16:False,9/16:False}
		labs = {1/16:'1/16',3/16:'3/16',9/16:'9/16'}
		for g_id in self.C_full:#c_data:
			if len(self.C_full[g_id]) > 1:
				g_dof = self.mesh.dofs[g_id]
				assert self.h==g_dof.h
				x,y,z = g_dof.x,g_dof.y,g_dof.z
				axind = int(x==.5)
				for t_id,val in self.C_full[g_id]:
					t_dof = self.mesh.dofs[t_id]
					assert t_dof.h!=self.h
					cx,cy,cz = t_dof.x,t_dof.y,t_dof.z
					if cy-y > .5: ty=cy-1
					elif y-cy>.5: ty=cy+1
					else: ty=cy
					if cz-z > .5: tz=cz-1
					elif z-cz>.5: tz=cz+1
					else: tz=cz
					m = markers[int(ty==cy),int(tz==cz)]
					ax[axind].scatter([y],[z],color='k',marker='o')
					ax[axind].scatter([ty],[tz],color='k',marker=m)
					if flags[val]==False:
						ax[axind].plot([y,ty],[z,tz],color=colors[val],label=labs[val])
						flags[val] = True
					else:
						ax[axind].plot([y,ty],[z,tz],color=colors[val])
					
		ax[0].set_title('coarse')
		ax[0].set_ylabel('z')
		ax[0].set_xlabel('y')
		ax[1].set_title('fine')
		ax[1].set_ylabel('z')
		ax[1].set_xlabel('y')
		ax[0].legend()
		plt.show()
		return

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


