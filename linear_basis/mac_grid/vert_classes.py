import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import sparse

from linear_basis.mac_grid.classes import Node, Element, Mesh, Solver
from linear_basis.mac_grid.helpers import indswap, inddel 


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
				
					if x==1 or y < 0 or y > 1:
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

					if z==1-3*H/2:
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
		self.dirichlet = np.zeros(num_dofs)

		Cr, Cc, Cd = [],[],[]

		c_inter = np.array(self.mesh.interface[0])
		f_inter = np.array(self.mesh.interface[1])
		for j in range(2):

			cgrid = c_inter[j].reshape((2,self.N+2,self.N+1))
			fgrid = f_inter[j].reshape((2,2*self.N+2,2*self.N+1))


			self.Id[fgrid[1-j].flatten()] = 1
			
			findsa = fgrid[1-j]
			findsb = fgrid[j]
			Cr += list(findsa.flatten())
			Cc += list(findsb.flatten())
			Cd += [-1]*findsa.size

			for k in range(2):
				for i in range(2):
					for l in range(2):
						end = None if l else -1
						val = 3/4 if i==l else 1/4
						finds0 = fgrid[1-j,i::2,::2]
						cinds0 = cgrid[k,l:end,:]

						Cr += list(finds0.flatten())
						Cc += list(cinds0.flatten())
						Cd += [val]*finds0.size

						for m in range(2):
							endx = None if m else -1
							finds1 = fgrid[1-j,i::2,1::2]
							cinds1 = cgrid[k,l:end,m:endx]
							Cr += list(finds1.flatten())
							Cc += list(cinds1.flatten())
							Cd += [val/2]*finds1.size

		dL,DL,maskL = np.array(self.mesh.periodic).T
		for (d,D,mask) in zip(dL,DL,maskL):
			if True:#mask:
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

		spC_full = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_dofs))
		c_data = {}
		for i,r in enumerate(spC_full.row):
			tup = (spC_full.col[i],spC_full.data[i])
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
		for g_id in self.C_full:
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
						assert tz==z
						m = markers[int(tx==x),int(ty==y)]
						if y==ty:
							yshft = self.h/10 if tx>x else -self.h/10
						else:
							yshft = 0
						if x==tx:
							xshft = self.h/10 if ty>y else -self.h/10
						else:
							xshft = 0
						ax[axind].scatter([x+xshft],[y+yshft],color='k',marker=m)
						ax[axind].scatter([tx+xshft],[ty+yshft],color='k',marker='o')
						ax[axind].plot([x+xshft,tx+xshft],[y+yshft,ty+yshft])#,color=colors[val])
		ax[0].set_title('coarse')
		ax[0].set_ylabel('y')
		ax[0].set_xlabel('x')
		ax[1].set_title('fine')
		ax[1].set_ylabel('y')
		ax[1].set_xlabel('x')
		if retfig: return fig
		plt.show()
		return

	def vis_constraints(self):
		fig,ax = plt.subplots(2,3,figsize=(24,16))
		markers = np.array([['s','^'],['v','o']])
		cols = {1/8:'C0',3/8:'C1',-1:'C2',1/4:'C3',3/4:'C4'}
		flags = {1/8:False,3/8:False,-1:False,1/4:False,3/4:False}
		labs = {1/8:'1/8',3/8:'3/8',-1:'-1',1/4:'1/4',3/4:'3/4'}
		axshow = []
		for g_id in self.C_full:#c_data:
			if len(self.C_full[g_id]) > 1:
				g_dof = self.mesh.dofs[g_id]
				assert self.h==g_dof.h
				x,y,z = g_dof.x,g_dof.y,g_dof.z
				axind = int(z>.75)
				for t_id,val in self.C_full[g_id]:
					t_dof = self.mesh.dofs[t_id]
					assert t_dof.h!=self.h or val==-1
					cx,cy,cz = t_dof.x,t_dof.y,t_dof.z

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
					ax[axind,ax2].scatter([y],[z],color='k',marker='o')
					ax[axind,ax2].scatter([ty],[tz],color='k',marker=m)
					if flags[val]==False:
						ax[axind,ax2].plot([y,ty],[z,tz],color=cols[val],label=labs[val])
						flags[val] = True
						if [axind,ax2] not in axshow:
							axshow.append([axind,ax2])
					else:
						ax[axind,ax2].plot([y,ty],[z,tz],color=cols[val])

		for inds in axshow:
			ax[inds[0],inds[1]].legend()
		ttl = ['z = 0','z = .5']
		for i in range(2):
			for j in range(3):
				ax[i,j].set_title(ttl[i])
				ax[i,j].set_ylabel('z')
				ax[i,j].set_xlabel('y')
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


