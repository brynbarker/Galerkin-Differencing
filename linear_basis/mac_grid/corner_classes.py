import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import sparse

from linear_basis.mac_grid.classes import Node, Element, Mesh, Solver
from linear_basis.mac_grid.helpers import inddel, full_swap_and_set

class CornerRefinementMesh(Mesh):
	def __init__(self,N):
		self.interface_f_dofs = [[],[]]
		super().__init__(N)

	def _make_coarse(self):
		self._make_q0()
		self._make_q1()
		self._make_q2()

	def _make_fine(self):
		self._make_q3()

	def _make_q0(self): #coarse
		self.interface[0] = [[],[],[],[]]
		H = self.h*2
		xdom = np.linspace(0,.5,int(self.N/2)+1)
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)
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
					self.dofs[dof_id] = Node(dof_id,j,i,k,x,y,z,H,0)

					if (x<.5) and (y<.5) and (z<1):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						e_id += 1

					if x==H:
						self.boundaries.append(dof_id)
					if (x == 0.5 or x==0):
						self.interface[0][x>0].append(dof_id)
					if (y > 0.5-H or y<H):
						self.interface[0][2+(y>.5-H)].append(dof_id)
					if z < 0 or z > 1:
						zind = zlen-2 if k==0 else k
						zind = 1 if k==zlen-1 else zind
					
						fill_id = zind*(xlen*ylen)+i*(xlen)+j
						self.periodic.append([dof_id,fill_id,x==0 or x==.5 or y>.5-H or y<H])

					dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q1(self): #coarse
		self.interface[1] = [[],[],[],[]]
		H = self.h*2
		xdom = np.linspace(.5,1,int(self.N/2)+1)
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)
		zdom = np.linspace(0-H/2,1+H/2,self.N+2)

		xlen,ylen,zlen = len(xdom),len(ydom),len(zdom)

		n_prev_dofs = self.dof_count
		dof_id,e_id = self.dof_count, self.el_count
		for k,z in enumerate(zdom):
			zinterface_element = (k==0 or k==zlen-2)
			zside = 2+(k==0)
			for i,y in enumerate(ydom):
				yinterface_element = (i==0 or i==ylen-2)
				yside = i==0
				for j,x in enumerate(xdom):
					self.dofs[dof_id] = Node(dof_id,j,i,k,x,y,z,H,1)

					if (y<.5) and (x<1) and (z<1):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						e_id += 1

					if (x == 0.5 or x==1):
						self.interface[1][x==1].append(dof_id)
					if (y > 0.5-H or y <H):
						self.interface[1][2+(y>H)].append(dof_id)
					if z < 0 or z > 1:
						zind = zlen-2 if k==0 else k
						zind = 1 if k==zlen-1 else zind
					
						fill_id = zind*(xlen*ylen)+i*(xlen)+j+n_prev_dofs
						self.periodic.append([dof_id,fill_id,x==0 or x==.5 or y>.5-H or y<H])

					dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q2(self):
		self.interface[2] = [[],[],[],[]]
		H = self.h*2
		xdom = np.linspace(0,0.5,int(self.N/2)+1)
		ydom = np.linspace(.5-H/2,1+H/2,int(self.N/2)+2)
		zdom = np.linspace(0-H/2,1+H/2,self.N+2)

		xlen,ylen,zlen = len(xdom),len(ydom),len(zdom)

		n_prev_dofs = self.dof_count
		dof_id,e_id = self.dof_count, self.el_count
		for k,z in enumerate(zdom):
			zinterface_element = (k==0 or k==zlen-2)
			zside = 2+(k==0)
			for i,y in enumerate(ydom):
				yinterface_element = (i==0 or i==ylen-2)
				yside = i==0
				for j,x in enumerate(xdom):
					self.dofs[dof_id] = Node(dof_id,j,i,k,x,y,z,H,2)

					if (x<.5) and (y<1) and (z<1):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						e_id += 1

					if x == H:
						self.boundaries.append(dof_id)
					if (x==.5) or (x==0):
						self.interface[2][x>0].append(dof_id)
					if (y<.5+H or y>1-H):
						self.interface[2][2+(y>1-H)].append(dof_id)
					if z < 0 or z > 1:
						zind = zlen-2 if k==0 else k
						zind = 1 if k==zlen-1 else zind
					
						fill_id = zind*(xlen*ylen)+i*(xlen)+j+n_prev_dofs
						self.periodic.append([dof_id,fill_id,x==0 or x==.5 or y>.5-H or y<H])

					dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id

	def _make_q3(self):
		self.interface[3] = [[],[],[],[]]
		H = self.h
		xdom = np.linspace(0.5,1.,self.N+1)
		ydom = np.linspace(.5-H/2,1+H/2,self.N+2)
		zdom = np.linspace(0-H/2,1+H/2,2*self.N+2)

		xlen,ylen,zlen = len(xdom),len(ydom),len(zdom)
		
		n_prev_dofs = self.dof_count

		dof_id,e_id = self.dof_count, self.el_count
		for k,z in enumerate(zdom):
			zinterface_element = (k==0 or k==zlen-2)
			zside = 2+(k==0)
			for i,y in enumerate(ydom):
				yinterface_element = (i==0 or i==ylen-2)
				yside = i==0
				for j,x in enumerate(xdom):
					self.dofs[dof_id] = Node(dof_id,j,i,k,x,y,z,H,3)

					if (x<1.) and (y<1) and (z<1):
						strt = dof_id
						element = Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						element.set_fine()
						self.elements.append(element)
						if yinterface_element: element.set_interface(yside)
						if zinterface_element: element.set_interface(zside)
						
						e_id += 1


					if (x == 0.5 or x==1.):
						self.interface[3][x==1].append(dof_id)
				
					elif (y<.5+H or y>1-H):
						self.interface[3][2+(y>1-H)].append(dof_id)
			

					elif z < 0 or z > 1:
						zind = zlen-2 if k==0 else k
						zind = 1 if k==zlen-1 else zind
					
						fill_id = zind*(xlen*ylen)+i*(xlen)+j+n_prev_dofs
						self.periodic.append([dof_id,fill_id,False])

					dof_id += 1

		self.dof_count = dof_id
		self.el_count = e_id

class CornerRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=CornerRefinementMesh)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.dirichlet = np.zeros(num_dofs)
		mymap = np.arange(num_dofs)

		Cr, Cc, Cd = [],[],[]

		q0 = self.mesh.interface[0]
		q1 = self.mesh.interface[1]
		q2 = self.mesh.interface[2]
		q3 = self.mesh.interface[3]
		

		# horizontal refinement (x = .5)
		for i in range(2):
			self.Id[q3[1-i]] = 1
			mymap[q3[1-i]] = -1

			fgrid = np.array(q3[1-i]).reshape((-1,self.N+2))
			cgrid = np.array(q2[i]).reshape((-1,int(self.N/2)+2))

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


		# vertical refinement (y = .5)
		for j in range(2):
			cgrid = np.array(q1[2+j]).reshape((-1,2,int(self.N/2)+1))
			fgrid = np.array(q3[3-j]).reshape((-1,2,self.N-1))

			self.Id[fgrid[:,j,:].flatten()] = 1
			mymap[fgrid[:,j,:].flatten()] = -1
			
			findsa = fgrid[:,j,:]
			findsb = fgrid[:,1-j,:]
			Cr += list(findsa.flatten())
			Cc += list(findsb.flatten())
			Cd += [-1]*findsa.size


			for k in range(2):
				for i in range(2):
					for l in range(2):
						end = None if l else -1
						val = 3/4 if i==l else 1/4
						finds0 = fgrid[i::2,j,1::2]
						cinds0 = cgrid[l:end,k,1:-1]


						Cr += list(finds0.flatten())
						Cc += list(cinds0.flatten())
						Cd += [val]*finds0.size

						for m in range(2):
							endx = None if m else -1
							finds1 = fgrid[i::2,j,::2]
							cinds1 = cgrid[l:end,k,m:endx]
							Cr += list(finds1.flatten())
							Cc += list(cinds1.flatten())
							Cd += [val/2]*finds1.size

		# same level horizontally
		for pair in [(q0[0],q1[1]),(q1[0],q0[1])]:
			# lower case are ghosts, upper case are true dofs
			B0,t0 = np.array(pair[0]+pair[1]).reshape((2,-1))#-1,2)).T
			ghost_list = np.array(t0)
			self.mesh.periodic_ghost.append(ghost_list)
			for (D,d) in zip(B0,t0):
				mymap[mymap==d] = mymap[D]
			self.Id[ghost_list] = 1.

		## same level vertically 
		xlen = int(self.N/2)+1
		for pair in [(q0[2],q2[3]),(q2[2],q0[3])]:
			tmp = np.array(pair[0]).reshape((-1,2,xlen)).copy()
			b0, B1 = tmp[:,0,:].flatten(),tmp[:,1,:].flatten()

			tmp = np.array(pair[1]).reshape((-1,2,xlen)).copy()
			T0, t1 = tmp[:,0,:].flatten(),tmp[:,1,:].flatten()

			ghost_list = np.hstack((b0,t1))
			self.mesh.periodic_ghost.append(ghost_list)
			Ds,ds = [T0,B1],[b0,t1]
			for (D,d) in zip(Ds,ds):
				for g,t in zip(d,D):
					mymap[mymap==g] = mymap[t]
			self.Id[ghost_list] = 1.

		# corners
		ylen = int(self.N/2)+2
		pairs = [(q1[0][ylen-1::ylen],q2[1][1::ylen]),(q2[1][::ylen],q1[0][ylen-2::ylen]),
				 (q1[0][::ylen],q2[1][ylen-2::ylen]),(q2[1][ylen-1::ylen],q1[0][1::ylen])]
		for (gs,ts) in pairs:
			self.Id[gs] = 1
			for g,t in zip(gs,ts):
				mymap[mymap==g] = mymap[t]

		dL,DL,maskL = np.array(self.mesh.periodic).T
		self.Id[dL] = 1
		for (d,D,mask) in zip(dL,DL,maskL):
			mymap[mymap==d] = mymap[D]

		mymap[self.mesh.boundaries] = -1
		for dof_id in self.mesh.boundaries:
			Cr,Cc,Cd = inddel(Cr,Cc,Cd,dof_id)
			assert dof_id not in Cr
			self.Id[dof_id] = 1.
			dof = self.mesh.dofs[dof_id]
			x,y,z = dof.x,dof.y,dof.z
			self.dirichlet[dof_id] = self.ufunc(x,y,z)

		self.true_dofs = list(np.where(self.Id==0)[0])
		assert np.allclose(mymap[self.true_dofs],self.true_dofs)
		self.mymap = mymap

		Cr,Cc,Cd,Cc_small = full_swap_and_set(Cr,Cc,Cd,mymap,self.true_dofs)

		spC_full = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_dofs))
		c_data = {}
		for i,r in enumerate(spC_full.row):
			tup = (spC_full.col[i],spC_full.data[i])
			if r in c_data.keys():
				c_data[r].append(tup)
			else:
				c_data[r] = [tup]
		self.C_full = c_data

		num_true = len(self.true_dofs)
		self.spC = sparse.coo_array((Cd,(Cr,Cc_small)),shape=(num_dofs,num_true)).tocsc()


	def vis_periodic(self,retfig=False):
		for g_id in self.C_full:#c_data:
			if len(self.C_full[g_id]) == 1:
				g_dof = self.mesh.dofs[g_id]
				x,y,z = g_dof.x,g_dof.y,g_dof.z
				for t_id,val in self.C_full[g_id]:
					if g_id!=t_id:
						t_dof = self.mesh.dofs[t_id]
						assert t_dof.h==g_dof.h
						assert val == 1
						tx,ty,tz = t_dof.x,t_dof.y,t_dof.z
						assert tx==x or (tx-x==1) or (x-tx==1)
						assert ty==y or (ty-y==1) or (y-ty==1)
						assert tz==z or (tz-z==1) or (z-tz==1)
		print('all good')
		return
		if retfig: return fig
		plt.show()
		return

	def vis_constraints(self,retfig=False):
		fig,ax = plt.subplots(2,4,figsize=(10,14),gridspec_kw={'width_ratios': [2.5, 1, 1, 1]})
		markers = np.array([['s','^'],['v','o']])
		colors = {1/16:'C0',3/16:'C1',9/16:'C2',1/8:'C3',3/8:'C4',-1:'C5',1/4:'C6',3/4:'C7'}
		flags = {1/16:False,3/16:False,9/16:False,1/8:False,3/8:False,-1:False,1/4:False,3/4:False}
		labs = {1/16:'1/16',3/16:'3/16',9/16:'9/16',1/8:'1/8',3/8:'3/8',-1:'-1',1/4:'1/4',3/4:'3/4'}
		axshow = []
		for g_id in self.C_full:#c_data:
			if len(self.C_full[g_id]) > 1:
				g_dof = self.mesh.dofs[g_id]
				assert self.h==g_dof.h
				x,y,z = g_dof.x,g_dof.y,g_dof.z
				first_col = (x==.5 or x==1)
				if first_col: axind = int(x==.5)
				else: axind = int(y>.75)
 
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

					if tx == x: ax2 = 2
					elif tx < x: ax2 = 1
					else: ax2 = 3
					if first_col: ax2 = 0

					m = markers[int(ty==cy),int(tz==cz)]

					ax[axind,ax2].scatter([y],[z],color='k',marker='o')
					ax[axind,ax2].scatter([ty],[tz],color='k',marker=m)
					if flags[val]==False:
						ax[axind,ax2].plot([y,ty],[z,tz],color=colors[val],label=labs[val])
						flags[val] = True
						if [axind,ax2] not in axshow:
							axshow.append([axind,ax2])
					else:
						ax[axind,ax2].plot([y,ty],[z,tz],color=colors[val])
					
		for inds in axshow:
			ax[inds[0],inds[1]].legend()
		ttl = ['y = .5','y = 1','x = 0','x = .5']
		for i in range(2):
			for j in range(4):
				if j>0: ax[i,j].set_title(ttl[i])
				else: ax[i,j].set_title(ttl[2+i])
				ax[i,j].set_aspect('equal')
				ax[i,j].set_ylabel('z')
				ax[i,j].set_xlabel('y')
		plt.tight_layout()
		plt.show()
		if retfig: return fig
	
	def vis_mesh(self,retfig=True):
		fig = super().vis_mesh(corner=True,retfig=True)
		if retfig: return fig

	def xy_to_e(self,x,y):
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12

		if (x<=.5) and (y<= .5): #q0
			x_ind = int(x/2/self.h)
			y_ind = int(y/2/self.h+1/2)
			e_ind = y_ind*self.N/2+x_ind
		elif (x<=.5):# q2
			x_ind = int(x/2/self.h)
			y_ind = int((y-.5)/2/self.h+1/2)
			e_ind = self.mesh.n_els[1]+y_ind*self.N/2+x_ind
		elif (y<=.5): #q1
			x_ind = int((x-.5)/2/self.h)
			y_ind = int(y/2/self.h+1/2)
			e_ind = self.mesh.n_els[0]+y_ind*self.N/2+x_ind
		else: #q3
			x_ind = int((x-.5)/self.h)
			y_ind = int((y-.5)/self.h+1/2)
			e_ind = self.mesh.n_els[2]+y_ind*self.N+x_ind
			
		e = self.mesh.elements[int(e_ind)]
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		return e
