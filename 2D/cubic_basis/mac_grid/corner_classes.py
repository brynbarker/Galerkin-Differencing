import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from 2D.cubic_basis.mac_grid.classes import Node, Element, Mesh, Solver
from 2D.cubic_basis.mac_grid.helpers import vis_constraints
from 2D.cubic_basis.mac_grid.shape_functions import phi3

class CornerRefinementMesh(Mesh):
	def __init__(self,N):
		self.dof_counts = []
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
		xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
		ydom = np.linspace(0-3*H/2,.5+3*H/2,int(self.N/2)+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (-H<y<.5) and (0 <= x<.5):
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if x==3*H:
					self.boundaries.append(dof_id)
				if (x == 0.5 or x==0):
					self.interface[0][x==.5].append(dof_id)
				if (y > 0.5-2*H or y<2*H):
					self.interface[0][2+(y>.5-2*H)].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id
		self.dof_counts.append(dof_id)

	def _make_q1(self): #coarse
		self.interface[1] = [[],[],[],[]]
		H = self.h*2
		xdom = np.linspace(0.5-H,1+H,int(self.N/2)+3)
		ydom = np.linspace(0-3*H/2,.5+3*H/2,int(self.N/2)+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (-H<y<.5) and (.5<=x<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if (x == 0.5 or x==1):
					self.interface[1][x==1].append(dof_id)
				if (y > 0.5-2*H or y < 2*H):
					self.interface[1][2+(y>2*H)].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.el_count = e_id
		self.dof_counts.append(dof_id)

	def _make_q2(self):
		self.interface[2] = [[],[],[],[]]
		H = self.h*2
		xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
		ydom = np.linspace(.5-3*H/2,1+3*H/2,int(self.N/2)+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0<=x<.5) and (.5-H<y<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if x == 3*H:
					self.boundaries.append(dof_id)
				if (x==.5) or (x==0):
					self.interface[2][x>0].append(dof_id)
				if (y<.5+2*H or y>1-2*H):
					self.interface[2][2+(y>1-2*H)].append(dof_id)

				dof_id += 1

		self.n_els.append(e_id)

		self.dof_count = dof_id
		self.dof_counts.append(dof_id)
		self.el_count = e_id

	def _make_q3(self):
		self.interface[3] = [[],[],[],[]]
		H = self.h
		xdom = np.linspace(0.5-H,1.+H,self.N+3)
		ydom = np.linspace(.5-3*H/2,1+3*H/2,self.N+4)

		xlen,ylen = len(xdom),len(ydom)

		dof_id,e_id = self.dof_count, self.el_count
		for i,y in enumerate(ydom):
			interface_element = (i==1 or i==ylen-3)
			side = i==1
			for j,x in enumerate(xdom):
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (.5<=x<1.) and (.5-H<y<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					if interface_element: element.set_interface(side)
					e_id += 1

				if (y<.5+2*H or y>1-2*H):
					self.interface[3][2+(y>1-2*H)].append(dof_id)
				if (x==.5) or (x==1):
					self.interface[3][x==1].append(dof_id)

				dof_id += 1

		self.dof_count = dof_id
		self.el_count = e_id

class CornerRefineSolver(Solver):
	def __init__(self,N,u,f=None,qpn=5):
		super().__init__(N,u,f,qpn,meshtype=CornerRefinementMesh)

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		q0 = self.mesh.interface[0]
		q1 = self.mesh.interface[1]
		q2 = self.mesh.interface[2]
		q3 = self.mesh.interface[3]

		fine_ill = []

		corner_swap_g = [[],[]]
		fine_corner = []
		coarse_corners = []
		# vertical refinement (y = .5)
		v1, v3 = phi3(1/2,1), phi3(3/2,1)
		for i in range(2):
			c0,c1,c2,c3 = np.array(q1[2+i]).reshape((4,-1))
			f0,f1,f2,f3 = np.array(q3[3-i]).reshape((4,-1))
			if i == 0:
				corner_swap_g[i] += [c0[:2],c1[:2],c0[-2:],c1[-2:]]
			if i == 1:
				corner_swap_g[i] += [c2[:2],c3[:2],c2[-2:],c3[-2:]]
			if i==0:
				f = f3.copy()
				flist = [f2,f1,f0]
				#fine_ill += [f3[0],f3[-1]]
			else:
				f = f0.copy()
				flist = [f1,f2,f3]
				#fine_ill += [f0[0],f0[-1]]

			self.Id[f] = 1
			self.C_full[f] *= 0
	

			for (v,c) in zip([v3,v1,v1,v3],[c0,c1,c2,c3]):
				self.C_full[f[2::2],c[:-2]] = v*v3/v3
				self.C_full[f[::2],c[:-1]] = v*v1/v3
				self.C_full[f[::2],c[1:]] = v*v1/v3
				self.C_full[f[:-1:2],c[2:]] = v*v3/v3
			for (v,fdof) in zip([v1,v1,v3],flist):
				self.C_full[f[::2],fdof[::2]] = -v/v3

			for (v,c) in zip([v3,v1,v1,v3],[c0,c1,c2,c3]):
				self.C_full[f[1::2],c[1:-1]] = v/v3
			for (v,fdof) in zip([v1,v1,v3],flist):
				self.C_full[f[1::2],fdof[1::2]] = -v/v3

			fine_corner += [f[0],f[-1]]
		corner_values = [v3,v3*v1/v3,v3*v1/v3,v3]

		v1, v3, v5, v7 = phi3(1/4,1), phi3(3/4,1), phi3(5/4,1), phi3(7/4,1)
		# horizontal refinement (x = .5)
		for i in range(2):
			self.Id[q3[1-i]] = 1
			self.C_full[q3[1-i]] *= 0

			self.C_full[q3[1-i][1::2],q2[i][:-2]] = v5
			self.C_full[q3[1-i][1::2],q2[i][1:-1]] = v1
			self.C_full[q3[1-i][1::2],q2[i][2:]] = v3
			self.C_full[q3[1-i][1:-1:2],q2[i][3:]] = v7

			self.C_full[q3[1-i][2::2],q2[i][:-3]] = v7
			self.C_full[q3[1-i][::2],q2[i][:-2]] = v3
			self.C_full[q3[1-i][::2],q2[i][1:-1]] = v1
			self.C_full[q3[1-i][::2],q2[i][2:]] = v5

		self.C_full[q3[0][-1],q1[0][4]] = v7
		self.C_full[q3[0][0],q1[0][-5]] = v7

		self.C_full[q3[1][-1],q0[0][4]] = v7
		self.C_full[q3[1][0],q0[0][-5]] = v7

		## same level horizontally
		tmp = [[],[]]
		for pair in [(q0[0],q1[1]),(q1[0],q0[1])]:
			# lower case are ghosts, upper case are true dofs
			B0,t0 = pair[0],pair[1]
			#b0,B1,B2,T0,t1,t2 = np.array(pair[0]+pair[1]).reshape((6,-1))
			ghost_list = np.array([t0])
			self.mesh.periodic_ghost.append(ghost_list)
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds,ds = [B0],[t0]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [0,1,2,3,-4,-3,-2,-1]:
					self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					self.C_full[:,d[ind]] = 0
					self.C_full[d[ind],:] = self.C_full[D[ind],:]
					if self.mesh.dofs[d[ind]].x==.5 and 0<self.mesh.dofs[d[ind]].y<.5:
						tmp[0].append(D[ind])
						tmp[1].append(d[ind])

		corner_swap_t = [[],[]]
		## same level vertically 
		for i,pair in enumerate([(q0[2],q2[3]),(q2[2],q0[3])]):
			# lower case are ghosts, upper case are true dofs
			b0,b1,B2,B3,T0,T1,t2,t3 = np.array(pair[0]+pair[1]).reshape((8,-1))
			coarse_corners += [[T0[-4],T1[-4],B2[-4],B3[-4]],[T0[3],T1[3],B2[3],B3[3]]]
			if i==0:
				corner_swap_t[i] += [T0[-3:-1],T1[-3:-1],T0[1:3],T1[1:3]]
			if i==1:
				corner_swap_t[i] += [B2[-3:-1],B3[-3:-1],B2[1:3],B3[1:3]]
			ghost_list = np.hstack((b0,b1,t2,t3))
			self.mesh.periodic_ghost.append(ghost_list)
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds,ds = [T0,T1,B2,B3],[b0,b1,t2,t3]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [1,-2]:
					self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					self.C_full[:,d[ind]] = 0
					self.C_full[d[ind],:] = self.C_full[D[ind],:]

		for f_ind,c_corner in zip(fine_corner,coarse_corners):
			for c_ind,v in zip(c_corner,corner_values):
				self.C_full[f_ind,c_ind] = v

		## corners
		for f_ind,c_corner in zip(fine_corner,coarse_corners):
			for c_ind,v in zip(c_corner,corner_values):
				self.C_full[f_ind,c_ind] = v

		t_list = list(np.array(corner_swap_t).flatten())
		g_list = list(np.array(corner_swap_g).flatten())
		for (t,g) in zip(t_list+tmp[0],g_list+tmp[1]):
			self.C_full[g] *= 0
			self.C_full[:,t] += self.C_full[:,g]
			self.C_full[:,g] = 0
			self.C_full[g,:] = self.C_full[t,:]
			self.Id[g] = 1
		
		self.C_full[:,list(np.where(self.Id==1)[0])] *= 0
		# dirichlet
		for dof_id in self.mesh.boundaries + fine_ill:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]

	def vis_constraints(self,retfig=False):
		fig = vis_constraints(self.C_full,self.mesh.dofs,self.mesh.interface[3],'corner')
		if retfig: return fig
	
	def vis_mesh(self,retfig=True):
		fig,ax = plt.subplots(2,2,figsize=(7,7))
		mk = ['^','o']
		inds = [[0,1],[0,0],[1,1],[1,0]]
		for ind,dof in enumerate(self.mesh.dofs.values()):
			m = mk[dof.h==self.mesh.h]
			axind = sum([ind < dcount for dcount in self.mesh.dof_counts])
			axi,axj = inds[axind]
			c = 'C0' if ind in self.true_dofs else 'C1'
			ax[axi,axj].plot(dof.x,dof.y,marker=m,color=c)

		plt.show()

	def vis_periodic(self,retfig=False):
		fig = super().vis_periodic('corner')
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
