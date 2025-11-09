import numpy as	np
import matplotlib.pyplot as	plt
import scipy.linalg	as la
import scipy.sparse.linalg as sla
from scipy import sparse
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


from linear_basis.stokes.helpers_2d	import *


class Node:
	def	__init__(self,ID,j,i,x,y,h):
		self.ID	= ID
		self.j = j
		self.i = i
		self.x = x
		self.y = y
		self.h = h
		self.elements =	{}

	def	add_element(self,e):
		if e.ID	not	in self.elements.keys():
			self.elements[e.ID]	= e

class Element:
	def	__init__(self,ID,j,i,p,x,y,h):
		self.ID	= ID
		self.j = j
		self.i = i
		self.p = p
		self.x = x
		self.y = y
		self.h = h
		self.dof_ids = []
		self.dof_list =	[]
		self.fine =	False
		self.interface = False
		self.side =	None
		self.dom = [x,x+h,y,y+h]

	def	add_dofs(self,strt,xlen):
		if len(self.dof_ids) !=	0:
			return
		for	ii in range(self.p[1]+1):
			for	jj in range(self.p[0]+1):
				self.dof_ids.append(strt+xlen*ii+jj)
		return

	def	update_dofs(self,dofs):
		if len(self.dof_list) != 0:
			return
		for	dof_id in self.dof_ids:
			dof	= dofs[dof_id]
			dof.add_element(self)
			self.dof_list.append(dof)
		return

	def	set_fine(self):
		self.fine =	True

	def	set_interface(self,yside):
		self.interface = True
		self.side =	yside
 
		if yside is	not	None:
			self.dom[3-yside] = self.y+self.h/2

class Mesh:
	def	__init__(self,N,p):
		self.N = N # number	of coarse elements 
				   # from x=0.0	to x=1.0
		self.h = 1./N
		self.p = p
		self.uniform = False
		self.dofs =	{}
		self.elements =	[]
		self.periodic =	[]

		self.dof_count = 0
		self.el_count =	0
		
		self.n_els = []

		self._make_coarse()
		self._make_fine()

		self._update_elements()

	def	_make_coarse(self):
		raise ValueError('virtual needs	to be overwritten')

	def	_make_fine(self):
		raise ValueError('virtual needs	to be overwritten')

	def	_update_elements(self):
		for	e in self.elements:
			e.update_dofs(self.dofs)

class UniformMesh(Mesh):
	def	__init__(self,N,p):
		super().__init__(N,p)
		self.uniform = True

	def	_make_coarse(self):
		H =	self.h
		xp,yp =	self.p
		xL,xR,xT = int((xp-1)/2), int(xp/2), xp-1
		yL,yR,yT = int((yp-1)/2), int(yp/2), yp-1
		xdom = np.linspace(0-H*xL,1+H*xR,self.N+1+xT)
		ydom = np.linspace(-H/2-yL*H,1+H/2+yR*H,self.N+2+yT)

		xlen,ylen =	len(xdom),len(ydom)

		dof_id,e_id	= 0,0
		for	i,y	in enumerate(ydom):
			for	j,x	in enumerate(xdom):
				self.dofs[dof_id] =	Node(dof_id,j,i,x,y,H)

				if (0<=x<1) and (-H<y<1):
					strt = dof_id-(xL+yL*xlen)
					element	= Element(e_id,j,i,self.p,x,y,H)
					interface = (y+H > 1) or (y<0)
					if interface: element.set_interface(y<0)
					element.add_dofs(strt,xlen)
					self.elements.append(element)
					e_id +=	1
				
				if x < 0 or	x >= 1 or y	< 0	or y >=	1:
					xind = j
					if x<0:	xind +=	(xlen-1-xT)
					if x>=1: xind -= (xlen-1-xT)
					
					yind = i
					if y<0:	yind +=	(ylen-2-yT)
					if y>=1: yind -= (ylen-2-yT)

					fill_id	= yind*(xlen)+xind
					
					self.periodic.append([dof_id,fill_id])
				
				dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els =	e_id

	def	_make_fine(self):
		pass

	def vis_mesh(self):
		frame = [[0,0,1,1,0],[0,1,1,0,0]]
		fig,ax = plt.subplots(1,2)#,figsize=(20,10))
		for k in range(2):
			ax[k].set_xlim(-3*self.h,1+3*self.h)
			ax[k].set_ylim(-3*self.h,1+3*self.h)
			ax[k].set_aspect("equal")

		xl,yl =	self.p[0]+1,self.p[1]+1
		size = xl*yl

		n_to_e_id = []
		e_id = 0
		for n in range(len(self.dofs)):
			dof = self.dofs[n]
			xcheck = (-self.h <= dof.x < 1)
			ycheck = (-self.h <= dof.y < 1)
			if xcheck and ycheck:
				e = self.elements[e_id]
				if (e.x==dof.x) and (e.y==dof.y):
					n_to_e_id.append(e_id)
					e_id += 1
				else:
					n_to_e_id.append(None)

				
				#if prev is None:
				#	prev = 0
				#else:
				#	e = self.elements[prev+1]
				#	if (e.x==dof.x) and (e.y==dof.y):
				#		prev += 1
				#n_to_e_id.append(prev)
			else:
				n_to_e_id.append(None)
	
		lines = []
		for k in range(2):
			line, = ax[k].plot(frame[0],frame[1],'lightgrey')
			lines.append(line)

		blocks = []
		for _ in range(size):
			block, = ax[0].plot([],[])
			blocks.append(block)
		dot, = ax[0].plot([],[],'ko',linestyle='None')

		eline, = ax[1].plot([],[])
		dots, = ax[1].plot([],[],'ko',linestyle='None')

		def update(n):
			dof = self.dofs[n]
			els = list(dof.elements.values())
			for line in lines:
				line.set_data(frame[0],frame[1])
			dot.set_data(dof.x,dof.y)
			for i in range(size):
				if i < len(dof.elements):
					e = els[i]
					x0,x1,y0,y1 = e.dom
					blocks[i].set_data([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0])
				else:
					blocks[i].set_data([],[])
			e_id = n_to_e_id[n]
			if e_id is not None:
				e = self.elements[e_id]
				x0,x1,y0,y1 = e.dom
				eline.set_data([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0])
				xs,ys = [],[]
				for dof in e.dof_list:
					xs.append(dof.x)
					ys.append(dof.y)
				dots.set_data(xs,ys)
			else:
				eline.set_data([],[])
				dots.set_data([],[])

			return [lines,blocks,dot,eline,dots]
		interval = 400
		ani = FuncAnimation(fig, update, frames=len(self.dofs), interval=interval)
		plt.close()
		return HTML(ani.to_html5_video())

class Solver:
	def	__init__(self,N,p,u,f=None,qpn=5,meshtype=Mesh):
		self.N = N
		self.p = p
		self.ufunc = u
		self.ffunc = f #needs to be	overwritten	
		self.qpn = qpn

		self.mesh =	meshtype(N,self.p)
		self.h = self.mesh.h
		self.uniform = self.mesh.uniform

		self.spC = None
		self.Id	= None

		self.U_lap = None
		self.U_proj	= None

		self._solved = False

		self._setup_constraints()

	def	_build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		F =	np.zeros(num_dofs)
		myfunc = self.ufunc	if proj	else self.ffunc

		xl,yl =	self.p[0]+1,self.p[1]+1
		id_to_ind =	{ID:[int(ID/xl),ID%xl] for ID in range(xl*yl)}

		for	e in self.mesh.elements:
			y0,y1 =	e.dom[2]-e.y, e.dom[3]-e.y
			for	test_id,dof	in enumerate(e.dof_list):
				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi_2d_ref(self.p,x,y,e.h,test_ind)
				func = lambda x,y: phi_test(x,y) * myfunc(x+e.x,y+e.y)
				val	= gauss(func,0,e.h,y0,y1,self.qpn)

				F[dof.ID] += val
		if proj:
			self.F_proj	= F
			self.block_F_proj = np.hstack((self.F_proj,np.array([0])))
		else:
			self.F = F
			self.block_F = np.hstack((self.F,np.array([0])))

	def	_build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		Kr, Kc, Kd = [],[],[]
		base_k = local_stiffness(self.h,self.p,qpn=self.qpn)
	
		interface_k0 = local_stiffness(self.h,self.p,qpn=self.qpn,y1=.5)
		interface_k1 = local_stiffness(self.h,self.p,qpn=self.qpn,y0=.5)
		interface_k	= [interface_k0,interface_k1]
		ks = [base_k]+interface_k

		for	e in self.mesh.elements:
			for	test_id,dof	in enumerate(e.dof_list):
				k_id = 0 if not e.interface else 1+e.side
				Kr += [dof.ID]*len(e.dof_ids)
				Kc += e.dof_ids
				Kd += list(ks[k_id][test_id])
		self.spK = sparse.coo_array((Kd,(Kr,Kc)),shape=(num_dofs,num_dofs)).tocsc()

		try:
			assert self.G is not None
		except:
			self._build_zero_mean()
		Grc = [ind for ind in range(num_dofs)]
		Gcr = [num_dofs]*num_dofs
		Gd = list(self.G)
		self.block_spK = sparse.coo_array(
			(Kd+Gd+Gd,(Kr+Grc+Gcr,Kc+Gcr+Grc)),
			shape=(num_dofs+1,num_dofs+1)).tocsc()

	def	_build_mass(self):
		num_dofs = len(self.mesh.dofs)
		Mr, Mc, Md = [],[],[]
		base_m = local_mass(self.h,qpn=self.qpn)
	
		interface_m0 = local_mass(self.h,qpn=self.qpn,y1=.5)
		interface_m1 = local_mass(self.h,qpn=self.qpn,y0=.5)
		interface_m	= [interface_m0,interface_m1]
		ms = [base_m]+interface_m

		for	e in self.mesh.elements:
			scale =	1 if e.fine	else 4
			for	test_id,dof	in enumerate(e.dof_list):
				m_id = 0 if not e.interface else 1+e.side
				Mr += [dof.ID]*len(e.dof_ids)
				Mc += e.dof_ids
				Md += list(ms[m_id][test_id]*scale)
		self.spM = sparse.coo_array((Md,(Mr,Mc)),shape=(num_dofs,num_dofs)).tocsc()
		
		try:
			assert self.G is not None
		except:
			self._build_zero_mean()
		Grc = [ind for ind in range(num_dofs)]
		Gcr = [num_dofs]*num_dofs
		Gd = list(self.G)
		self.block_spM = sparse.coo_array(
			(Md+Gd+Gd,(Mr+Grc+Gcr,Mc+Gcr+Grc)),
			shape=(num_dofs+1,num_dofs+1)).tocsc()

	def _build_zero_mean(self):
		num_dofs = len(self.mesh.dofs)
		self.G = np.zeros(num_dofs)

		base_g = local_zero_mean(self.h,self.p,qpn=self.qpn)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			for test_id,dof in enumerate(e.dof_list):
				self.G[dof.ID] += base_g[test_id] * scale

	def	projection(self,construct_only=False):
		self._build_mass()
		self._build_force(proj=True)

		# build block system
		self.LHS = self.block_spC.T @ self.block_spM @ self.block_spC
		self.RHS = self.block_spC.T.dot(-self.block_F_proj)
		if construct_only:
			return
		try:
			spx,conv = sla.cg(self.LHS,self.RHS,rtol=1e-14)
			assert conv==0
			self.U_proj = self.spC.dot(spx)
			self._solved = True
		except:
			print('something went wrong')


	def	laplace(self,construct_only=False):
		if self.ffunc is None:
			raise ValueError('f	not	set, call .add_force(func)')
		self._build_stiffness()
		self._build_force()

		# build block system
		self.LHS = self.block_spC.T @ self.block_spK @ self.block_spC
		self.RHS = self.block_spC.T.dot(-self.block_F)
		if construct_only:
			return
		try:
			spx,conv = sla.cg(self.LHS,self.RHS,rtol=1e-14)
			assert conv==0
			self.U_lap = self.spC.dot(spx)
			self._solved = True
		except:
			print('something went wrong')

	def	_setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		Cr, Cc, Cd = [],[],[]

		dL,DL = np.array(self.mesh.periodic).T

		for (d,D) in zip(dL,DL):
			self.Id[d] = 1
			Cr.append(d)
			Cc.append(D)
			Cd.append(1.)

		self.true_dofs = list(np.where(self.Id==0)[0])

		for true_ind in self.true_dofs:
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
		self.C_full_sp = c_data

		Cc_array = np.array(Cc)
		masks = []
		for true_dof in self.true_dofs:
			masks.append(Cc_array==true_dof)
		for j,mask in enumerate(masks):
			Cc_array[mask] = j
		Cc = list(Cc_array)

		num_true = len(self.true_dofs)
		self.spC = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_true)).tocsc()
		self.block_spC = sparse.coo_array(
			(Cd+[1],(Cr+[num_dofs],Cc+[num_true])),shape=(num_dofs+1,num_true+1)).tocsc()

	def	add_force(self,f):
		self.ffunc = f

	def	add_field(self,u):
		self.ufunc = u

	def	vis_mesh(self,corner=False,retfig=False):
		fig,ax = plt.subplots(1,figsize=(7,7))
		mk = ['^','o']
		for	dof	in self.mesh.dofs.values():
			m =	mk[dof.h==self.mesh.h]
			c =	'grey' if self.Id[dof.ID] else 'k'
			if dof.i ==	0 and dof.j==0 and dof.ID !=0:
				plt.scatter(dof.x,dof.y,marker=m,color=c,label='dof')
			plt.scatter(dof.x,dof.y,marker=m,color=c)

		fine_inter = [dof for side in self.mesh.interface[2*corner+1] for dof in side]
		#fine_inter	= self.mesh.interface[2*corner+1][0]+self.mesh.interface[2*corner+1][1]
		for	i,i_id in enumerate(fine_inter):
			assert self.Id[i_id] or	corner
			dof	= self.mesh.dofs[i_id]
			assert dof.h==self.h
			c =	'C1' if	self.Id[i_id] else 'k'
			if i==0:
				plt.scatter(dof.x,dof.y,marker='o',color=c,label='interface')
			plt.scatter(dof.x,dof.y,marker='o',color=c)

		for	level in range(2):
			for	i,g_id in enumerate(self.mesh.periodic_ghost[level]):
				assert self.Id[g_id]
				dof	= self.mesh.dofs[g_id]
				m =	'^'	if corner else mk[level]
				if i==0	and	level==1:
					plt.scatter(dof.x,dof.y,marker=m,color='C0',label='periodic')
				plt.scatter(dof.x,dof.y,marker=m,color='C0')

		for	i,b_id in enumerate(self.mesh.boundaries):
			assert self.Id[b_id]
			dof	= self.mesh.dofs[b_id]
			if i==0:
				plt.scatter(dof.x,dof.y,marker='^',color='C2',label='dirichlet')
			plt.scatter(dof.x,dof.y,marker='^',color='C2')

		ax.xaxis.set_major_locator(MultipleLocator(2*self.h))
		ax.xaxis.set_minor_locator(MultipleLocator(self.h))
		
		ax.yaxis.set_major_locator(MultipleLocator(2*self.h))
		ax.yaxis.set_minor_locator(MultipleLocator(self.h))
		
		plt.plot([0,1,1,0,0],[0,0,1,1,0],'k:',linewidth=2)
		
		ax.xaxis.grid(True,'minor',linewidth=.5)
		ax.yaxis.grid(True,'minor',linewidth=.5)
		ax.xaxis.grid(True,'major',linewidth=1)
		ax.yaxis.grid(True,'major',linewidth=1)
		plt.xlim(-2*self.h,1+2*self.h)
		plt.ylim(-2*self.h,1+2*self.h)
		plt.xticks([0,.5,1])
		plt.yticks([0,.5,1])
		plt.legend()
		if retfig: return fig
		#plt.show()
		#return

	def	vis_dof_sol(self,proj=False,err=False):
		U =	self.U_proj	if proj	else self.U_lap
		fig	= plt.figure(figsize=(10,6))
		x0,y0,c0 = [], [], []
		x1,y1,c1 = [], [], []
		for	dof	in self.mesh.dofs.values():
			if dof.h ==	self.h:
				x1.append(dof.x)
				y1.append(dof.y)
				val	= U[dof.ID]
				if err:	val	= abs(val-self.ufunc(dof.x,dof.y))
				c1.append(val)


			else:
				x0.append(dof.x)
				y0.append(dof.y)
				val	= U[dof.ID]
				if err:	val	= abs(val-self.ufunc(dof.x,dof.y))
				c0.append(val)
		
		m =	['o' for v in c1]+['^' for v in	c0]
		
		lo = min(c0+c1)
		hi = max(c0+c1)
		plt.subplot(121)
		plt.scatter(x0,y0,marker='^',vmin=lo,vmax=hi,c=c0,cmap='jet')#,vmin=vmin,vmax=vmax)
		plt.colorbar(location='left')
		#plt.xlim(.5-.5*self.h,1+.5*self.h)
		plt.xlim(-1.5*self.h,1+1.5*self.h)
		plt.ylim(-1.5*self.h,1+1.5*self.h)
		plt.subplot(122)
		plt.scatter(x1,y1,marker='o',vmin=lo,vmax=hi,c=c1,cmap='jet')#,vmin=vmin,vmax=vmax)
		plt.colorbar(location='right')
		#plt.xlim(.5-.5*self.h,1+.5*self.h)
		plt.xlim(-1.5*self.h,1+1.5*self.h)
		plt.ylim(-1.5*self.h,1+1.5*self.h)
		plt.tight_layout()
		plt.show()

	def	xy_to_e(self,x,y):
		try:
			assert self.uniform
		except:
			raise ValueError('virtual xy_to_e func not overwritten')
        
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		x_ind = int(x/self.h)
		y_ind = int(y/self.h+.5)
		el_ind = y_ind*self.N+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert e.dom[0] <= x <= e.dom[1]
		assert e.dom[2] <= y <= e.dom[3]
		return e

	def	sol(self, weights=None,	proj=False):

		if weights is None:
			assert self._solved
			if proj:
				assert self.U_proj is not None
				weights	= self.U_proj
			else:
				assert self.U_lap is not None
				weights	= self.U_lap

		def	solution(x,y):
			e =	self.xy_to_e(x,y)

			val	= 0
			for	dof in e.dof_list:
				val	+= weights[dof.ID]*phi_2d_eval(self.p,x,y,dof.h,dof.x,dof.y)
			
			return val
		return solution

	def	error(self,qpn=5,proj=False):
		uh = self.sol(proj=proj)
		l2_err = 0.
		for	e in self.mesh.elements:
			func = lambda x,y: (self.ufunc(x,y)-uh(x,y))**2
			x0,x1,y0,y1	= e.dom
			val	= gauss(func,x0,x1,y0,y1,qpn)
			l2_err += val
		return np.sqrt(l2_err)
