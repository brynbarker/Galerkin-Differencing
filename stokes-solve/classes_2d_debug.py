import numpy as	np
import matplotlib.pyplot as	plt
import scipy.linalg	as la
import scipy.sparse.linalg as sla
from scipy import sparse
import pickle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


from linear_basis.stokes.helpers_2d	import *

keys = [None,0,1]
LOOKUP={}
REVERSE_LOOKUP = {}

count = 0
for xkey in keys:
	tmp = {}
	for ykey in keys:
		tmp[ykey] = count
		REVERSE_LOOKUP[count] = [xkey,ykey]
		count += 1
	LOOKUP[xkey] = tmp

DOMAIN_LOOKUP = {None:[0,1],0:[0,.5],1:[.5,1]}

class DoF:
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
		
class VelocityDoF(DoF):
	def __init__(self,ID,j,i,x,y,h,dim):
		self.dim = dim
		super().__init__(ID,j,i,x,y,h)

class PressureDoF(DoF):
	def __init__(self,ID,j,i,x,y,h):
		super().__init__(ID,j,i,x,y,h)

class Element:
	def	__init__(self,ID,j,i,order,x,y,h,dim=None,pressure=False):
		self.ID	= ID
		self.j = j
		self.i = i
		self.order = order
		self.x = x
		self.y = y
		self.h = h
		self.dim = dim
		self.pressure = pressure
		self.dof_ids = []
		self.dof_list =	[]
		self.fine =	False
		self.side =	0
		self.side_ops = [0,1]
		self.dom = [x,x+h,y,y+h]

	def	add_dofs(self,strt,xlen):
		if len(self.dof_ids) !=	0:
			return
		if self.pressure:

			print('error should not be calling this function')
			px,py = self.order,self.order
		else:
			px,py = self.order
		for	ii in range(py+1):
			for	jj in range(px+1):
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

	def set_interface(self,xside,yside):
		self.side = LOOKUP[xside][yside]
 
		loc = [self.x,self.y]
		for dim,side in enumerate([xside,yside]):
			if side is not None:
				tmp = 2*dim+1-side
				self.dom[tmp] = loc[dim]+self.h/2

		if not self.pressure:
			sides = [xside,yside]
			if sides[self.dim] != None:
				self.side_ops = [sides[self.dim]]

class Mesh:
	def	__init__(self,N,order):
		self.N = N # number	of coarse elements 
				   # from x=0.0	to x=1.0
		self.h = 1./N
		self.order = order 
		self.uniform = False
		self.u_dofs =	{}
		self.p_dofs =	{}
		self.u_elements =	[]
		self.p_elements =	[]
		self.periodic =	{'vel':[],'p':[]}

		self.dof_counts = []
		self.el_counts = []
		
		self._setup_velocity()
		# self._setup_pressure()

		self._update_elements()

	def	_setup_velocity(self):
		raise ValueError('virtual needs	to be overwritten')

	def	_setup_pressure(self):
		raise ValueError('virtual needs	to be overwritten')

	def	_update_elements(self):
		for	e in self.u_elements:
			e.update_dofs(self.u_dofs)
		for	e in self.p_elements:
			e.update_dofs(self.p_dofs)

class UniformMesh(Mesh):
	def	__init__(self,N,p):
		super().__init__(N,p)
		self.uniform = True

	def	_setup_velocity(self):
		dof_id,e_id	= 0,0
		for dim in range(1):
			H =	self.h
			normal, transverse = self.order#, self.order#-1
			nR,nL,nT = int((normal-1)/2), int(normal/2), normal-1
			tR,tL,tT = int((transverse-1)/2), int(transverse/2), transverse-1
			ndom = np.linspace(0-H*nL,1+H*nR,self.N+1+nT)
			tdom = np.linspace(0-H*tL,1+H*tR,self.N+1+tT)
			#ndom = np.linspace(0-H*nL,1+H*nR,self.N+1+nT)
			#tdom = np.linspace(-H/2-tL*H,1+H/2+tR*H,self.N+2+tT)

			nlen,tlen =	len(ndom),len(tdom)

			doms = [ndom,tdom]
			lens = [nlen,tlen]
			Ls = [nL,tL]
			ords = [normal,transverse]

			shifts = [nlen-1-nT,tlen-1-tT]
			for	i,y	in enumerate(doms[1-dim]):
				yside = None if (0<=y<=1-H) else y<0
				for	j,x	in enumerate(doms[dim]):
					xside = None if (0<=x<=1-H) else x<0
					self.u_dofs[dof_id] =	VelocityDoF(dof_id,j,i,x,y,H,dim)

					xcheck = (-H<x<1) if dim else (-H<x<1)
					ycheck = (-H<y<1) if dim else (-H<y<1)

					if xcheck and ycheck:#(-H<x<1) and (-H<y<1):
						strt = dof_id-(Ls[dim]+Ls[1-dim]*lens[dim])
						element	= Element(e_id,j-Ls[1-dim],i-Ls[dim],
										  [ords[dim],ords[1-dim]],
										  x,y,H,dim=dim)
						interface = (yside is not None) or (xside is not None)
						#if interface: element.set_interface(xside,yside)
						element.add_dofs(strt,lens[dim])
						self.u_elements.append(element)
						e_id +=	1
						
					icheck = (i==0) or (i==lens[1-dim]-1)
					jcheck = (j==0) or (j==lens[dim]-1)
					if icheck or jcheck:#(icheck and not dim) or (jcheck and dim):
						self.periodic['vel'].append(dof_id)
					if False:#x < 0 or	x >= 1 or y	< 0	or y >=	1:
						xind = j
						if x<0:	xind +=	shifts[dim]
						if x>=1: xind -= shifts[dim]

						yind = i
						if y<0:	yind +=	shifts[1-dim]
						if y>=1: yind -= shifts[1-dim]

						fill_id	= yind*(lens[dim])+xind
						if dim: fill_id += self.dof_counts[0]

						self.periodic['vel'].append([dof_id,fill_id])
				
					dof_id += 1

			self.dof_counts.append(dof_id)
			self.el_counts.append(e_id)

	def	_setup_pressure(self):

		print('error should not be calling this function')
		H =	self.h
		pp = self.order-1
		R,L,T = int((pp-1)/2), int(pp/2), pp-1
		dom = np.linspace(-H/2-L*H,1+H/2+R*H,self.N+2+T)
		dlen = len(dom)

		dof_id,e_id	= 0,0
		for	i,y	in enumerate(dom):
			yside = None if (0<y<1-H) else y<0
			for	j,x	in enumerate(dom):
				xside = None if (0<x<1-H) else x<0
				self.p_dofs[dof_id] = PressureDoF(dof_id,j,i,x,y,H)

				if (-H<x<1) and (-H<y<1):
					strt = dof_id-(L+L*dlen)
					element	= Element(e_id,j,i,pp,x,y,H,pressure=True)
					interface = (yside is not None) or (xside is not None)
					if interface: element.set_interface(xside,yside)
					element.add_dofs(strt,dlen)
					self.p_elements.append(element)
					e_id +=	1
				
				if x < 0 or	x >= 1 or y	< 0	or y >=	1:
					xind = j
					if x<0:	xind +=	(dlen-2-T)
					if x>=1: xind -= (dlen-2-T)
					
					yind = i
					if y<0:	yind +=	(dlen-2-T)
					if y>=1: yind -= (dlen-2-T)

					fill_id	= yind*(dlen)+xind
					
					self.periodic['p'].append([dof_id,fill_id])
				
				dof_id += 1

		self.dof_counts.append(dof_id)
		self.el_counts.append(e_id)

	def plot_mesh(self):
		fig = plt.figure(figsize=(20,20))
		dom = np.linspace(0,1,self.N+1)
		for x in dom:
			plt.plot([x,x],[0,1],'k',lw=2)
			plt.plot([0,1],[x,x],'k',lw=2)

		for dof_id in self.u_dofs:
			dof = self.u_dofs[dof_id]
			plt.plot(dof.x,dof.y,'.',ms=20)
		for e in self.u_elements:
			x0,x1,y0,y1 = e.dom
			plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],alpha=.2,lw=5)
		plt.show()


	def vis_mesh(self,comp=0):
		frame = [[0,0,1,1,0],[0,1,1,0,0]]
		fig,ax = plt.subplots(1,2)#,figsize=(20,10))
		for k in range(2):
			ax[k].set_xlim(-3*self.h,1+3*self.h)
			ax[k].set_ylim(-3*self.h,1+3*self.h)
			ax[k].set_aspect("equal")

		xl,yl =	self.order#,self.order+1
		size = xl*yl

		if comp == 0:
			nstart,nend = 0,self.dof_counts[0]
			estart,eend = 0,self.el_counts[0]
		elif comp == 1:
			nstart,nend = self.dof_counts[:-1]
			estart,eend = self.el_counts[:-1]

		n_to_e_id = []
		e_id = estart
		for n in range(nstart,nend):#len(self.u_dofs)):
			dof = self.u_dofs[n]
			xcheck = (-self.h <= dof.x < 1)
			ycheck = (-self.h <= dof.y < 1)
			if xcheck and ycheck:
				e = self.u_elements[e_id]
				if (e.x==dof.x) and (e.y==dof.y):
					n_to_e_id.append(e_id)
					e_id += 1
				else:
					n_to_e_id.append(None)
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

		def update(index):
			n = index+nstart
			dof = self.u_dofs[n]
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
			e_id = n_to_e_id[index]
			if e_id is not None:
				e = self.u_elements[e_id]
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
		ani = FuncAnimation(fig, update, frames=nend-nstart, interval=interval)
		plt.close()
		return HTML(ani.to_html5_video())

class Stokes:
	def	__init__(self,N,order,u,v=None,f=[None,None],mu=1,qpn=5,meshtype=Mesh):
		self.N = N
		self.order = order
		self.ufunc = u
		self.vfunc = v
		self.ffunc = f #needs to be	overwritten	
		self.mu = mu
		self.qpn = qpn

		self.mesh =	meshtype(N,self.order)
		self.h = self.mesh.h
		self.uniform = self.mesh.uniform

		self.spC = None
		self.Id	= None

		self.U = None
		self.V = None
		self.P = None

		self._solved = False

		self._setup_constraints()

		# self._grab_local_ks()
		# self._grab_local_ms()
		return
		self._grab_local_divs()

	def _grab_local_ms(self):
		fname = 'local_ms_p{}_q{}.pickle'.format(
			self.order,self.qpn)
		try:
			# np.aray()
			with open(fname,'rb') as handle:
				m_dict = pickle.load(handle)
		except:
			m_list = []
			for key in range(9):
				xside,yside = REVERSE_LOOKUP[key]
				m_dict.append(
						local_mass(
							1,self.order,self.qpn,
							xside,yside))

			with open(fname,'wb') as handle:
				pickle.dump(m_dict,handle,
					protocol=pickle.HIGHEST_PROTOCOL)
		self.local_ms = m_dict
			

	def _grab_local_ks(self):
		fname = 'local_ks_p{}_q{}.pickle'.format(
			self.order,self.qpn)
		try:
			# np.aray()
			with open(fname,'rb') as handle:
				k_dict = pickle.load(handle)
		except:
			ps = {0:self.order,
		 		  1:self.order[::-1]}
			#ps = {0:[self.order,self.order-1],
		 	#	  1:[self.order-1,self.order]}
			k_dict = {0:[],1:[]}
			for key in range(9):
				xside,yside = REVERSE_LOOKUP[key]
				for dim in range(2):
					k_dict[dim].append(
						local_stiffness(
							1,ps[dim],self.qpn,
							xside,yside))

			with open(fname,'wb') as handle:
				pickle.dump(k_dict,handle,
					protocol=pickle.HIGHEST_PROTOCOL)
		self.local_ks = k_dict
			

	def _grab_local_divs(self):

		print('error should not be calling this function')
		fname = 'local_divs_p{}_q{}.pickle'.format(
			self.order,self.qpn)
		try:
			with open(fname,'rb') as handle:
				div_dict = pickle.load(handle)
		except:

			div_dict = {}
			for key in range(9):
				xside,yside = REVERSE_LOOKUP[key]
				div_dict[key] = local_divergence(1,self.order,self.qpn,xside,yside)

			with open(fname,'wb') as handle:
				pickle.dump(div_dict,handle,
					protocol=pickle.HIGHEST_PROTOCOL)
		self.local_divs = div_dict

	def	_build_force(self):
		num_u_dofs = len(self.mesh.u_dofs)
		self.F =	np.zeros(num_u_dofs)

		for	e in self.mesh.u_elements:
			p0,p1 = e.order
			xl,yl = p0+1,p1+1
			id_to_ind =	{ID:[int(ID/xl),ID%xl] for ID in range(xl*yl)}
			y0,y1 =	e.dom[2]-e.y, e.dom[3]-e.y
			for	test_id,dof	in enumerate(e.dof_list):
				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi_2d_ref([p0,p1],x,y,e.h,test_ind)
				func = lambda x,y: phi_test(x,y) * self.ffunc[e.dim](x+e.x,y+e.y)
				val	= gauss(func,0,e.h,y0,y1,self.qpn)

				self.F[dof.ID] += val

	def	_build_lap_force(self,myf=None,proj=False):
		num_u_dofs = len(self.mesh.u_dofs)
		F =	np.zeros(num_u_dofs)

		if proj:
			myf = self.ufunc

		for	e in self.mesh.u_elements:
			p0,p1 = e.order
			xl,yl = p0+1,p1+1

			id_to_ind =	{ID:[int(ID/xl),ID%xl] for ID in range(xl*yl)}
			for	test_id,dof	in enumerate(e.dof_list):
				test_ind = id_to_ind[test_id]
				phi_test = lambda x,y: phi_2d_ref([p0,p1],x,y,e.h,test_ind)
				func = lambda x,y: phi_test(x,y) * myf(x+e.x,y+e.y)
				val	= gauss(func,0,self.h,0,self.h,self.qpn)

				F[dof.ID] += val
		if proj:
			self.F_proj=F
		else:
			self.F_lap = -F

	def _uv_to_p_element(self,e):
		print('error should not be calling this function')
		p_dict = {0:None,1:None}
		e_ID = e.ID-e.dim*self.mesh.el_counts[0]
		if e.dim == 0:
			i,j = int(e_ID/self.N), e_ID % self.N
			pID0 = (self.N+1)*i+j 
			pID1 = pID0+1
		else:
			assert e.dim == 1
			pID0 = e_ID
			pID1 = e_ID+self.N+1
		pIDs = [pID0,pID1]
		for side in e.side_ops:
			p_dict[side] = self.mesh.p_elements[pIDs[side]]
		return p_dict


	def _build_divergence(self):
		print('error should not be calling this function')
		num_u_dofs = len(self.mesh.u_dofs)
		num_p_dofs = len(self.mesh.p_dofs)
		Dr,Dc,Dd = [],[],[]

		for e in self.mesh.u_elements:
			p_els = self._uv_to_p_element(e)
			mydivs = self.local_divs[e.side][e.dim]#*self.h
			for s in mydivs:
				mydiv = mydivs[s]*self.h
				p_el = p_els[s]
				for test_id,dof in enumerate(p_el.dof_list):
					Dr += [dof.ID]*len(e.dof_ids)
					Dc += e.dof_ids
					Dd += list(mydiv[test_id])
		self.spDiv = sparse.coo_array((Dd,(Dr,Dc)),
								shape=(num_p_dofs,num_u_dofs)).tocsc()


	def	_build_stiffness(self):
		num_u_dofs = len(self.mesh.u_dofs)
		Kr, Kc, Kd = [],[],[]
		local_k = local_stiffness(self.h,self.order,self.qpn)

		for	e in self.mesh.u_elements:
			for	test_id,dof	in enumerate(e.dof_list):
				Kr += [dof.ID]*len(e.dof_ids)
				Kc += e.dof_ids
				Kd += list(local_k[test_id])
		self.spK = sparse.coo_array((Kd,(Kr,Kc)),shape=(num_u_dofs,num_u_dofs)).tocsc()

	def	_build_mass(self):
		num_u_dofs = len(self.mesh.u_dofs)
		Mr, Mc, Md = [],[],[]
		local_m = local_mass(self.h,self.order,self.qpn)

		for	e in self.mesh.u_elements:
			for	test_id,dof	in enumerate(e.dof_list):
				Mr += [dof.ID]*len(e.dof_ids)
				Mc += e.dof_ids
				Md += list(local_m[test_id])
		self.spM = sparse.coo_array((Md,(Mr,Mc)),shape=(num_u_dofs,num_u_dofs)).tocsc()
		
	def _build_zero_mean(self):
		print('error should not be calling this function')
		num_dofs = len(self.mesh.u_dofs)
		self.G = np.zeros(num_dofs)
		Gr, Gc, Gd = [],[],[]

		base_gu = local_zero_mean(self.h,[self.order,self.order-1],qpn=self.qpn)
		base_gv = local_zero_mean(self.h,[self.order-1,self.order],qpn=self.qpn)
		base_g = [base_gu,base_gv]

		for e in self.mesh.u_elements:
			for test_id,dof in enumerate(e.dof_list):
				Gr += [dof.ID]
				Gc += [0]
				Gd += [base_g[e.dim][test_id]]
				# self.G[dof.ID] += base_g[test_id] * scale
		self.spG = sparse.coo_array((Gd,(Gr,Gc)),
								shape=(num_dofs,1)).tocsc()

	def	build(self):
		self._build_stiffness()
		self._build_divergence()
		self._build_zero_mean()
		self._build_force()
		zero_block = np.zeros(len(self.mesh.p_dofs)+1)

		# build block system
		self.block_sys = sparse.bmat(np.array(
			[[self.mu*self.spK,self.spDiv.T,self.spG],
			 [self.spDiv,None,None],
			 [self.spG.T,None,None]]),format='csc')
		self.LHS = self.spC.T @ self.block_sys @ self.spC
		self.RHS = self.spC.T.dot(np.hstack((self.F,zero_block)))
		#self.LHS = self.block_spC.T @ self.block_spK @ self.block_spC
		#self.RHS = self.block_spC.T.dot(-self.block_F)
		
	def solve(self):
		try:
			spx,conv = sla.gmres(self.LHS,self.RHS,rtol=1e-14)
			assert conv==0
			self.U = self.spC.dot(spx)
			self._solved = True
		except:
			print('something went wrong')


	def	laplace(self,lap_force):
		self._build_stiffness()
		self._build_lap_force(lap_force)

		self.LHS_lap = self.spC_u.T @ self.spK @ self.spC_u
		rhs_tmp = self.F_lap - self.spK.dot(self.dirichlet)
		self.RHS_lap = self.spC_u.T.dot(rhs_tmp)
		if True:#try:
			spx,conv = sla.cg(self.LHS_lap,self.RHS_lap,rtol=1e-14)
			assert conv==0
			self.lap_x = spx
			self.U_lap = self.spC_u.dot(spx)+self.dirichlet
			self._solved = True
		return

	def	projection(self):
		self._build_mass()
		self._build_lap_force(proj=True)

		self.LHS_proj = self.spC_u.T @ self.spM @ self.spC_u
		rhs_tmp = self.F_proj - self.spM.dot(self.dirichlet)
		self.RHS_proj = self.spC_u.T.dot(rhs_tmp)
		if True:#try:
			spx,conv = sla.cg(self.LHS_proj,self.RHS_proj,rtol=1e-14)
			assert conv==0
			self.proj_x = spx
			self.U_proj = self.spC_u.dot(spx)+self.dirichlet
			self._solved = True
		return
		

	def	_setup_constraints(self):
		self._setup_velocity_constraints()
		return
		self._setup_pressure_constraints()


		self.spLambda = sparse.coo_array(([1],([0],[0])),
								shape=(1,1)).tocsc()

		self.spC = sparse.bmat(np.array(
			[[self.spC_u,None,None],
			 [None,self.spC_p,None],
			 [None,None,self.spLambda]]),format='csc')



	def _setup_velocity_constraints(self):
		num_dofs = len(self.mesh.u_dofs)
		self.Id = np.zeros(num_dofs)
		self.dirichlet = np.zeros(num_dofs)
		Cr, Cc, Cd = [],[],[]

		cut = self.mesh.dof_counts[0]

		dD = self.mesh.periodic['vel']
		for dof_id in dD:
			self.Id[dof_id] = 1
			dof= self.mesh.u_dofs[dof_id]
			if True:#dof_id < cut:
				self.dirichlet[dof_id] = self.ufunc(dof.x,dof.y)
			else:
				self.dirichlet[dof_id] = self.vfunc(dof.x,dof.y)
		
		#dL,DL = np.array(self.mesh.periodic['vel']).T

		#for (d,D) in zip(dL,DL):
		#	self.Id[d] = 1
		#	Cr.append(d)
		#	Cc.append(D)
		#	Cd.append(1.)

		self.true_vel_dofs = list(np.where(self.Id==0)[0])

		for true_ind in self.true_vel_dofs:
			Cr.append(true_ind)
			Cc.append(true_ind)
			Cd.append(1.)

		self.spC_full_u = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_dofs))
		c_data = {}
		for i,r in enumerate(self.spC_full_u.row):
			tup = (self.spC_full_u.col[i],self.spC_full_u.data[i])
			if r in c_data.keys():
				c_data[r].append(tup)
			else:
				c_data[r] = [tup]
		self.spC_full_u = c_data

		Cc_array = np.array(Cc)
		masks = []
		for true_dof in self.true_vel_dofs:
			masks.append(Cc_array==true_dof)
		for j,mask in enumerate(masks):
			Cc_array[mask] = j
		Cc = list(Cc_array)

		num_true = len(self.true_vel_dofs)
		self.spC_u = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_true)).tocsc()
		# self.block_spC = sparse.coo_array(
			# (Cd+[1],(Cr+[num_dofs],Cc+[num_true])),shape=(num_dofs+1,num_true+1)).tocsc()

	def _setup_pressure_constraints(self):

		print('error should not be calling this function')
		num_dofs = len(self.mesh.p_dofs)
		self.Id_p = np.zeros(num_dofs)
		Cr, Cc, Cd = [],[],[]

		dL,DL = np.array(self.mesh.periodic['p']).T

		for (d,D) in zip(dL,DL):
			self.Id_p[d] = 1
			Cr.append(d)
			Cc.append(D)
			Cd.append(1.)

		self.true_p_dofs = list(np.where(self.Id_p==0)[0])

		for true_ind in self.true_p_dofs:
			Cr.append(true_ind)
			Cc.append(true_ind)
			Cd.append(1.)

		self.spC_full_p = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_dofs))
		c_data = {}
		for i,r in enumerate(self.spC_full_p.row):
			tup = (self.spC_full_p.col[i],self.spC_full_p.data[i])
			if r in c_data.keys():
				c_data[r].append(tup)
			else:
				c_data[r] = [tup]
		self.spC_full_p = c_data

		Cc_array = np.array(Cc)
		masks = []
		for true_dof in self.true_p_dofs:
			masks.append(Cc_array==true_dof)
		for j,mask in enumerate(masks):
			Cc_array[mask] = j
		Cc = list(Cc_array)

		num_true = len(self.true_p_dofs)
		self.spC_p = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_true)).tocsc()

	def	add_force(self,f):
		self.ffunc = f

	def	add_field(self,u):
		self.ufunc = u

	def	vis_mesh(self,corner=False,retfig=False):
		fig,ax = plt.subplots(1,figsize=(7,7))
		mk = ['^','o']
		for	dof	in self.mesh.u_dofs.values():
			m =	mk[dof.h==self.mesh.h]
			c =	'grey' if self.Id[dof.ID] else 'k'
			if dof.i ==	0 and dof.j==0 and dof.ID !=0:
				plt.scatter(dof.x,dof.y,marker=m,color=c,label='dof')
			plt.scatter(dof.x,dof.y,marker=m,color=c)

		fine_inter = [dof for side in self.mesh.interface[2*corner+1] for dof in side]
		#fine_inter	= self.mesh.interface[2*corner+1][0]+self.mesh.interface[2*corner+1][1]
		for	i,i_id in enumerate(fine_inter):
			assert self.Id[i_id] or	corner
			dof	= self.mesh.u_dofs[i_id]
			assert dof.h==self.h
			c =	'C1' if	self.Id[i_id] else 'k'
			if i==0:
				plt.scatter(dof.x,dof.y,marker='o',color=c,label='interface')
			plt.scatter(dof.x,dof.y,marker='o',color=c)

		for	level in range(2):
			for	i,g_id in enumerate(self.mesh.periodic_ghost[level]):
				assert self.Id[g_id]
				dof	= self.mesh.u_dofs[g_id]
				m =	'^'	if corner else mk[level]
				if i==0	and	level==1:
					plt.scatter(dof.x,dof.y,marker=m,color='C0',label='periodic')
				plt.scatter(dof.x,dof.y,marker=m,color='C0')

		for	i,b_id in enumerate(self.mesh.boundaries):
			assert self.Id[b_id]
			dof	= self.mesh.u_dofs[b_id]
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

	def	vis_dof_sol(self,err=False,alt_sol=None):
		fig= plt.figure(figsize=(15,7))
		x0,y0,c0 = [], [], []
		x1,y1,c1 = [], [], []
		myU = self.U if alt_sol is None else alt_sol
		cut = self.mesh.dof_counts[0]
		for	dof	in self.mesh.u_dofs.values():
			if dof.ID < cut:
				x0.append(dof.x)
				y0.append(dof.y)
				val	= myU[dof.ID]
				if err:	val	= abs(val-self.ufunc(dof.x,dof.y))
				c0.append(val)


			else:
				x1.append(dof.x)
				y1.append(dof.y)
				val	= myU[dof.ID]
				if err:	val	= abs(val-self.vfunc(dof.x,dof.y))
				c1.append(val)
		
		dom = np.linspace(0,1,self.N+1)
		lo = min(c0+c1)
		hi = max(c0+c1)
		plt.subplot(121)
		for i in range(2):
			for j in range(3):
				for x in dom:
					plt.plot([x,x],[0,1],'lightgrey',lw=2,zorder=0)
					plt.plot([0,1],[x,x],'lightgrey',lw=2,zorder=0)
		plt.scatter(x0,y0,marker='^',
			  		vmin=min(c0),vmax=max(c0),
					c=c0,cmap='jet',zorder=1)
		plt.colorbar(location='left')
		plt.xlim(-2*self.h,1+2*self.h)
		plt.ylim(-2*self.h,1+2*self.h)
		plt.subplot(122)
		for i in range(2):
			for j in range(3):
				for x in dom:
					plt.plot([x,x],[0,1],'lightgrey',lw=2,zorder=0)
					plt.plot([0,1],[x,x],'lightgrey',lw=2,zorder=0)
		plt.scatter(x1,y1,marker='o',
			  		vmin=min(c1),vmax=max(c1),
					c=c1,cmap='jet',zorder=1)
		plt.colorbar(location='right')
		plt.xlim(-2*self.h,1+2*self.h)
		plt.ylim(-2*self.h,1+2*self.h)
		plt.tight_layout()
		plt.show()

	def	xy_to_e(self,x,y,u=False,v=False,p=False):
		try:
			assert self.uniform
		except:
			raise ValueError('virtual xy_to_e func not overwritten')
        
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12

		to_return = []

		if u:
			ux_ind,uy_ind = int(x/self.h),int(y/self.h)#+.5)
			u_el_ind = uy_ind*self.N+ux_ind
			u_e = self.mesh.u_elements[int(u_el_ind)]
			assert u_e.dom[0] <= x <= u_e.dom[1]
			assert u_e.dom[2] <= y <= u_e.dom[3]
			to_return.append(u_e)

		if v:
			vx_ind,vy_ind = int(x/self.h+.5),int(y/self.h)
			v_el_ind = vy_ind*(self.N+1)+vx_ind+self.mesh.el_counts[0]
			v_e = self.mesh.u_elements[int(v_el_ind)]
			assert v_e.dom[0] <= x <= v_e.dom[1]
			assert v_e.dom[2] <= y <= v_e.dom[3]
			to_return.append(v_e)

		if p:
			px_ind,py_ind = int(x/self.h+.5),int(y/self.h+.5)
			p_el_ind = py_ind*(self.N+1)+px_ind
			p_e = self.mesh.p_elements[int(p_el_ind)]
			assert p_e.dom[0] <= x <= p_e.dom[1]
			assert p_e.dom[2] <= y <= p_e.dom[3]
			to_return.append(p_e)
		
		if len(to_return) == 0:
			raise ValueError('both u and v set to False')
		elif len(to_return) == 1:
			return to_return[0]
		return to_return

	def	sol(self, weights=None,	sep=True, p=False):

		if weights is None:
			assert self._solved
			assert self.U is not None
			weights	= self.U

		def	u_solution(x,y):
			u_e = self.xy_to_e(x,y,u=True)

			u_val	= 0
			for	dof in u_e.dof_list:
				phi_val = phi_2d_eval(
					self.order,#[self.order,self.order-1],
					x,y,dof.h,dof.x,dof.y)
				u_val	+= weights[dof.ID]*phi_val

			return u_val
		return u_solution

		def	v_solution(x,y):
			v_e = self.xy_to_e(x,y,v=True)

			v_val	= 0
			for	dof in v_e.dof_list:
				phi_val = phi_2d_eval(
					self.order,#[self.order-1,self.order],
					x,y,dof.h,dof.x,dof.y)
				v_val	+= weights[dof.ID]*phi_val
			
			return v_val

		if p:
			def p_solution(x,y):
				p_e = self.xy_to_e(x,y,p=True)

				p_val	= 0
				for	dof in p_e.dof_list:
					phi_val = phi_2d_eval(
						[self.order-1,self.order-1],
						x,y,dof.h,dof.x,dof.y)
					p_val	+= weights[dof.ID]*phi_val
				
				return p_val

		if sep:
			if p:
				return u_solution, v_solution, p_solution
			return u_solution, v_solution
		
		def solution(x,y):
			if p:
				return u_solution(x,y),v_solution(x,y),p_solution(x,y)
			return u_solution(x,y),v_solution(x,y)
		return solution

	def	error(self,qpn=5,which=None):
		uh = self.sol(sep=True,weights=which)
		l2_err = 0#np.zeros(2)
		for	e in self.mesh.u_elements:
			if e.dim == 0:
				func = lambda x,y: (self.ufunc(x,y)-uh(x,y))**2
			else:
				func = lambda x,y: (self.vfunc(x,y)-vh(x,y))**2
			x0,x1,y0,y1	= e.dom
			val	= gauss(func,x0,x1,y0,y1,qpn)
			l2_err += val #[e.dim] += val
		return np.sqrt(l2_err)

	def split(self,myU):
		cut = self.mesh.dof_counts[0]
		return myU[:cut],myU[cut:]