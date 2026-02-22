import numpy as	np
from tqdm import tqdm
import matplotlib.pyplot as	plt
import scipy.linalg	as la
import scipy.sparse.linalg as sla
from scipy import sparse

from cubic_basis.nodal_grid.helpers_3d import *

class Node:
	def	__init__(self,ID,j,i,k,x,y,z,h):
		self.ID	= ID
		self.j = j
		self.i = i
		self.k = k
		self.x = x
		self.y = y
		self.z = z
		self.h = h
		self.elements =	{}

	def	add_element(self,e):
		if e.ID	not	in self.elements.keys():
			self.elements[e.ID]	= e

class Element:
	def	__init__(self,ID,j,i,k,x,y,z,h):
		self.ID	= ID
		self.j = j
		self.i = i
		self.k = k
		self.x = x
		self.y = y
		self.z = z
		self.h = h
		self.dof_ids = []
		self.dof_list =	[]
		self.fine =	False
		self.interface = False
		self.dom = [x,x+h,y,y+h,z,z+h]
		self.vol = h**3

	def	add_dofs(self,index,xlen,ylen):
		if len(self.dof_ids) !=	0:
			return
		for	kk in range(4):
			for	ii in range(4):
				for	jj in range(4):
					self.dof_ids.append(index+jj+ii*xlen+kk*xlen*ylen)
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
	def	set_interface(self):
		self.interface = True

class Mesh:
	def	__init__(self,N):
		self.N = N # number	of fine	elements 
				   # from x=0.5	to x=1.0
		self.h = 0.5/N
		self.dofs =	{}
		self.elements =	[]
		self.boundaries	= []
		self.periodic_sp =	[]
		self.interface_offset =	{0:([],[]),1:([],[])}
		
		self._make_coarse()
		self._make_fine()

		self._update_elements()

	def	_make_coarse(self):
		H =	self.h*2
		xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
		ydom = np.linspace(-H,1+H,self.N+3)
		zdom = np.linspace(-H,1+H,self.N+3)

		xlen,ylen,zlen = len(xdom),len(ydom), len(zdom)
		y_per_map =	{0:ylen-3,ylen-2:1,ylen-1:2}
		z_per_map =	{0:zlen-3,zlen-2:1,zlen-1:2}

		i_check,j_check,k_check	= [1,ylen-3],[1,xlen-3],[1,zlen-3]

		dof_id,e_id	= 0,0
		offset = xlen*ylen+xlen+1
		for	k,z	in enumerate(zdom):
			for	i,y	in enumerate(ydom):
				for	j,x	in enumerate(xdom):
					interface_element =	i in i_check or	j in j_check or	k in k_check
					self.dofs[dof_id] =	Node(dof_id,j,i,k,x,y,z,H)

					if (-H<x<.5) and (-H<y<1.) and (-H<z<1.):
						strt = dof_id-offset

						#strt_dof = self.dofs[strt]
						#try:
							#assert x-H == strt_dof.x
							#assert y-H == strt_dof.y
							#assert z-H == strt_dof.z
						#except:
							#print((strt_dof.x/H,strt_dof.y/H,strt_dof.z/H),(x/H,y/H,z/H))
							#raise ValueError()
						element	= Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						self.elements.append(element)
						if interface_element: element.set_interface()
						e_id +=	1

					if (.5 == x):# and (0 <= y <1):
						if 0<=y<1 and 0<=z<1:
							self.interface_offset[0][0].append(dof_id)
					if (0 == x):#	and	(0 <= y	<1):
						if 0<=y<1 and 0<=z<1:
							self.interface_offset[1][0].append(dof_id)

					if y < 0 or	y >= 1 or z	< 0	or z >=	1:	
						yind = y_per_map[i] if (y<0 or y>=1) else i
						zind = z_per_map[k] if (z<0 or z>=1) else k
						fill_id	= zind*(xlen*ylen)+yind*(xlen)+j
						self.periodic_sp.append([dof_id,fill_id,False])

					dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els =	e_id

	def	_make_fine(self):
		H =	self.h
		xdom = np.linspace(0.5-H,1+H,self.N+3)
		ydom = np.linspace(-H,1+H,2*self.N+3)
		zdom = np.linspace(-H,1+H,2*self.N+3)

		xlen,ylen,zlen = len(xdom),len(ydom), len(zdom)
		y_per_map =	{0:ylen-3,ylen-2:1,ylen-1:2}
		z_per_map =	{0:zlen-3,zlen-2:1,zlen-1:2}

		i_check,j_check,k_check	= [1,ylen-3],[1,xlen-3],[1,zlen-3]

		dof_id,e_id	= self.n_coarse_dofs,self.n_coarse_els
		for	k,z	in enumerate(zdom):
			for	i,y	in enumerate(ydom):
				for	j,x	in enumerate(xdom):
					interface_element =	i in i_check or	j in j_check or	k in k_check
					self.dofs[dof_id] =	Node(dof_id,j,i,k,x,y,z,H)

					if (0.5-H<x<1.)	and	(-H<y<1.) and (-H<z<1.):
						strt = dof_id-1-xlen*ylen-xlen
						element	= Element(e_id,j,i,k,x,y,z,H)
						element.add_dofs(strt,xlen,ylen)
						element.set_fine()
						self.elements.append(element)
						if interface_element: element.set_interface()
						e_id +=	1

					intrfc = False
					if (x == .5):# and (0 <= y <1):
						intrfc = True
						if True:#0<=y<1 and 0<=z<1:
							self.interface_offset[0][1].append(dof_id)
					if (x == 1):# and (0 <=	y <1):
						intrfc = True
						if True:#(0<=y<1) and 0<=z<1:
							self.interface_offset[1][1].append(dof_id)
					if x==1-2*H:
						self.boundaries.append(dof_id)
					elif y < 0 or	y >= 1 or z	< 0	or z >=	1:
						yind = y_per_map[i] if (y<0 or y>=1) else i
						zind = z_per_map[k] if (z<0 or z>=1) else k
						fill_id	= zind*(xlen*ylen)+yind*(xlen)+j+self.n_coarse_dofs
						self.periodic_sp.append([dof_id,fill_id,intrfc])

					dof_id += 1

	def	_update_elements(self):
		for	e in self.elements:
			e.update_dofs(self.dofs)

class Solver:
	def	__init__(self,N,u,f=None,qpn=5,meshtype=Mesh,opt=False):
		self.N = N
		self.ufunc = u
		self.ffunc = f #needs to be	overwritten	
		self.qpn = qpn

		self.mesh =	meshtype(N)
		self.h = self.mesh.h

		self.solved	= False
		self.C = None
		self.Id	= None
		self.opt = opt

		self._setup_constraints()
		try:
			phi_ops = np.load('node_interface_gauss_{}.npy'.format(qpn))
			[p,w] = np.polynomial.legendre.leggauss(qpn)
			self.quad_vals = phi_ops, p, w
		except:
			print('quad vals not loaded')
			self.quad_vals = compute_gauss(qpn)
			np.save('node_interface_gauss_{}.npy'.format(qpn),
		   			np.array(self.quad_vals[0]))


	def	_build_force(self):
		num_dofs = len(self.mesh.dofs)
		self.F = np.zeros(num_dofs)

		phi_gauss_vals,P,W	= self.quad_vals

		for	e in self.mesh.elements:
			func = lambda x,y,z: self.ffunc(x+e.x,y+e.y,z+e.z)
			f_vals = gauss_vals(func,0,e.h,0,e.h,0,e.h,self.qpn,P)
			scale = e.vol/8
			for test_id,dof in enumerate(e.dof_list):
				phi_vals = phi_gauss_vals[test_id]
				# val	= super_quick_gauss(f_vals,phi_vals,0,e.h,0,e.h,0,e.h,self.qpn,W)
				val = (f_vals*phi_vals)@W@W@W*scale
				self.F[dof.ID] += val

		del phi_gauss_vals
		del P
		del W

	def	_build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		Kr,	Kc,	Kd = [],[],[]

		base = None
		try:
			base = np.load('stiffness_3d.npy')
		except:
			print('stiffness not loaded')
			base = local_stiffness(qpn=self.qpn)
			np.save('stiffness_3d.npy',base)
		k = base*self.h


		for	e in tqdm(self.mesh.elements):
			scale = 1 if e.fine else 2
			for	test_id,dof	in enumerate(e.dof_list):
				Kr += [dof.ID]*len(e.dof_ids)
				Kc += e.dof_ids
				Kd += list(k[test_id]*scale)
		self.spK = sparse.coo_array((Kd,(Kr,Kc)),shape=(num_dofs,num_dofs)).tocsc()

		del Kr
		del Kc
		del Kd
		del k


	def	_build_mass(self):
		num_dofs = len(self.mesh.dofs)
		Mr,	Mc,	Md = [],[],[]
		
		base = None
		try:
			base = np.load('mass_3d.npy')
		except:
			print('mass not loaded')
			base = local_mass(qpn=self.qpn)
			np.save('mass_3d.npy',base)
		m = base*self.h**3

		for	e in tqdm(self.mesh.elements):
			scale =	1 if e.fine	else 8
			for	test_id,dof	in enumerate(e.dof_list):
				Mr += [dof.ID]*len(e.dof_ids)
				Mc += e.dof_ids
				Md += list(m[test_id]*scale)
		self.spM = sparse.coo_array((Md,(Mr,Mc)),shape=(num_dofs,num_dofs)).tocsc()

		del Mr
		del Mc
		del Md
		del m

	def	_setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id	= np.zeros(num_dofs)
		self.dirichlet = np.zeros(num_dofs)
		Cr,	Cc,	Cd = [],[],[]
		self.debug = [[],[]]

		v12,v32	= 9/16,-1/16
		vals,offs = [v32,v12,v12,v32],[1,0,-1,-2]

		for	j in range(2):
			c_side = np.array(self.mesh.interface_offset[j][0])
			f_side = np.array(self.mesh.interface_offset[j][1])
			
			assert c_side.size == (self.N)**2
			nc, nf = self.N, 2*self.N+3
			cgrid,ffull = c_side.reshape((nc,nc)), f_side.reshape((nf,nf))
			fgrid = ffull[1:-2,1:-2]

			self.Id[ffull] = 1

			Cr += list(fgrid[::2,::2].flatten())
			Cr += list(ffull[-2,1:-2:2])+list(ffull[1:-2:2,-2])+[ffull[-2,-2]]
			Cc += list(cgrid.flatten())+list(cgrid[0,:])+list(cgrid[:,0])+[cgrid[0,0]]
			Cd += [1]*(nc**2+2*nc+1)

			finds = fgrid[1::2,1::2]
			sz = finds.size
			for yv,yoff in zip(vals,offs):
				do_dirichlet = True
				Cr += list(fgrid[1::2,::2].flatten())+list(fgrid[::2,1::2].flatten())
				Cc += list(np.roll(cgrid,yoff,0).flatten())
				Cc += list(np.roll(cgrid,yoff,1).flatten())
				Cd += [yv]*2*nc**2

				do_colloc_side = True
				for zv,zoff in zip(vals,offs):
					cinds = np.roll(cgrid,(zoff,yoff),(0,1))
					Cr += list(finds.flatten())
					Cc += list(cinds.flatten())
					Cd += [yv*zv]*(sz)

					for side in [0,-1]:
						Cr += list(ffull[side,2:-1:2])+list(ffull[2:-1:2,side])
						Cc += list(cinds[-side-1,:])+list(cinds[:,-side-1])
						Cd += [yv*zv]*2*nc
						if do_colloc_side:
							yside = np.roll(cgrid,yoff,axis=1)
							zside = np.roll(cgrid,yoff,axis=0)
							csides = list(zside[-side-1,:])+[zside[-side-1,0]]
							csides += list(yside[:,-side-1])+[yside[0,-side-1]]
							Cr += list(ffull[side,1::2])+list(ffull[1::2,side])
							Cc += csides
							Cd += [yv]*len(csides)
							if do_dirichlet:
								Cr += list(ffull[-2,2:-1:2])+list(ffull[2:-1:2,-2])
								Cc += list(yside[0,:])+list(zside[:,0])
								Cd += [yv]*2*nc
								do_dirichlet = False

						for pair_i,f_ind in enumerate([-1,0]):
							Cr += [ffull[side,f_ind]]
							Cc += [cinds[-side-1,-pair_i]]
							Cd += [yv*zv]
					do_colloc_side = False
							

		dL,DL,replaceL = np.array(self.mesh.periodic_sp).T
		for	(d,D,doubleghost) in zip(dL,DL,replaceL):
			self.Id[d] = 1
			if doubleghost:
				pass
			else:
				Cr.append(d)
				Cc.append(D)
				Cd.append(1.)

		for	dof_id in self.mesh.boundaries:
			#Cr,Cc,Cd = inddel(Cr,Cc,Cd,dof_id)
			#assert dof_id not in Cr
			self.Id[dof_id]	= 1.
			dof	= self.mesh.dofs[dof_id]
			x,y,z =	dof.x,dof.y,dof.z
			self.dirichlet[dof_id] = self.ufunc(x,y,z)

		self.true_dofs = list(np.where(self.Id==0)[0])

		for	true_ind in	self.true_dofs:
			if true_ind	not	in self.mesh.boundaries:
				Cr.append(true_ind)
				Cc.append(true_ind)
				Cd.append(1.)

		spC_full =	sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_dofs))
		c_data = {}
		for	i,r	in enumerate(spC_full.row):
			tup	= (spC_full.col[i],spC_full.data[i])
			if r in	c_data.keys():
				c_data[r].append(tup)
			else:
				c_data[r] =	[tup]
		self.C_full = c_data

		Cc_array = np.array(Cc)
		masks =	[]
		for	true_dof in	self.true_dofs:
			masks.append(Cc_array==true_dof)
		for	j,mask in enumerate(masks):
			Cc_array[mask] = j
		Cc = list(Cc_array)

		num_true = len(self.true_dofs)
		self.spC = sparse.coo_array((Cd,(Cr,Cc)),shape=(num_dofs,num_true)).tocsc()

		del Cc
		del Cr 
		del Cd 
		del Cc_array
		del spC_full
		del masks

	def	solve(self):
		print('virtual not overwritten')

	def	vis_constraints(self,probs=None):
		fig,ax = plt.subplots(2,1,figsize=(50,100))
		pts,h = np.linspace(0,1,2*self.N+1,retstep=True)
		for axind in range(2):
			for i,p in enumerate(pts):
				c = 'r' if i%2 else 'k'
				ax[axind].plot([p,p],[-h,1+h],c+':',lw=5)
				ax[axind].plot([-h,1+h],[p,p],c+':',lw=5)
				ax[axind].plot([p,p],[0,1],c,lw=5)
				ax[axind].plot([0,1],[p,p],c,lw=5)
		markers = np.array([['s','^'],['v','o']])
		v12,v32	= 9/16,-1/16
		w =	[1,v12,v32,v12**2,v12*v32,v32**2]
		w = [round(v,8) for v in w]
		colors = ['C{}'.format(i) for i	in range(10) if i!=5]
		cols = {v:colors[i]	for	(i,v) in enumerate(w)}
		flags =	{v:False for v in w}
		labels = ['1','phi(1/2)','phi(3/2)','phi(1/2)^2','phi(1/2)phi(3/2)','phi(3/2)^2']
		labs = {v:lab for (v,lab) in zip(w,labels)}
		axshow = []
		unknown	= {}
		unknown_vals = []
		unknown_cs = {}
		#dists = []
		if probs is None: probs = list(self.C_full.keys())
		for ind in probs:#self.C_full:#,b in enumerate(self.Id):
			if self.Id[ind]:# b:
				row = self.C_full[ind]
				dof = self.mesh.dofs[ind]
				x,y,z = dof.x,dof.y,dof.z
				if dof.h==self.h:
					checkh = self.mesh.dofs[row[0][0]].h != dof.h
					checkL = int(len(row) > 1)
					checkx = int(x > .75)
					#assert(len(row)==67 or len(row)==1)
					if checkh:
						for pair in (row):
							cind,val = pair
							val = round(val,8)
							cdof = self.mesh.dofs[cind]
							cx,cy,cz = cdof.x,cdof.y,cdof.z
							assert cx == x or cx+1==x
							if cy-y > .5: ty=cy-1
							elif y-cy>.5: ty=cy+1
							else: ty=cy
							if cz-z > .5: tz=cz-1
							elif z-cz>.5: tz=cz+1
							else: tz=cz
								
							if val in flags:# or (cy==y and cz==z):
								#ydist,zdist = (ty-y),(tz-z)#/32/self.h, (tz-z)/32/self.h
								#dists += [ydist,zdist]
								ydist,zdist = (ty-y)/12, (tz-z)/12
								m = markers[int(ty==cy),int(tz==cz)]

								ax[checkx].plot([y],[z],color='k',marker='.',ms = 40)
								if flags[val] == False:
									if not checkL:
										ax[checkx].plot(y+ydist,z+zdist,marker=m,fillstyle='none',
						   								  ms=50,color=cols[val],label=labs[val])
									else:
										ax[checkx].plot(y+ydist,z+zdist,marker=m,
						   							  ms=20,color=cols[val],label=labs[val])
									flags[val] = True
									if checkx not in axshow: axshow.append(checkx)
								else:
									if not checkL:
										ax[checkx].plot(y+ydist,z+zdist,marker=m,fillstyle='none',
						   							  ms=50,color=cols[val])
									else:
										ax[checkx].plot(y+ydist,z+zdist,marker=m,
						   							  ms=20,color=cols[val])
								

							else:
								if dof.ID not in unknown:
									unknown[dof.ID]	= [val]
									unknown_cs[dof.ID] = [(cdof.x,cdof.y,cdof.h==self.h)]
								else:
									unknown[dof.ID].append(val)
									unknown_cs[dof.ID].append((cdof.x,cdof.y,cdof.h==self.h))
								if val not in unknown_vals:
									unknown_vals.append(val)
		for ind in axshow:
			ax[ind].legend(fontsize=20)
		for ind in range(2):
			ttl = 'x = 1.0' if ind else 'x = 0.5'
			ax[ind].set_title(ttl)
		plt.show()
		if len(unknown)==0:
			return flags#, dists
			return None, flags, None, None

		for	id in unknown.keys():
			dof	= self.mesh.dofs[id]
			m =	'o'	if dof.h==self.h else 's'
			for	(x,y,h)	in unknown_cs[id]:
				plt.plot([dof.x,x],[dof.y,y],'lightgrey')
				plt.plot(x,y,'k.')
			plt.plot(dof.x,dof.y,m)
		plt.title('unknowns')
		plt.show()
		print(unknown_vals)
		return unknown,	flags,w,unknown_vals
		return
 
	def	vis_mesh(self,corner=False,retfig=False):

		fig	= plt.figure()
		ax = fig.add_subplot(projection='3d')
		mk = ['^','o']
		for	ind,dof	in enumerate(self.mesh.dofs.values()):
			m =	mk[dof.h==self.mesh.h]
			c =	'C0' if	ind	in self.true_dofs else 'C1'
			cind = 2*(ind in self.true_dofs)+((ind in self.true_dofs)==(dof.h==self.mesh.h))
			c =	'C'+str(cind)
			alpha =	1 if ind in	self.true_dofs else	.5
			ax.scatter(dof.x,dof.y,dof.z,marker=m,color=c,alpha=alpha)

		plt.show()
		return

	def	vis_dof_sol(self,proj=False,err=False,fltr=False,fval=.9,dsp=False,myU=None,onlytrue=True):
		U =	self.U
		U =	myU	if myU is not None else	U
		id0,x0,y0,z0,c0	= [],[], [], [], []
		id1,x1,y1,z1,c1	= [],[], [], [], []
		for	dof	in self.mesh.dofs.values():
			if onlytrue	and	dof.ID in self.true_dofs:
				if dof.h ==	self.h:
					id1.append(dof.ID)
					x1.append(dof.x)
					y1.append(dof.y)
					z1.append(dof.z)
					val	= U[dof.ID]
					if err:	val	= abs(val-self.ufunc(dof.x,dof.y,dof.z))
					c1.append(val)


				else:
					id0.append(dof.ID)
					x0.append(dof.x)
					y0.append(dof.y)
					z0.append(dof.z)
					val	= U[dof.ID]
					if err:	val	= abs(val-self.ufunc(dof.x,dof.y,dof.z))
					c0.append(val)
		
		m =	['o' for v in c1]+['^' for v in	c0]
		
		if fltr:
			mx = max(c0)
			msk	= np.array(c0)>fval*mx
			id0	= np.array(id0)[msk]
			x0 = np.array(x0)[msk]
			y0 = np.array(y0)[msk]
			z0 = np.array(z0)[msk]
			c0 = np.array(c0)[msk]
			vals = np.array([x0,y0,z0,c0]).T
			if dsp:print(vals)

			mx = max(c1)
			msk	= np.array(c1)>fval*mx
			id1	= np.array(id1)[msk]
			x1 = np.array(x1)[msk]
			y1 = np.array(y1)[msk]
			z1 = np.array(z1)[msk]
			c1 = np.array(c1)[msk]
			vals = np.array([x1,y1,z1,c1]).T
			if dsp:print(vals)
			


		fig	= plt.figure(figsize=plt.figaspect(1))
		ax = fig.add_subplot(projection='3d')
		plot1 =	ax.scatter(x0,y0,z0,marker='^',c=c0,cmap='jet')
		fig.colorbar(plot1,location='left')
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		ax.set_zlim(-1.5*self.h,1+1.5*self.h)
		plt.show()

		fig	= plt.figure(figsize=plt.figaspect(1))
		ax = fig.add_subplot(projection='3d')
		plot2 =	ax.scatter(x1,y1,z1,marker='o',c=c1,cmap='jet')
		fig.colorbar(plot2,location='left')
		ax.set_xlim(-1.5*self.h,1+1.5*self.h)
		ax.set_ylim(-1.5*self.h,1+1.5*self.h)
		ax.set_zlim(-1.5*self.h,1+1.5*self.h)
		plt.show()

		if fltr	and	dsp: return	id0,id1

	def vis_dofs(self):
		fig,ax = plt.subplots(1,3,figsize=(15,5))
		for ind in range(3):
			ax[ind].set_xlim(-4*self.h,1+4*self.h)
			ax[ind].set_ylim(-4*self.h,1+4*self.h)

		size = 64#16#int((self.p+1)**2)

		pts = np.linspace(0-2*self.h,1+2*self.h,17+4)
		lines = []
		line_data = []
		for ind in range(3):
			for i,p in enumerate(pts):
				col = 'lightgrey' if i%2 else 'darkgrey'
				linev, = ax[ind].plot([p,p],[-2*self.h,1+2*self.h],col)
				lineh, = ax[ind].plot([-2*self.h,1+2*self.h],[p,p],col)
				lines += [linev,lineh]
				line_data += [([p,p],[-2*self.h,1+2*self.h]),([-2*self.h,1+2*self.h],[p,p])]

		blocks = []
		for ind in range(3):
			axblocks = []
			for _ in range(size):
				block, = ax[ind].plot([],[])
				axblocks.append(block)
			blocks.append(axblocks)
		dots = []
		for ind in range(3):
			dot_dof, = ax[ind].plot([],[],'ko',linestyle='None')
			dot_ghost, = ax[ind].plot([],[],'C7o',linestyle='None')
			dots += [(dot_ghost,dot_dof)]

		def update(n):
			dof = self.mesh.dofs[n]
			els = list(dof.elements.values())
			for i,data in enumerate(line_data):
				lines[i].set_data(data[0],data[1])

			true = n in self.true_dofs
			dots[0][true].set_data(dof.x,dof.y)
			dots[0][1-true].set_data([],[])
			dots[1][true].set_data(dof.x,dof.z)
			dots[1][1-true].set_data([],[])
			dots[2][true].set_data(dof.y,dof.z)
			dots[2][1-true].set_data([],[])

			for i in range(size):
				if i < len(dof.elements):
					e = els[i]
					x0,x1,y0,y1,z0,z1 = e.dom
					blocks[0][i].set_data([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0])
					blocks[1][i].set_data([x0,x0,x1,x1,x0],[z0,z1,z1,z0,z0])
					blocks[2][i].set_data([y0,y0,y1,y1,y0],[z0,z1,z1,z0,z0])
				else:
					for ind in range(3):
						blocks[ind][i].set_data([],[])
			return [lines,blocks,dots]
		interval = 400
		ani = FuncAnimation(fig, update, frames=len(self.mesh.dofs), interval=interval)
		plt.close()
		return HTML(ani.to_html5_video())

	def vis_elements(self):
		frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
		fig,ax = plt.subplots(figsize=(5,5))
		ax.set_xlim(-4*self.h,1+4*self.h)
		ax.set_ylim(-4*self.h,1+4*self.h)
	
		line, = ax.plot(frame[0],frame[1],'lightgrey')
		eline, = ax.plot([],[])
		dot, = ax.plot([],[],'ko',linestyle='None')

		def update(n):
			e = self.mesh.elements[n]
			line.set_data(frame[0],frame[1])
			eline.set_data(e.plot[0],e.plot[1])
			xs,ys = [],[]
			for dof in e.dof_list:
				xs.append(dof.x)
				ys.append(dof.y)
			dot.set_data(xs,ys)
			return [line,eline,dot]
		interval = 400
		ani = FuncAnimation(fig, update, frames=len(self.mesh.elements), interval=interval)
		plt.close()
		return HTML(ani.to_html5_video())
	def	xyz_to_e(self,x,y,z):
		n_els =	[self.N+1,2*self.N+1]
		n_x_els	= [self.N/2+1,self.N+1]
		
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		z -= (z==1)*1e-12
		fine = True	if x >=	0.5	else False
		x_ind =	int((x-fine*.5)/((2-fine)*self.h))
		y_ind =	int(y/((2-fine)*self.h))
		z_ind =	int(z/((2-fine)*self.h))
		el_ind = fine*self.mesh.n_coarse_els
		el_ind += z_ind*(n_els[fine]*n_x_els[fine])
		el_ind += y_ind*n_x_els[fine]+x_ind
		e =	self.mesh.elements[int(el_ind)]
		assert x >=	e.dom[0] and x <=e.dom[1]
		assert y >=	e.dom[2] and y <=e.dom[3]
		assert z >=	e.dom[4] and z <=e.dom[5]
		return e

	def	sol(self, weights=None):

		if weights is None:
			assert self.solved
			weights	= self.U

		def	solution(x,y,z,e=None):
			if e is None:
				e =	self.xyz_to_e(x,y,z)

			val	= 0
			for	dof	in e.dof_list:
				val	+= weights[dof.ID]*phi3_3d_eval(x,y,z,dof.h,dof.x,dof.y,dof.z)
			
			return val
		return solution

	def	error(self,myU=None):
		gauss_phi_vals,P,W	= self.quad_vals

		if myU is None:
			myU = self.U

		l2_err = 0.
		for	e in self.mesh.elements:
			uh_vals	= 0
			for	local_id,dof in	enumerate(e.dof_list):
				uh_vals	+= myU[dof.ID]*gauss_phi_vals[local_id]
			x0,x1,y0,y1,z0,z1 = e.dom
			
			u_vals = gauss_vals(self.ufunc,x0,x1,y0,y1,z0,z1,self.qpn,P)
			scale = e.vol/8
			val = ((u_vals-uh_vals)**2)@W@W@W*scale
			l2_err += val

		del gauss_phi_vals
		del P
		del W
		del uh_vals
		del u_vals
		return np.sqrt(l2_err)

class Laplace(Solver):
	def	__init__(self,N,u,f,qpn=5,meshtype=Mesh,disp=False):
		super().__init__(N,u,f,qpn,meshtype)
		if disp:print('constraints done')
		self.disp = disp

	def	solve(self,construct_only=False):
		self._build_stiffness()
		if self.disp:print('stiffness done')
		self._build_force()
		if self.disp:print('force done')
		self.LHS = self.spC.T @	self.spK @ self.spC
		self.RHS = self.spC.T.dot(-self.F -	self.spK.dot(self.dirichlet))
		if construct_only:
			return
		try:
			spx,conv = sla.cg(self.LHS,self.RHS,rtol=1e-14)
			assert conv==0
			self.U = self.spC.dot(spx) + self.dirichlet
			self.solved	= True
			if self.disp:print('solve done')
			return spx
		except:
			print('something went wrong')

class Projection(Solver):
	def	__init__(self,N,u,qpn=5,meshtype=Mesh,disp=False):
		super().__init__(N,u,u,qpn,meshtype)
		if disp:print('constraints done')
		self.disp = disp

	def	solve(self,construct_only=False):
		self._build_mass()
		if self.disp:print('mass done')
		self._build_force()
		if self.disp:print('force done')
		self.LHS = self.spC.T @	self.spM @ self.spC
		self.RHS = self.spC.T.dot(self.F -	self.spM.dot(self.dirichlet))
		if construct_only:
			return
		try:
			spx,conv = sla.cg(self.LHS,self.RHS,rtol=1e-14)
			assert conv==0
			self.U = self.spC.dot(spx) + self.dirichlet
			self.solved	= True
			if self.disp:print('solve done')
			return spx
		except:
			print('something went wrong')
