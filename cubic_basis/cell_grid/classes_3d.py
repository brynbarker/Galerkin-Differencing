import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy import sparse

from cubic_basis.cell_grid.helpers_3d import *
from cubic_basis.cell_grid.shape_functions_3d import phi3_dxx

keys = [None,0,1]
id = 0
LOOKUP = {}
for xk in keys:
    temp = {}
    for yk in keys:
        temp[yk] = id
        id += 1
    LOOKUP[xk] = temp

class Node:
	def __init__(self,ID,j,i,k,x,y,z,h):
		self.ID = ID
		self.j = j
		self.i = i
		self.k = k
		self.x = x
		self.y = y
		self.z = z
		self.h = h
		self.elements = {}

	def add_element(self,e):
		if e.ID not in self.elements.keys():
			self.elements[e.ID] = e

class Element:
	def __init__(self,ID,j,i,k,x,y,z,h):
		self.ID = ID
		self.j = j
		self.i = i
		self.k = k
		self.x = x
		self.y = y
		self.z = z
	    self.h = h
		self.dof_ids = []
		self.dof_list = []
		self.fine = False
		self.Interface = False
		self.side = [None,None]
		self.dom = [x,x+h,y,y+h,z,z+h]

	def add_dofs(self,strt,xlen,ylen):
		if len(self.dof_ids) != 0:
			return
		for ii in range(4):
			for jj in range(4):
				for kk in range(4):
					shift = xlen*ylen*kk+xlen*ii
				    self.dof_ids.append(strt+shift+jj)
		return

	def update_dofs(self,dofs):
		if len(self.dof_list) != 0:
			return
		for dof_id in self.dof_ids:
			dof = dofs[dof_id]
			dof.add_element(self)
			self.dof_list.append(dof)
		return

	def set_fine(self):
		self.fine = True
	def set_interface(self,xside,yside,zside):
		self.interface = True
		self.side = [xside,yside,zside]
 
		loc = [self.x,self.y,self.z]
		for dim,side in enumerate(self.side):
			if side is not None:
				self.dom[2*dim+1-side] = loc[dim]+self.h/2

class Mesh:
	def __init__(self,N):
		self.N = N # number of fine elements 
				   # from x=0.5 to x=1.0
		self.h = 0.5/N
		self.dofs = {}
		self.elements = []
		self.boundaries = []
		self.periodic = [[],[]]
		self.interface_offset = {0:[[] for _ in range(16)],
						    	 1:[[] for _ in range(16)]}
		
		self._make_coarse()
		self._make_fine()

		self._update_elements()

	def _make_coarse(self):
		H = self.h*2
		xdom = np.linspace(0-3*H/2,0.5+3*H/2,int(self.N/2)+4)
		ydom = np.linspace(-3*H/2,1+3*H/2,self.N+4)
		zdom = np.linspace(-3*H/2,1+3*H/2,self.N+4)

		xlen,ylen,zlen = len(xdom),len(ydom), len(zdom)

		i_check,j_check,k_check = [1,ylen-3],[1,xlen-3],[1,zlen-3]

		dof_id,e_id = 0,0
        for k,z in enumerate(zdom):
            zside = k==1 if k in k_check else None
		    for i,y in enumerate(ydom):
		    	yside = i==1 if i in i_check else None
		    	for j,x in enumerate(xdom):
		    		interface_element = i in i_check or j in j_check or k in k_check
		    		xside = j==1 if j in j_check else None
		    		self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

		    		if (-H<x<.5) and (-H<y<1.) and (-H<z<1.):
		    			strt = dof_id-1-xlen
		    			element = Element(e_id,j,i,k,x,y,z,H)
		    			element.add_dofs(strt,xlen,ylen)
		    			self.elements.append(element)
		    			if interface_element: element.set_interface(xside,yside,zside)
		    			e_id += 1

		    		#if x==H/2:
		    		#	self.boundaries.append(dof_id)
		    		if y < 2*H or y > 1.-2*H:
		    			self.periodic[0].append(dof_id)
    
		    		if (.5-2*H <= x) and (0 <= y <1):
		    			if x < .5-H: self.interface_offset[0][0].append(dof_id)
		    			elif x < .5: self.interface_offset[0][1].append(dof_id)
		    			elif x > .5+H: self.interface_offset[0][3].append(dof_id)
		    			else: self.interface_offset[0][2].append(dof_id)
		    		if (2*H >= x) and (0 <= y <1):
		    			if x < -H: self.interface_offset[1][3].append(dof_id)
		    			elif x < 0: self.interface_offset[1][2].append(dof_id)
		    			elif x > H: self.interface_offset[1][0].append(dof_id)
		    			else: self.interface_offset[1][1].append(dof_id)

		    		dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		H = self.h
		xdom = np.linspace(0.5-3*H/2,1.+3*H/2,self.N+4)
		ydom = np.linspace(-3*H/2,1+3*H/2,2*self.N+4)

		xlen,ylen = len(xdom),len(ydom)

		i_check,j_check = [1,ylen-3],[1,xlen-3]

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			yside = i==1 if i in i_check else None
			for j,x in enumerate(xdom):
				interface_element = i in i_check or j in j_check
				xside = j==1 if j in j_check else None
				self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

				if (0.5-H<x<1.) and (0-H<y<1.):
					strt = dof_id-1-xlen
					element = Element(e_id,j,i,x,y,H)
					element.add_dofs(strt,xlen)
					element.set_fine()
					self.elements.append(element)
					if interface_element: element.set_interface(xside,yside)
					e_id += 1

				if x==1-5*H/2:
					self.boundaries.append(dof_id)
				elif y < 2*H or y > 1.-2*H:
					self.periodic[1].append(dof_id)
				if (x <= .5+2*H) and (0 <= y <1):
					if x < .5-H: self.interface_offset[0][4].append(dof_id)
					elif x < .5: self.interface_offset[0][5].append(dof_id)
					elif x < .5+H: self.interface_offset[0][6].append(dof_id)
					else: self.interface_offset[0][7].append(dof_id)
				if (x >= 1-2*H) and (0 <= y <1):
					if x < 1-H: self.interface_offset[1][7].append(dof_id)
					elif x < 1: self.interface_offset[1][6].append(dof_id)
					elif x < 1+H: self.interface_offset[1][5].append(dof_id)
					else: self.interface_offset[1][4].append(dof_id)

				dof_id += 1

	def _update_elements(self):
		for e in self.elements:
			e.update_dofs(self.dofs)

class Solver:
	def __init__(self,N,u,f=None,qpn=5):
		self.N = N
		self.ufunc = u
		self.ffunc = f #needs to be overwritten 
		self.qpn = qpn

		self.mesh = Mesh(N)
		self.h = self.mesh.h

		self.solved = False
		self.C = None
		self.Id = None

		self._setup_constraints()
		self.quad_vals = compute_gauss(qpn)

	def _build_force(self):
		num_dofs = len(self.mesh.dofs)
		self.F = np.zeros(num_dofs)

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		g_interface,q_p,g_w = self.quad_vals

		for e in self.mesh.elements:
			y0,y1 = e.dom[2]-e.y, e.dom[3]-e.y
			x0,x1 = e.dom[0]-e.x, e.dom[1]-e.x
			func = lambda x,y: self.ffunc(x+e.x,y+e.y)
			f_vals = gauss_vals(func,x0,x1,y0,y1,self.qpn,q_p)
			dom_id = LOOKUP[e.side[0]][e.side[1]]
			for test_id,dof in enumerate(e.dof_list):
				phi_vals = g_interface[dom_id][test_id]
				val = super_quick_gauss(f_vals,phi_vals,x0,x1,y0,y1,self.qpn,g_w)

				self.F[dof.ID] += val

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))
		Kr, Kc, Kd = [],[],[]

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		
		ks = []
		for j in [None,0,1]:
			for i in [None,0,1]:
				ks.append(local_stiffness(self.h,
							  			  qpn=self.qpn,
										  xside=j,yside=i))

		for e in self.mesh.elements:
			k_id = LOOKUP[e.side[0]][e.side[1]]
			for test_id,dof in enumerate(e.dof_list):
				Kr += [dof.ID]*len(e.dof_ids)
				Kc += e.dof_ids
				Kd += list(ks[k_id][test_id])
				self.K[dof.ID,e.dof_ids] += ks[k_id][test_id]
		self.spK = sparse.coo_array((Kd,(Kr,Kc)),shape=(num_dofs,num_dofs)).tocsc()


	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))
		Mr, Mc, Md = [],[],[]

		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
		
		ms = []
		for j in [None,0,1]:
			for i in [None,0,1]:
				ms.append(local_mass(self.h,
							   		 qpn=self.qpn,
									 xside=j,yside=i))

		for e in self.mesh.elements:
			scale = 1 if e.fine else 4
			m_id = LOOKUP[e.side[0]][e.side[1]]
			for test_id,dof in enumerate(e.dof_list):
				Mr += [dof.ID]*len(e.dof_ids)
				Mc += e.dof_ids
				Md += list(ms[m_id][test_id]*scale)
				self.M[dof.ID,e.dof_ids] += ms[m_id][test_id]*scale
		self.spM = sparse.coo_array((Md,(Mr,Mc)),shape=(num_dofs,num_dofs)).tocsc()


	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)
		Cr, Cc, Cd = [],[],[]

		for j in range(2):
			c_side = np.array(self.mesh.interface_offset[j][:4])
			f_side = np.array(self.mesh.interface_offset[j][4:])

			v12,v32 = 9/16,-1/16
			v14,v34,v54,v74 = 105/128,35/128,-7/128,-5/128

			self.Id[f_side[0]] = 1
			self.C_full[f_side[0]] *= 0
			for ind,vhorz in enumerate([v32,v12,v12,v32]):
				for vvert, offset in zip([v74,v34,v14,v54],[2,1,0,-1]):
					self.C_full[f_side[0][::2],np.roll(c_side[ind],offset)] = vvert*vhorz/v32
					Cr += list((f_side[0][::2]).flatten())
					Cc += list(np.roll(c_side[ind],offset).flatten())
					Cd += [vvert*vhorz/v32]*(f_side[0][::2]).size
				for vvert, offset in zip([v54,v14,v34,v74],[1,0,-1,-2]):
					self.C_full[f_side[0][1::2],np.roll(c_side[ind],offset)] = vvert*vhorz/v32
					Cr += list((f_side[0][1::2]).flatten())
					Cc += list(np.roll(c_side[ind],offset).flatten())
					Cd += [vvert*vhorz/v32]*(f_side[0][1::2]).size
			for ind,vhorz in enumerate([v12,v12,v32]):
				self.C_full[f_side[0],f_side[ind+1]] = -vhorz/v32
				Cr += list((f_side[0]).flatten())
				Cc += list((f_side[ind+1]).flatten())
				Cd += [-vhorz/v32]*(f_side[0]).size


		for level in range(2):
			# lower are true dofs, upper are ghosts
			b0,b1,B2,B3,T0,T1,t2,t3 = np.array(self.mesh.periodic[level]).reshape((8,-1))
			ghost_list = np.array([b0,b1,t2,t3])
			self.C_full[ghost_list] *= 0.
			self.Id[ghost_list] = 1.
			Ds, ds = [T0,T1,B2,B3],[b0,b1,t2,t3]
			for (D,d) in zip(Ds,ds):
				self.C_full[d,D] = 1
				for ind in [0,1,2,3,-4,-3,-2,-1]:
					self.C_full[:,D[ind]] += self.C_full[:,d[ind]]
					self.C_full[d[ind],:] = self.C_full[D[ind],:]
			
		for dof_id in self.mesh.boundaries:
			Cr,Cc,Cd = inddel(Cr,Cc,Cd,dof_id)
			assert dof_id not in Cr
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			x,y = self.mesh.dofs[dof_id].x,self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(x,y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]
		return
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

	def solve(self):
		print('virtual not overwritten')

	def vis_constraints(self,alt=False):
		fig,ax = plt.subplots(1,figsize=(16,24))
		markers = np.array([['s','^'],['v','o']])
		
		v12,v32 = 9/16,-1/16
		if alt: v12,v32 =-1/16,9/16
		v14,v34,v54,v74 = 105/128,35/128,-7/128,-5/128
		w = [-1,-v12/v32]+[vh*vv/v32 for vh in [v12,v32] for vv in [v14,v34,v54,v74]]
		colors = ['C{}'.format(i) for i in range(10)] + ['magenta']
		colors += ['skyblue','limegreen','yellow','salmon','darkgoldenrod']
		cols = {v:colors[i] for (i,v) in enumerate(w)}
		flags = {v:False for v in w}
		labs = {v:str(v) for v in w}
		unknown = {}
		unknown_vals = []
		unknown_cs = {}
		for ind,b in enumerate(self.Id):
			if b:
				row = self.C_full[ind]
				dof = self.mesh.dofs[ind]
				x,y = dof.x,dof.y
				if dof.h==self.h:
					for cind,val in enumerate(row):
						if abs(val)>1e-12:
							val = round(val,8)
							cdof = self.mesh.dofs[cind]
							cx,cy = cdof.x,cdof.y
							if cdof.h!=dof.h or cy==y:
								if cx-x > .5: tx=cx-1
								elif x-cx>.5: tx=cx+1
								else: tx=cx
								if cy-y > .5: ty=cy-1
								elif y-cy>.5: ty=cy+1
								else: ty=cy

								
								m = markers[int(ty==cy),int(tx==cx)]
								ms = 40 if cdof.h!=self.h else 20
								ax.scatter([tx],[ty],color='k',marker=m,s=ms)
								ax.scatter([x],[y],color='grey',marker='o',s=ms)
								if val not in flags:
									if dof.x==.5:
										print(val, cdof.x, cdof.y)
									if dof.ID not in unknown:
										unknown[dof.ID] = [val]
										unknown_cs[dof.ID] = [(cdof.x,cdof.y,cdof.h==self.h)]
									else:
										unknown[dof.ID].append(val)
										unknown_cs[dof.ID].append((cdof.x,cdof.y,cdof.h==self.h))
									if val not in unknown_vals:
										unknown_vals.append(val)
								else:
									if flags[val]==False:
										ax.plot([x,tx],[y,ty],color=cols[val],label=labs[val])
										flags[val] = True
									else:
										ax.plot([x,tx],[y,ty],color=cols[val])
							
							
					
					
		ax.legend()
		plt.show()


		if len(unknown)==0:
			return None, flags

		for id in unknown.keys():
			dof = self.mesh.dofs[id]
			m = 'o' if dof.h==self.h else 's'
			for (x,y,h) in unknown_cs[id]:
				plt.plot([dof.x,x],[dof.y,y],'lightgrey')
				plt.plot(x,y,'k.')
			plt.plot(dof.x,dof.y,m)
		plt.title('unknowns')
		plt.show()
		print(unknown_vals)
		return unknown, flags
	def vis_dof_sol(self,proj=False,err=False,fltr=False,fval=.9,dsp=False,myU=None,onlytrue=False):
		U = self.U
		if myU is not None: U=myU
		id0,x0,y0,c0 = [],[], [], []
		id1,x1,y1,c1 = [],[], [], []
		for dof in self.mesh.dofs.values():
			#if onlytrue and dof.ID in self.true_dofs:
			if not onlytrue or dof.ID in self.true_dofs:
				if dof.h == self.h:
					id1.append(dof.ID)
					x1.append(dof.x)
					y1.append(dof.y)
					val = U[dof.ID]
					if err: val = abs(val-self.ufunc(dof.x,dof.y))
					c1.append(val)


				else:
					id0.append(dof.ID)
					x0.append(dof.x)
					y0.append(dof.y)
					val = U[dof.ID]
					if err: val = abs(val-self.ufunc(dof.x,dof.y))
					c0.append(val)
		
		m = ['o' for v in c1]+['^' for v in c0]
		
		if fltr:
			mx = max(c0)
			msk = np.array(c0)>fval*mx
			id0 = np.array(id0)[msk]
			x0 = np.array(x0)[msk]
			y0 = np.array(y0)[msk]
			c0 = np.array(c0)[msk]
			vals = np.array([x0,y0,c0]).T
			if dsp:print(vals)

			mx = max(c1)
			msk = np.array(c1)>fval*mx
			id1 = np.array(id1)[msk]
			x1 = np.array(x1)[msk]
			y1 = np.array(y1)[msk]
			c1 = np.array(c1)[msk]
			vals = np.array([x1,y1,c1]).T
			if dsp:print(vals)

		fig,ax = plt.subplots(1,2,figsize=(10,5))
		plot1 = ax[0].scatter(x0,y0,marker='^',c=c0,cmap='jet')
		fig.colorbar(plot1,location='left')

		plot2 = ax[1].scatter(x1,y1,marker='o',c=c1,cmap='jet')
		fig.colorbar(plot2,location='left')
		plt.show()

		if fltr and dsp: return id0,id1

	def vis_mesh(self,corner=False,retfig=False):

		fig = plt.figure()
		mk = ['^','o']
		
		ind,tick = 0,0
		while tick <= 1:
			if ind % 2:
				plt.plot([.5,1],[tick,tick],'grey')
				if tick > .5:
					plt.plot([tick,tick],[0,1],'grey')
			else:
				plt.plot([tick,tick],[0,1],'grey')
				plt.plot([0,1],[tick,tick],'grey')
			tick += self.h
			ind += 1

		for ind,dof in enumerate(self.mesh.dofs.values()):
			m = mk[dof.h==self.mesh.h]
			c = 'C0' if ind in self.true_dofs else 'C1'
			cind = 2*(ind in self.true_dofs)+((ind in self.true_dofs)==(dof.h==self.mesh.h))
			c = 'C'+str(cind)
			alpha = 1 if ind in self.true_dofs else .5
			plt.scatter(dof.x,dof.y,marker=m,color=c,alpha=alpha)

		plt.show()
		return

	def vis_dofs(self):
		frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
		fig,ax = plt.subplots(figsize=(5,5))
		ax.set_xlim(-4*self.h,1+4*self.h)
		ax.set_ylim(-4*self.h,1+4*self.h)

		size = 16#int((self.p+1)**2)
	
		line, = ax.plot(frame[0],frame[1],'lightgrey')
		blocks = []
		for _ in range(size):
			block, = ax.plot([],[])
			blocks.append(block)
		dot, = ax.plot([],[],'ko',linestyle='None')

		def update(n):
			dof = self.mesh.dofs[n]
			els = list(dof.elements.values())
			line.set_data(frame[0],frame[1])
			dot.set_data(dof.x,dof.y)
			for i in range(size):
				if i < len(dof.elements):
					e = els[i]
					blocks[i].set_data(e.plot[0],e.plot[1])
				else:
					blocks[i].set_data([],[])
			return [line,blocks,dot]
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

	def xy_to_e(self,x,y):
		n_x_els = [self.N/2+1,self.N+1]
        
		x -= (x==1)*1e-12
		y -= (y==1)*1e-12
		fine = True if x >= 0.5 else False
		x_ind = int((x-fine*.5)/((2-fine)*self.h)+.5)
		y_ind = int(y/((2-fine)*self.h)+.5)
		el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
		e = self.mesh.elements[int(el_ind)]
		assert x >= min(e.plot[0]) and x <= max(e.plot[0])
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		return e

	def sol(self, weights=None):

		if weights is None:
			assert self.solved
			weights = self.U

		def solution(x,y):
			e = self.xy_to_e(x,y)

			val = 0
			for dof in e.dof_list:
				val += weights[dof.ID]*phi3_2d_eval(x,y,dof.h,dof.x,dof.y)
			
			return val
		return solution

	def error(self,qpn=5):
		uh = self.sol()
		l2_err = 0.
		for e in self.mesh.elements:
			x0,x1,y0,y1 = e.dom
			func = lambda x,y: (self.ufunc(x,y)-uh(x,y))**2
			val = gauss(func,x0,x1,y0,y1,qpn)
			l2_err += val
		return np.sqrt(l2_err)

class Laplace(Solver):
	def __init__(self,N,u,f,qpn=5):
		super().__init__(N,u,f,qpn)
		self._setup_constraints()

	def solve(self):
		self._build_stiffness()
		self._build_force()
		self.LHS = self.C.T @ self.K @ self.C
		self.RHS = self.C.T @(-self.F - self.K @ self.dirichlet)
		#self.LHS = self.spC.T @ self.spK @ self.spC
		#self.RHS = self.spC.T.dot(-self.F - self.spK.dot(self.dirichlet))
		try:
			#x,conv = sla.cg(self.LHS,self.RHS,rtol=1e-14)
			#assert conv==0
			x = la.solve(self.LHS,self.RHS)
			self.U = self.C@x + self.dirichlet
			#self.U = self.spC.dot(x) + self.dirichlet
			self.solved = True
		except:
			print('something went wrong')


class Projection(Solver):
	def __init__(self,N,u,qpn=5):
		super().__init__(N,u,u,qpn)
		self._setup_constraints()

	def solve(self):
		self._build_mass()
		self._build_force()
		#self.LHS = self.spC.T @ self.spM @ self.spC
		#self.RHS = self.spC.T @ (self.F - self.spM @ self.dirichlet)
		self.LHS = self.C.T @ self.M @ self.C
		self.RHS = self.C.T @ (self.F - self.M @ self.dirichlet)
		try:
			#x,conv = sla.cg(self.LHS,self.RHS,rtol=1e-14)
			#assert conv==0
			x = la.solve(self.LHS,self.RHS)
			self.U = self.C@x + self.dirichlet
			self.solved = True
		except:
			print('something went wrong')

