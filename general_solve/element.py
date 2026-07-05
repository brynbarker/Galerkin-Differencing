import numpy as np

class Element:
	def __init__(self,ID,dim,inds,loc,h,ords,cart=True):
		for index in range(dim):
			if ords[index] == 0:
				loc[index] -= h/2
		self.ID = ID
		self.dim = dim
		self.h = h
		self.loc = loc
		self.ind = inds
		self.ords = ords
		self.regular = cart
		self.tri=False
		self.zigzag = False
		self.global_ID = ID

		if dim == 2:
			self.i,self.j = inds
			self.x,self.y = loc
			self.k,self.z = None, None
		elif dim == 3:
			self.i,self.j,self.k = inds
			self.x,self.y,self.z = loc
		else:
			raise ValueError('dim must be 2 or 3')

		self.bounds = []
		for x in loc:
			self.bounds.append(x)
			self.bounds.append(x+h)

		self.dof_lookup_ids = [] # lookup ids
		self.dof_ids = [] # not lookup ids
		self.dof_list = []
		self.fine = False
		self.interface = False
		self.dom = [[coord,coord+h] for coord in loc]
		self.mid = [self.x+self.h/2,self.y+self.h/2]
		
		x0,x1,y0,y1 = self.bounds
		self.to_plot = [[x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0]]

	def set_zigzag(self):
		self.zigzag = True

	def set_global_ID(self,shft):
		# return
		self.global_ID = self.ID+shft

			
	def add_dofs(self,strt,lens):
		# these are lookup ids!
		if len(self.dof_ids) != 0:
			return
		if self.dim == 2:
			return self.add_dofs_2d(strt,lens)
		else:
			return self.add_dofs_3d(strt,lens)

	def add_dofs_2d(self,strt,xlen):
		for ii in range(self.ords[1]+1):
			for jj in range(self.ords[0]+1):
				self.dof_lookup_ids.append(strt+xlen*ii+jj)
		return

	def add_dofs_3d(self,strt,xlen):
		for	kk in range(self.ords[2]+1):
			for	ii in range(self.ords[1]+1):
				for	jj in range(self.ords[0]+1):
					self.dof_lookup_ids.append(strt+jj+ii*xlen+kk*xlen*xlen)
		return

	def update_dofs(self,dofs):
		if len(self.dof_list) != 0:
			return
		for dof_lookup_id in self.dof_lookup_ids:
			dof = dofs[dof_lookup_id]
			dof.add_element(self)

			self.dof_list.append(dof)
			self.dof_ids.append(dof.ID)
		return

	def set_fine(self):
		self.fine = True

	def transform(self,nu,rho):
		return self.x+nu*self.h,self.y+rho*self.h

	def set_support(self,quads):
		self.quads = quads


	def set_interface(self):
		self.interface = True

	def fill(self,eps=0):
		x = [self.x+eps,self.x+self.h-eps]
		return x,self.y+eps,self.y+self.h-eps

	def check_loc(self,loc):
		check = True
		for d in range(self.dim):
			check_step = loc[d] >= self.dom[d][0] and loc[d] <= self.dom[d][1]
			check = check and check_step

		return check
			# assert loc[d] >= self.dom[d][0] and loc[d] <= self.dom[d][1]
	
	def get_dof_ids(self,id=None):
		return self.dof_ids

	def reassign_dof(self,old_dof,new_dof):
		ind = self.dof_ids.index(old_dof.ID)
		self.dof_ids[ind] = new_dof.ID
		self.dof_list[ind] = new_dof

		old_dof.remove_element(self.global_ID)
		new_dof.add_element(self)

abcd = {0:[(0,0),(1/2,1/4),(0,1),(0,-1/2)],
		2:[(1/2,1/4),(1/2,-1/4),(0,1/2),(0,1/2)],
		1:[(0,0),(1,0),(1/4,1/2),(-1/2,0)],
		3:[(1/4,1/2),(1/2,0),(-1/4,1/2),(1/2,0)]}

corner_shift = {i:abcd[i][0] for i in abcd}

gabcd = {0:[(0,0),(1,-1/2),(0,1),(0,1)],
		2:[(0,-1/2),(1,1/2),(0,2),(0,-1)],
		1:[(0,0),(1,0),(-1/2,1),(1,0)],
		3:[(-1/2,0),(2,0),(1/2,1),(-1,0)]}

tri_corner_shift = {i:gabcd[i][0] for i in gabcd}
tri_corner_shift = {0:(0,0),2:(1/2,-1/4),1:(0,0),3:(-1/4,1/2)}

inv =  {0:[-1/2,1,0,1,-1,0],
		2:[1/2,1,-1/2,0,1,0],
		1:[1,-1/2,0,1,0,-1],
		3:[1,1/2,-1/2,0,0,1]}

ginv = {0:[1/2,1,0,1,1,0],
		2:[-1/2,1,1/2,2,-1,0],
		1:[1,1/2,0,1,0,1],
		3:[1,-1/2,1/2,2,0,-1]}

ab = {0:[1,2,0,1,-2,0],
	  2:[-1,2,1,-1,-2,1],
	  1:[2,1,0,-2,1,0],
	  3:[2,-1,1,-2,-1,1]}

cstar = {0:(0,0),
		 2:(1,0),
		 1:(0,0),
		 3:(0,1)}

def get_trap_corners(K,L,dofloc,H):
	if dofloc == 'xside':
		if round(K*H,8) in [.25,.75]:
			corners = [(K,L),(K+1/2,L+1/4),(K,L+1),(K+1/2,L+3/4)]
		else:
			corners = [(K+1/2,L+1/4),(K+1,L),(K+1/2,L+3/4),(K+1,L+1)]
	elif dofloc == 'yside':
		if round(L*H,8) in [.25,.75]: 
			corners = [(K,L),(K+1,L),(K+1/4,L+1/2),(K+3/4,L+1/2)]
		else:
			corners = [(K+1/4,L+1/2),(K+3/4,L+1/2),(K,L+1),(K+1,L+1)]
	else:
		raise ValueError('not able to handle this case')
	return corners

def get_tri_corners(K,L,dofloc,H):
	if dofloc == 'xside':
		if round(K*H,8) in [.25,.75]:
			corners = [(K,L),(K+1/2,L-1/4),(K+1/2,L+1/4)]
		else:
			corners = [(K+1/2,L-1/4),(K+1,L),(K+1/2,L+1/4)]
	elif dofloc == 'yside':
		if round(L*H,8) in [.25,.75]: 
			corners = [(K,L),(K-1/4,L+1/2),(K+1/4,L+1/2)]
		else:
			corners = [(K-1/4,L+1/2),(K+1/4,L+1/2),(K,L+1)]
	else:
		raise ValueError('not able to handle this case')
	return corners

class NonCartElement(Element):
	def __init__(self,ID,dim,inds,loc,h,ords):
		super().__init__(ID, dim, inds, loc, h, ords, cart=False)
		self.K,self.L = self.x/self.h, self.y/self.h
		# inds = corner_nodes[0].ind
		# loc = corner_nodes[0].loc
		# h = max(corner_nodes[0].h,corner_nodes[1].h)
		# dim = corner_nodes[0].dim

		# self.dof_len = (ords[0]+1)*(ords[1]+1)
		# super().__init__(ID, dim, inds, loc, h, ords, cart=False)

		# self.corners = corner_nodes
		# self.dof_list = []
		# self.dof_ids = []
		# self.local_dof_ids = []

		# self.quads = [True]*4

		# for c in self.corners:
		# 	self.add_dof(c)
		# self._set_corners()

	def set_jacobian(self,Js_inv,Jt_det,Js_coefs):#d_j_dets,d_j_vals):
		self.Js_inv = Js_inv
		self.Jt_det = Jt_det
		self.Js_inv_coefs = Js_coefs

	def set_jacobian_eval(self,jt_det,jd_inv):#d_j_dets,d_j_vals):

		self.jt_det_eval = jt_det[self.map_type]
		self.jd_inv_eval = {i:jd_inv[i][1] for i in range(4)}
		self.jd_invmat_eval = {i:jd_inv[i][0] for i in range(4)}

		# J_vals,Jinv_vals = {},{}
		# for i in range(4):
		# 	J_vals[i] = d_j_vals[self.map_type][i][0]
		# 	Jinv_vals[i] = d_j_vals[self.map_type][i][1]

		# self.J_vals = J_vals
		# self.Jinv_vals = Jinv_vals

	def get_usub_det(self,quad_id=None,pts=None):
		if quad_id is not None:
			return self.jt_det_eval[quad_id]
		assert pts is not None
		return [self.Jt_det(pt[0],pt[1]) for pt in pts]

	def get_deriv_jac_vals(self,quad_id=None,pts=None):
		if quad_id is not None:
			return self.jd_inv_eval[quad_id]
		assert pts is not None
		return [self.Js_inv_coefs(pt[0],pt[1]) for pt in pts]


	def _set_corners(self,corners):
		self.corners = []
		for (kloc,lloc) in corners:
			self.corners.append((kloc*self.h,lloc*self.h))

	# def _preorder_dofs(self):
	# 	tmp_list = self.dof_list
	# 	return self._order_dofs(tmp_list)

	# def _order_dofs(self,tmp_list):

	# 	A,B,C,D,e0,e1 = tmp_list
	# 	if self.map_type%2 == 0:
	# 		self.dof_list = [e0,A,B,e1,C,D]
	# 	else:
	# 		self.dof_list = [e0,e1,A,C,B,D]

	# 	self.local_dof_ids = [dof.ID for dof in self.dof_list]

	# def set_dof_ids(self,id_shift):
	# 	for dof in self.dof_list:
	# 		if dof.h == self.h: # fine
	# 			self.dof_ids.append(dof.ID+id_shift)
	# 		else:
	# 			self.dof_ids.append(dof.ID)

	# def add_dof(self,dof):
	# 	if dof not in self.dof_list:
	# 		self.dof_list.append(dof)
	# 		self.local_dof_ids.append(dof.ID)
	# 		dof.add_element(self)

	# 	if len(self.dof_list) == self.dof_len:
	# 		self._preorder_dofs()

	def ginv_transform(self,x,y):
		xi_ind,eta_ind = self.map_type%2,self.map_type%2==0
		xnum,ynum,cnum,cden,xden,yden = ginv[self.map_type]

		pars = [x/self.h-self.K,y/self.h-self.L]
		main = pars[xi_ind]

		num = xnum*pars[0]+ynum*pars[1]+cnum
		den = cden+xden*pars[0]+yden*pars[1]

		vars = [main,num/den]
		xi_star = vars[xi_ind]
		eta_star = vars[eta_ind]
		return xi_star,eta_star


class TrapElement(NonCartElement):
	def __init__(self,ID,dim,inds,loc,h,ords):
		super().__init__(ID, dim, inds, loc, h, ords)

		self.tri = False
		self.tris = []

	def add_tri(self,newtri):
		self.tris.append(newtri)
	
	def update_dofs(self,dofs):
		super().update_dofs(dofs)

		for tri in self.tris:
			tri.add_lookup_ids(self.dof_lookup_ids)

	def set_corners(self,xside,yside):
		if xside:
			corners = get_trap_corners(self.K,self.L,'xside',self.h)
		elif yside:
			corners = get_trap_corners(self.K,self.L,'yside',self.h)
		else:
			raise ValueError('not able to handle this case')
		self._set_corners(corners)
		# self.corners = [c for c in self.corners]
		xa,xb,xc,xd = [n[0] for n in self.corners]
		ya,yb,yc,yd = [n[1] for n in self.corners]
		# self.x = xa
		# self.y = ya

		self.to_plot = [[xa,xb,xd,xc,xa],[ya,yb,yd,yc,ya]]

		if xb==xd:
			if ya < yb:
				self.map_type = 0
				# self.check_loc = lambda x: (xa<=x[0]<=xb) and (x[0]-xa<=2*(x[1]-ya)<=xa-x[0]+4*self.h)
				# self.x0, self.y0 = self.x, self.y
			else:
				self.map_type = 2
				# self.check_loc = lambda x: (xa<=x[0]<=xb) and (xa-x[0]<=2*(x[1]-ya)<=x[0]-xa+2*self.h)
				# self.x0, self.y0 = self.x-self.h/2, self.y-self.h/4
			self.fill = lambda t: [[xa+t,xb-t],[ya+t,yb+t],[yc-t,yd-t]]
			# self.mid = [(xa+xb)/2,(ya+yc)/2]

		else:
			if xa < xb:
				self.map_type = 1
				# self.check_loc = lambda x: (ya<=x[1]<=yb) and (x[1]-ya<=2*(x[0]-xa)<=ya-x[1]+4*self.h)
				self.fill = lambda t: [[xa+t,xb,xd,xc-t],[ya+t]*4,[ya+t,yb-t,yb-t,ya+t]]
				# self.x0, self.y0 = self.x-self.h/4, self.y-self.h/2
			else:
				self.map_type = 3
				# self.check_loc = lambda x: (ya<=x[1]<=yb) and (ya-x[1]<=2*(x[0]-xa)<=x[1]-ya+2*self.h)
				self.fill = lambda t: [[xb+t,xa,xc,xd-t],[yb-t,ya+t,ya+t,yb-t],[yb-t]*4]
				# self.x0, self.y0 = self.x, self.y
			# self.mid = [(xa+xc)/2,(ya+yb)/2]

		# ksh,lsh = corner_shift[self.map_type]
		# self.K,self.L = self.x/self.h-ksh, self.y/self.h-lsh


	def transform(self,nu,rho):
		(ax,ay),(bx,by),(cx,cy),(dx,dy) = abcd[self.map_type]

		x = self.h*(self.K+ax+bx*nu+cx*rho+dx*nu*rho)
		y = self.h*(self.L+ay+by*nu+cy*rho+dy*nu*rho)
		return x,y

	def inv_transform(self,x,y):
		nu_ind,rho_ind = self.map_type%2,self.map_type%2==0
		xnum,ynum,cnum,cden,xden,yden = inv[self.map_type]

		pars = [x/self.h-self.K,y/self.h-self.L]
		main = 2*(pars[nu_ind]+cnum)

		num = xnum*pars[0]+ynum*pars[1]+cnum
		den = cden+xden*pars[0]+yden*pars[1]

		vars = [main,num/den]
		nu = vars[nu_ind]
		rho = vars[rho_ind]
		return nu,rho

	def check_loc(self,loc):
		nu,rho = self.inv_transform(loc[0],loc[1])
		return 0 <= nu <= 1 and 0 <= rho <= 1
		# return min(alpha,beta,1-alpha-beta)>=0


	def gtransform(self,nu,rho):
		(ax,ay),(bx,by),(cx,cy),(dx,dy) = gabcd[self.map_type]

		x = self.h*(self.K+ax+bx*nu+cx*rho+dx*nu*rho)
		y = self.h*(self.L+ay+by*nu+cy*rho+dy*nu*rho)
		return x,y

	def phi_input_local(self,nu,rho,i=0,j=0):
		ind0,ind1 = self.map_type%2,self.map_type%2==0
		vars = [nu,rho]
		v0,v1 = vars[ind0],vars[ind1]
		
		if self.map_type < 2:
			c_main = 0
			a,b,c,d = 0,1,2,-1
			den = [2,1]
		else:
			c_main = 1
			a,b,c,d = 1,-1,1,1
			den = [3,-1]

		main = (v0+c_main)/2
		other = (a+b*v0+c*v1+d*v0*v1)/(den[0]+den[1]*v0)

		stars = [main,other]
		xi_star = stars[ind0]
		eta_star = stars[ind1]
		return xi_star+i,eta_star+j

	def phi_input_global(self,x,y,i=0,j=0):
		xi_star,eta_star = self.ginv_transform(x,y)
		return xi_star+i,eta_star+j

class TriElement(NonCartElement):
	def __init__(self,ID,dim,inds,loc,h,ords,sh_dim=0):
		super().__init__(ID, dim, inds, loc, h, ords)
		# self.dof_len -= 1
		self.tri = True
		self.quads = {}
		self.trap_count = 0



	# def add_traps(self,trap_list):
	# 	self.traps = []
	# 	for trap in trap_list:
	# 		if trap is not None:
	# 			if (trap.K==self.K) or (trap.L==self.L):
	# 				self.traps.append(trap)

	def add_lookup_ids(self,trap_lookup_ids):
		self.trap_count += 1
		for new_id in trap_lookup_ids:
			if new_id not in self.dof_lookup_ids:
				self.dof_lookup_ids.append(new_id)
		self.dof_lookup_ids = sorted(self.dof_lookup_ids)


	def set_corners(self,xside,yside):
		if xside:
			self.ind_shift = [self.ind[0]-1,self.ind[1]]
			corners = get_tri_corners(self.K,self.L,'xside',self.h)
		elif yside:
			self.ind_shift = [self.ind[0],self.ind[1]-1]
			corners = get_tri_corners(self.K,self.L,'yside',self.h)
		else:
			raise ValueError('not able to handle this case')
		self._set_corners(corners)
		# self.corners = [c for c in self.corners]
		xa,xb,xc = [n[0] for n in self.corners]
		ya,yb,yc = [n[1] for n in self.corners]

		self.to_plot = [[xa,xb,xc,xa],[ya,yb,yc,ya]]

		if xb==xc:
			self.map_type = 0
			self.fill = lambda t: [[xa+t,xb-t],[ya,yb+t],[ya,yc-t]]
		elif xa==xc:
			self.map_type = 2
			self.fill = lambda t: [[xa+t,xb-t],[ya+t,yb],[yc-t,yb]]
		elif yb==yc:
			self.map_type = 1
			self.fill = lambda t: [[xb+t,xa,xc-t],[yb-t,ya+t,yc-t],[yb-t]*3]
		else:
			assert ya==yc
			self.map_type = 3
			self.fill = lambda t: [[xa+t,xb,xc-t],[ya+t]*3,[ya+t,yb-t,ya+t]]

		# ksh,lsh = tri_corner_shift[self.map_type]
		# self.K,self.L = self.x/self.h-ksh, self.y/self.h-lsh

	def inv_transform(self,x,y):
		xa,ya,ca,xb,yb,cb = ab[self.map_type]
		xfrac,yfrac = x/self.h-self.K, y/self.h-self.L
		alpha = xa*xfrac+ya*yfrac+ca
		beta = xb*xfrac+yb*yfrac+cb
		return alpha,beta

	def check_loc(self,loc):
		alpha,beta = self.inv_transform(loc[0],loc[1])
		return min(alpha,beta,1-alpha-beta)>=0

	def transform(self,alpha,beta):
		xc,yc = cstar[self.map_type]
		xind,yind = self.map_type%2,self.map_type%2==0
		sgn = 1 if self.map_type<2 else -1

		v0 = sgn*self.h*(alpha+beta)/2
		v1 = self.h*(alpha-beta)/4
		vars = [v0,v1]

		x = vars[xind] + self.h*(self.K+xc)
		y = vars[yind] + self.h*(self.L+yc)
		return x,y

	def ginv_transform(self, x, y):
		xb,yb,cb = ab[self.map_type][-3:]
		beta = xb*(x/self.h-self.K)+yb*(y/self.h-self.L)+cb
		stars = list(super().ginv_transform(x, y))
		shift = beta/(2+beta)

		stars[1-self.map_type%2] -= shift
		xi_star,eta_star = stars
		return xi_star, eta_star

	def phi_input_local(self,alpha,beta):
		tmp = (alpha+beta)/2
		main = tmp if self.map_type<2 else 1-tmp
		other = alpha/(2+alpha+beta)-beta/(2+beta)

		xi_ind,eta_ind = self.map_type%2,self.map_type%2==0
		vars = [main,other]
		xi_star = vars[xi_ind]
		eta_star = vars[eta_ind]
		return xi_star,eta_star

	def phi_input_global(self,x,y):
		return self.ginv_transform(x,y)

	# def _order_dofs(self,tmp_list):

	# 	A,B,C,e0,e1 = tmp_list
	# 	if self.map_type < 2:
	# 		self.dof_list = [e0,e1,A,B,C]#A,B,e1,C,D]
	# 		self.ref_shifts = [[-1,(1,-1,1)],[-1,(1,-1,0)],[0,None],[1,(0,1,0)],[1,(0,1,1)]]
	# 	else:
	# 		self.dof_list = [e0,e1,A,C,B]#,D]
	# 		self.ref_shifts = [[-1,(1,-1,1)],[-1,(1,-1,0)],[0,(0,1,0)],[0,(0,1,1)],[1,None]]

	# 	self.local_dof_ids = [dof.ID for dof in self.dof_list]

class PseudoElement:
	def __init__(self):
		self.dof_id_lists = {0:{},1:{}}

		self.comp = None
		self.p0 = False

	def set_comp(self,comp):
		self.comp = comp

	def set_const(self):
		self.p0 = True

	def add_dof_ids(self,dim,q_id,dof_ids=None):
		self.dof_id_lists[dim][q_id] = dof_ids

	def get_dof_ids(self,q_id,dim=None):
		if dim is None:
			dim = self.comp

		if self.p0:
			return self.dof_id_lists[dim][q_id]
		if dim == 0:
			return self.dof_id_lists[dim][q_id%2]
		else:
			return self.dof_id_lists[dim][int(q_id/2)]
