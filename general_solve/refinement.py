import numpy as	np

refinement_type	= {'uniform':0,
				   'finecenter':1,
				   'coarsecenter':2}

square_refinement_type	= {'finecenter':0,
				   'coarsecenter':1}

stripe_refinement_type	= {'vertfinecenter':0,
				   		   'vertcoarsecenter':1,
						   'horzfinecenter':2,
						   'horzcoarsecenter':3}

# the way this works is we 
# 1. setup the info
# 2. get the info
class RefinementPattern:
	def	__init__(self,name,dofloc,N,dim,ords):#=[3,3]):
		self.name =	name
		self.N = N
		self.h = 1/N
		self.dim = dim
		self.cell =	dofloc=='cell'
		self.node =	dofloc=='node'
		self.xside = dofloc=='xside'
		self.yside = dofloc=='yside'
		self.ords =	ords
		 
		self.shifts_R =	[int((ord-1)/2)	for	ord	in self.ords]
		self.shifts_L =	[int(ord/2)	for	ord	in self.ords]
		self.shifts_T =	[ord-1 for ord in self.ords]

		self.supports_R = [int(ord/2) for ord in ords]
		self.supports_L = [int((ord-1)/2) for ord in ords]

		self.d_data = {i:[[],[]] for i in range(2)} # ind list and loc list
		self.e_data = {i:[[],[],[]] for i in range(2)} # ind list and loc list and quads
		self.b_data = {i:[[],[]] for i in range(2)} # periodic and dirichlet
		self.i_data = {i:[[],[],[]] for i in range(2)}
		self.g_data = {i:[[],[]] for i in range(2)}

	def get_patch_id(self,loc):
		return 0

	def	_get_el_quads(self,start,H,check):
		quads =	[]

		if self.dim	== 2:
			shifts = [[0,0],[1,0],[0,1],[1,1]]			
		if self.dim	== 3:
			shifts = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]			
		
		for	shift in shifts:
			corner = [og+H*shft	for	(og,shft) in zip(start,shift)]
			quads.append(check(corner))

		return quads

	def	_closest_point(self,loc,H=None):
		# must be overwritten
		return None
		if H is None:
			H = self.h
		# find point going inward
		if False:#self.rtype == 1:
			shifts,dirs	= [],[]
			for	x in loc:
				if x < .25 or x	> .75:
					ops	= [abs(.25-x),abs(.75-x)]
					minloc = np.argmin(ops)
					sgn	= 1	if minloc==0 else -1

					shifts.append(ops[minloc])
					dirs.append(sgn)
				else:
					shifts.append(0)
					dirs.append(1)

			if 0 in	shifts or shifts[0]/shifts[1]==1:
				pass
			else:
				short_ind =	np.argmin(shifts)
				shifts[short_ind] =	shifts[short_ind]*3/2

			nearest_point =	[]
			for	x,shft,dir in zip(loc,shifts,dirs):
				nearest_point.append(x+shft*dir)
			return nearest_point

		# find point going out
		side_vals =	[.25,.75]
		ops	= [(abs(.25-x),abs(.75-x)) for x in	loc]
		sides =	[np.argmin(op) for op in ops]
		vals = [min(op)	for	op in ops]

		mindist	= min(vals)
		axes = [i for i	in range(self.dim) if vals[i]==mindist]
		nearest_point =	loc.copy()
		for	ax in axes:
			cont = True
			if self.ords[ax] %2 == 0:
				cont = False
				shift = (side_vals[sides[ax]] - loc[ax])/H
				if shift < 0 and abs(shift) < 1+self.supports_L[ax]:
					cont = True
				if shift > 0 and shift < 1+self.supports_R[ax]:
					cont = True
			if cont:
				nearest_point[ax] =	side_vals[sides[ax]]
		return nearest_point

	def _all_d(self,func,loc):
		try:
			return sum([func(x)	for	x in loc]) == self.dim
		except:
			return sum([func(x,d) for d,x in enumerate(loc)]) == self.dim
	
	def _not_all_d(self,func,loc):
		try:
			return sum([func(x)	for	x in loc]) != self.dim
		except:
			return sum([func(x,d) for d,x in enumerate(loc)]) != self.dim

	def _at_least_one(self,func,loc):
		try:
			return sum([func(x)	for	x in loc]) >= 1
		except:
			return sum([func(x,d) for d,x in enumerate(loc)]) >= 1

	def	_setup_info(self,coarse=True):
		if coarse:
			H =	self.h
			myN = self.N
		else:
			H = self.h/2
			myN = 2*self.N
		start =	0 if (self.node or self.xside) else 0-H/2
		end	= 1 if (self.node or self.xside) else 1+H/2
		xdom = np.linspace(start-H*self.shifts_L[0],
						  end+H*self.shifts_R[0],
						  myN+1+self.shifts_T[0]+self.cell+self.yside)
		
		start =	0 if (self.node or self.yside) else 0-H/2
		end	= 1 if (self.node or self.yside) else 1+H/2
		ydom = np.linspace(start-H*self.shifts_L[1],
						  end+H*self.shifts_R[1],
						  myN+1+self.shifts_T[1]+self.cell+self.xside)

		doms = [xdom,ydom]
		if self.dim	== 3:
			zdom = np.linspace(start-H*self.shifts_L[2],
						  end+H*self.shifts_R[2],
						  myN+1+self.shifts_T[2]+self.cell)
			doms.append(zdom)

		edges =	{}
		for	i in range(self.dim):

			if i==0:
				offset = (self.cell+self.yside)*H/2
			if i==1:
				offset = (self.cell+self.xside)*H/2

			edge1a = 0.25 -	offset - H*self.shifts_L[i]
			edge1b = 0.25 +	offset + H*self.shifts_R[i]
			edge2a = 0.75 -	offset - H*self.shifts_L[i]
			edge2b = 0.75 +	offset + H*self.shifts_R[i]

			edges[i] = [edge1a,edge1b,edge2a,edge2b]

		# for quad checks
		center =	lambda x: .25 <= x <= .75
		loose_center	= lambda x:	.25	< x	< .75
		domain =	lambda x: 0	<= x <=	1
		
		# for dof search
		periodic_check = lambda	x: 0 <=	x <	1
		periodic_check_full	= lambda loc: self._not_all_d(periodic_check,loc)
		dirichlet_check = lambda i,d: i in [0,len(doms[d])-1]
		dirichlet_check_full = lambda index: self._at_least_one(dirichlet_check,index)

		i_edges	= {}
		extraL, extraR = [],[]
		for	i in range(self.dim):
			same = (i==0 and self.yside) or (i==1 and self.xside)
			if self.cell or same:
				i_edges[i] = [val for val in edges[i]]
				extraL.append(0)
				extraR.append(0)
			else:
				i_edges[i] = [.25,.25,.75,.75]
				extraL.append(H*self.shifts_L[i])
				extraR.append(H*self.shifts_R[i])
			
		far_in = lambda x,d: i_edges[d][1] <= x <= i_edges[d][2]
		far_out = lambda x,d: i_edges[d][0] >= x or x >= i_edges[d][3]

		block =	lambda x,d: i_edges[d][0]-extraL[d] <=	x <= i_edges[d][-1]+extraR[d]
		slice =	lambda x,d: (i_edges[d][0]<=x<=i_edges[d][1]) or (i_edges[d][2]<=x<=i_edges[d][-1])

		funcs = [center,loose_center, domain, far_in, far_out,
		   		 periodic_check_full, dirichlet_check_full, block, slice]
		return doms, [edges,i_edges], funcs

	def _setup_coarse_info(self):
		tmp = self._setup_info()
		### must be overwritten
	
	def _setup_fine_info(self):
		tmp = self._setup_info(coarse=False)
		### must be overwritten
  
	def _get_info(self,coarse=True):
		if coarse:
			doms, checks = self._setup_coarse_info()
			H = self.h
			L = 0
		else:
			doms, checks = self._setup_fine_info()
			H = self.h/2
			L = 1
		check, echeck, periodic, dirichlet = checks[:4]
		low_support, quad, interface, ghost = checks[-4:]

		if self.dim	== 2:
			xdom,ydom = doms
			for	i,y	in enumerate(ydom):
				for	j,x	in enumerate(xdom):
					if check([x,y]):
						self.d_data[L][0].append([i,j])
						self.d_data[L][1].append([x,y])
						self.b_data[L][0].append(periodic([x,y]))
						self.b_data[L][1].append(dirichlet([j,i]))
						self.i_data[L][2].append(low_support([x,y]))
					if echeck([x,y]):
						self.e_data[L][0].append([i,j])
						self.e_data[L][1].append([x,y])
						self.e_data[L][2].append(self._get_el_quads([x,y],H,quad))
					if interface([x,y]):
						self.i_data[L][0].append([i,j])
						if ghost([x,y]):
							nearest_point =	self._closest_point([x,y])
						else:
							nearest_point =	None
						self.i_data[L][1].append(nearest_point)

		#if self.dim	== 3:
		#	xdom,ydom,zdom = doms
		#	for	k,z	in enumerate(zdom):
		#		for	i,y	in enumerate(ydom):
		#			for	j,x	in enumerate(xdom):
		#				if check([x,y,z]):
		#					d_ind_list.append([i,j,k])
		#					d_loc_list.append([x,y,z])
		#					d_periodic.append(periodic_check_full([x,y,z]))
		#					d_dirichlet.append([dirichlet_check(i,j,k)])
		#					d_square.append(low_support_square([x,y,z]))
		#				if echeck([x,y,z]):
		#					e_ind_list.append([i,j,k])
		#					e_loc_list.append([x,y,z])
		#					e_quads.append(self._get_el_quads([x,y,z],H,quadcheck))
		
		d_info = self.d_data[L] + self.b_data[L] + [self.i_data[L][2]]
		return d_info,self.e_data[L],self.i_data[L][:2],[len(dom) for dom in doms]

	def	get_coarse_info(self):
		return self._get_info(coarse=True)
	def	get_fine_info(self):
		return self._get_info(coarse=False)
	

class UniformRefinement(RefinementPattern):
	def	__init__(self,name,dofloc,N,dim,ords):#=[3,3]):
		super().__init__(name,dofloc,N,dim,ords)
		self.rshade = ['none','all']

	def _closest_point(self, loc, H=None):
		return super()._closest_point(loc, H)

	def _setup_coarse_info(self):
		H = self.h
		doms,edge_dicts,funcs = super()._setup_info()
		edges, i_edges = edge_dicts
		center,loose_center, domain, far_in, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]

		check =	lambda x: True

		emini =	lambda x: -H < x < 1
		echeck = lambda	loc: self._all_d(emini,loc)

		interface	= lambda x:	True
		quad =	lambda loc:	self._all_d(domain,loc)
		low_support = lambda loc: False

		ghost = lambda loc: False

		checks = [check, echeck, periodic, dirichlet,
				  low_support, quad, interface, ghost]
		return doms, checks

	def _setup_fine_info(self):
		doms = [[] for i in range(self.dim)]
		tmp = lambda loc: False
		return doms, [tmp]*8

class StripeRefinement(RefinementPattern):
	def	__init__(self,name,dofloc,N,dim,ords):#=[3,3]):
		super().__init__(name,dofloc,N,dim,ords)
		self.rtype = stripe_refinement_type[name]
		self.rdim = int(self.rtype/2) # vertical or horizontal stripe

		rindex_to_shade = {0:['vstripe','vedge'],#vfine
						   1:['vedge','vstripe'],#vco
						   2:['hstripe','hedge'],#hfine
						   3:['hedge','hstripe']}
		self.rshade = rindex_to_shade[self.rtype]

	def get_patch_id(self, loc):
		in_stripe = .25 <= loc[self.rdim] < .75

		stripe_id = 1 - (self.rtype%2)

		if in_stripe:
			return stripe_id
		else:
			return 1-stripe_id


	def _closest_point(self, loc, H=None):
		# find point going out
		side_vals =	[.25,.75]
		ops	= (abs(.25-loc[self.rdim]),abs(.75-loc[self.rdim]))
		nearest_point = np.copy(loc)

		nearest_point[self.rdim] = side_vals[np.argmin(ops)]

		return nearest_point

	def stripe_checks(self,H,edges,domain,center):
		rdim = int(self.rtype/2) # vertical or horizontal stripe
		check = lambda loc: edges[rdim][0]<= loc[rdim] <= edges[rdim][-1]
		emini =	lambda x: .25-H	< x	< .75
		emini_nonr = lambda x: -H < x < 1
		echeck = lambda loc: emini(loc[rdim]) and emini_nonr(loc[1-rdim])
		quad =	lambda loc:	domain(loc[1-rdim]) and center(loc[rdim])

		return check, echeck, quad

	def edge_checks(self,H,edges,domain,loose_center):
		rdim = int(self.rtype/2) # vertical or horizontal stripe
		check = lambda loc: edges[rdim][1] >= loc[rdim] or edges[rdim][2]<=loc[rdim]
		emini =	lambda x: -H < x < .25 or .75-H < x < 1
		emini_nonr = lambda x: -H < x < 1
		echeck = lambda loc: emini(loc[rdim]) and emini_nonr(loc[1-rdim])
		quad =	lambda loc:	self._all_d(domain,loc) and not loose_center(loc[rdim])
		#quad =	lambda loc:	domain(loc[1-rdim]) and not loose_center(loc[rdim])

		return check, echeck, quad

	def _setup_coarse_info(self):
		H = self.h
		doms,edge_dicts,funcs = super()._setup_info()
		edges, i_edges = edge_dicts
		center,loose_center, domain, far_in, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]


		if self.rtype % 2: # coarse stripe
			check, echeck, quad = self.stripe_checks(H,edges,domain,center)
		else: # coarse edges
			check, echeck, quad = self.edge_checks(H,edges,domain,loose_center)

		rdim = int(self.rtype/2) # vertical or horizontal stripe
		interface = lambda loc: slice(loc[rdim],rdim)

		low_support = lambda loc: False
		ghost = lambda loc: False

		checks = [check, echeck, periodic, dirichlet,
				  low_support, quad, interface, ghost]
		return doms, checks

	def _setup_fine_info(self):
		H = self.h/2
		doms,edge_dicts,funcs = super()._setup_info(coarse=False)
		edges, i_edges = edge_dicts
		center,loose_center, domain, far_in, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]


		rdim = int(self.rtype/2) # vertical or horizontal stripe
		# here is what far in is doing
		# let's look at the values for edges[rdim][1,2]
		#far_in = lambda x,d: edges[d][1] <= x <= edges[d][2]
		if self.rtype % 2: # fine edges
			check, echeck, quad = self.edge_checks(H,edges,domain,loose_center)
			myfar_in = lambda x,d: i_edges[d][1] <= x <= i_edges[d][2]
			ghost = lambda loc: myfar_in(loc[rdim],rdim)
		else: # fine stripe
			check,echeck,quad = self.stripe_checks(H,edges,domain,center)
			ghost = lambda loc: far_out(loc[rdim],rdim)

		interface = lambda loc: slice(loc[rdim],rdim)

		low_support = lambda loc: False

		checks = [check, echeck, periodic, dirichlet,
				  low_support, quad, interface, ghost]
		return doms, checks

class SquareRefinement(RefinementPattern):
	def	__init__(self,name,dofloc,N,dim,ords):#=[3,3]):
		super().__init__(name,dofloc,N,dim,ords)
		self.rtype = square_refinement_type[name]
		rindex_to_shade ={0:['in','out'],1:['out','in']}
		self.rshade = rindex_to_shade[self.rtype]

	def get_patch_id(self, loc):
		check = lambda x: .25 <= x < .75
		in_center = self._all_d(check,loc)

		if in_center:
			return 1-self.rtype
		else:
			return self.rtype

	def _closest_point(self, loc, H=None):
		# find point going out
		if .25 in loc or .75 in loc:
			if self._all_d(lambda x:.25 <= x <= .75,loc):
				return loc

		side_vals =	[.25,.75]
		ops	= [(abs(.25-x),abs(.75-x)) for x in	loc]
		sides =	[np.argmin(op) for op in ops]
		vals = [min(op)	for	op in ops]

		mindist	= min(vals)
		axes = [i for i	in range(self.dim) if vals[i]==mindist]
		nearest_point =	np.copy(loc)
		for	ax in axes:
			cont = True
			if self.ords[ax] %2 == 0:
				cont = False
				shift = (side_vals[sides[ax]] - loc[ax])/H
				if shift < 0 and abs(shift) < 1+self.supports_L[ax]:
					cont = True
				if shift > 0 and shift < 1+self.supports_R[ax]:
					cont = True
			if cont:
				nearest_point[ax] =	side_vals[sides[ax]]
		return nearest_point

	def center_checks(self,H,edges,center,far_out):
		mini = lambda x,d: edges[d][0]<= x <= edges[d][-1]
		check = lambda loc: self._all_d(mini,loc)
		emini =	lambda x: .25-H	< x	< .75
		echeck = lambda loc: self._all_d(emini,loc)
		quad =	lambda loc:	self._all_d(center,loc)

		### let's change things so that low support gives us the corners
		corner = lambda x,d: x <= edges[d][0] or x >= edges[d][-1]
		low_support = lambda loc: self._all_d(corner,loc)
		# if self.node:
		# 	low_support = lambda loc: False
		# else:
		# 	low_support = lambda loc: self._all_d(far_out,loc)

		return check, echeck, quad, low_support

	def outside_checks(self,H,edges,domain,loose_center,far_in):
		minicheck = lambda x,d: edges[d][1] >= x or edges[d][2]<=x
		check = lambda loc: self._at_least_one(minicheck,loc)

		eoutside = lambda x: -H < x < 1
		einside = lambda x: .25 <= x <= .75-H
		echeck = lambda loc: self._not_all_d(einside,loc) and self._all_d(eoutside,loc)
		quad =	lambda loc:	self._all_d(domain,loc) and self._not_all_d(loose_center,loc)

		low_support = lambda loc: False
		# if self.node:
		# 	low_support = lambda loc: False
		# else:
		# 	low_support = lambda loc: self._all_d(far_in,loc)

		return check, echeck, quad, low_support

	def _setup_coarse_info(self):
		H = self.h
		doms,edge_dicts,funcs = super()._setup_info()
		edges, i_edges = edge_dicts
		center,loose_center, domain, far_in, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]

		if self.rtype == 1: # coarse center
			check, echeck, quad, low_support = self.center_checks(
						H,edges,center,far_out)
		else: # coarse edges
			check, echeck, quad, low_support = self.outside_checks(
						H,edges,domain,loose_center,far_in)

		check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
		check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		interface	= lambda loc: check1(loc) or check2(loc)

		ghost = lambda loc: False

		checks = [check, echeck, periodic, dirichlet,
				  low_support, quad, interface, ghost]
		return doms, checks


	def _setup_fine_info(self):
		H = self.h/2
		doms,edge_dicts,funcs = super()._setup_info(coarse=False)
		edges, i_edges = edge_dicts
		center,loose_center, domain, far_in, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]

		if self.rtype == 0: # fine center
			check, echeck, quad, low_support = self.center_checks(
						H,edges,center,far_out)
			# ghost_x_a =	lambda x,d: i_edges[d][2]+H*self.shifts_L[d] <= x <= .75
			# ghost_x_b =	lambda x,d: .25 <= x <= i_edges[d][1]-H*self.shifts_R[d]
			# ghost_x = lambda x,d: ghost_x_a(x,d) or ghost_x_b(x,d)
			in_i_edge_a = lambda x,d: i_edges[d][0]<=x<=i_edges[d][1]
			in_i_edge_b = lambda x,d: i_edges[d][2]<=x<=i_edges[d][3]
			in_i_edge = lambda x,d: in_i_edge_a(x,d) or in_i_edge_b(x,d)
			ghost	= lambda loc: self._all_d(center,loc) and self._at_least_one(in_i_edge,loc)
		else: # fine edges
			check, echeck, quad, low_support = self.outside_checks(
						H,edges,domain,loose_center,far_in)
			ghost_x =	lambda x,d: i_edges[d][1] <= x <= i_edges[d][2]
			ghost	= lambda loc: self._all_d(ghost_x,loc)

		check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
		check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		interface	= lambda loc: check1(loc) or check2(loc)

		checks = [check, echeck, periodic, dirichlet,
				  low_support, quad, interface, ghost]
		return doms, checks