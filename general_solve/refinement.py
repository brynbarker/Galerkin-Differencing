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
		
		far_in = lambda x,d: edges[d][1] <= x <= edges[d][2]
		far_out = lambda x,d: edges[d][0] <= x or x >= edges[d][3]

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
			
		block =	lambda x,d: i_edges[d][0]-extraL[d] <=	x <= i_edges[d][-1]+extraR[d]
		slice =	lambda x,d: (i_edges[d][0]<=x<=i_edges[d][1]) or (i_edges[d][2]<=x<=i_edges[d][-1])

		funcs = [center,loose_center, domain, far_in, far_out,
		   		 periodic_check_full, dirichlet_check_full, block, slice]
		return doms, [edges,i_edges], funcs

		check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
		check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		interface_check	= lambda loc: check1(loc) or check2(loc)
		ghost_check	= lambda loc: False

		if self.rtype == 0:	# uniform
			check =	lambda x: True
			emini =	lambda x: -H < x < 1
			echeck = lambda	loc: sum([emini(x) for x in	loc]) == self.dim 
			interface_check	= lambda x:	True
			quadcheck =	lambda loc:	domain_full(loc)
			low_support_square = lambda loc: False

		elif self.rtype	== 1: #	fine center
			# mini = lambda x: (x	<= edge1b) or (x >=	edge2a)
			mini = lambda x,d: (x	<= edges[d][1]) or (x >= edges[d][2])
			check =	lambda loc:	sum([mini(x,i) for i,x in enumerate(loc)])	>= 1

			eout = lambda x: -H	< x	< 1
			emini =	lambda x: .25 <= x <= .75-H
			notcheck = lambda loc: sum([emini(x) for x in loc])	< self.dim
			outcheck = lambda loc: sum([eout(x)	for	x in loc]) == self.dim
			echeck = lambda	loc: notcheck(loc) and outcheck(loc)

			quadcheck =	lambda loc:	not_center_full(loc)
			low_support_square = lambda loc: sum([far_in_1d(x,d) for (d,x) in enumerate(loc)])==self.dim

		elif self.rtype	== 2: #	coarse center
			# mini = lambda x: (x	>= edge1a) and (x <= edge2b)
			mini = lambda x,d: (x	>= edges[d][0]) and (x <= edges[d][-1])
			check =	lambda loc:	sum([mini(x,i) for i,x in enumerate(loc)])	== self.dim

			emini =	lambda x: .25-H	< x	< .75
			echeck = lambda	loc: sum([emini(x) for x in	loc]) == self.dim

			quadcheck =	lambda loc:	center_full(loc)
			dirichlet_check	= lambda i,j: False
			low_support_square = lambda loc: sum([far_out_1d(x,d) for (d,x) in enumerate(loc)])==self.dim

		else:
			raise ValueError('refinement type not supported')

	def _setup_coarse_info(self):
		tmp = self._setup_info()
		### must be overwritten
	
	def _setup_fine_info(self):
		tmp = self._setup_info(coarse=False)
		### must be overwritten
  
	def get_info(self,coarse=True):
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
		return self.get_info(coarse=True)
	def	get_fine_info(self):
		return self.get_info(coarse=False)
		#H =	self.h/2
		#start =	0 if self.node else	-H/2
		#end	= 1	if self.node else 1+H/2

		#xdom = np.linspace(start-H*self.shifts_L[0],
		#				  end+H*self.shifts_R[0],
		#				  2*self.N+1+self.shifts_T[0]+self.cell)
		#ydom = np.linspace(start-H*self.shifts_L[1],
		#				  end+H*self.shifts_R[1],
		#				  2*self.N+1+self.shifts_T[1]+self.cell)
		#if self.dim	== 3:
		#	zdom = np.linspace(start-H*self.shifts_L[2],
		#				  end+H*self.shifts_R[2],
		#				  2*self.N+1+self.shifts_T[2]+self.cell)
		#d_ind_list,d_loc_list =	[],[]
		#e_ind_list,e_loc_list =	[],[]
		#int_ind_list,int_ghost_list	= [],[]
		#e_quads	= []
		#d_periodic = []
		#d_dirichlet	= []
		#d_square = []

		#edges =	{}
		#for	i in range(self.dim):

		#	edge1a = 0.25 -	(self.cell*H/2)	- H*self.shifts_L[i]
		#	edge1b = 0.25 +	(self.cell*H/2)	+ H*self.shifts_R[i]
		#	edge2a = 0.75 -	(self.cell*H/2)	- H*self.shifts_L[i]
		#	edge2b = 0.75 +	(self.cell*H/2)	+ H*self.shifts_R[i]

		#	edges[i] = [edge1a,edge1b,edge2a,edge2b]
		
		#center_1d =	lambda x: .25 <= x <= .75
		#center_full	= lambda loc: sum([center_1d(x)	for	x in loc]) == self.dim
		#loose_center_1d	= lambda x:	.25	< x	< .75
		#loose_center_full =	lambda loc:	sum([loose_center_1d(x)	for	x in loc]) == self.dim
		#domain_1d =	lambda x: 0	<= x <=	1
		#domain_full	= lambda loc: sum([domain_1d(x)	for	x in loc]) == self.dim

		#not_center_full	= lambda loc: domain_full(loc) and not loose_center_full(loc)
		#far_in_1d = lambda x,d: edges[d][1] <= x <= edges[d][2]
		#far_out_1d = lambda x,d: edges[d][0] <= x or x >= edges[d][3]

		#periodic_check = lambda	x: 0 <=	x <	1
		#periodic_check_full	= lambda loc: sum([periodic_check(x) for x in loc])	!= self.dim
		## dirichlet_check	= lambda i,j: i	in [0,len(dom)-1] or j in [0,len(dom)-1]
		#dirichlet_check	= lambda i,j: i	in [0,len(ydom)-1] or j	in [0,len(xdom)-1]

		#i_edges	= {}
		#for	i in range(self.dim):
		#	i_edge1a = .25 if self.node	else .25- H/2 -	H*self.shifts_L[i]
		#	i_edge1b = .25 if self.node	else .25+ H/2 +	H*self.shifts_R[i]
		#	i_edge2a = .75 if self.node	else .75- H/2 -	H*self.shifts_L[i]
		#	i_edge2b = .75 if self.node	else .75+ H/2 +	H*self.shifts_R[i]
		#	
		#	i_edges[i] = [i_edge1a,i_edge1b,i_edge2a,i_edge2b]
		#	
		## block =	lambda x: i_edge1a<= x <= i_edge2b
		#block =	lambda x,d: i_edges[d][0] <= x <= i_edges[d][-1]
		## slice =	lambda x: (i_edge1a<=x<=i_edge1b) or (i_edge2a<=x<=i_edge2b)
		#slice =	lambda x,d: (i_edges[d][0]<=x<=i_edges[d][1]) or (i_edges[d][2]<=x<=i_edges[d][-1])
		#check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
		#check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		#interface_check	= lambda loc: check1(loc) or check2(loc)
		#if self.node:
		#	ghost_check	= lambda loc: True

		#if self.rtype == 0:	# uniform
		#	return [[],[]],[[],[],[]],[[],[]],0
		#	 
		#elif self.rtype	== 1: #	finecenter
		#	# mini = lambda x: (x	>= edge1a) and (x <= edge2b)
		#	mini = lambda x,d: (x >= edges[d][0]) and (x <= edges[d][-1])
		#	check =	lambda loc:	sum([mini(x,i) for i,x in enumerate(loc)]) == self.dim

		#	emini =	lambda x: .25-H	< x	< .75
		#	echeck = lambda	loc: sum([emini(x) for x in	loc]) == self.dim

		#	quadcheck =	lambda loc:	center_full(loc)
		#	dirichlet_check	= lambda i,j: False
		#	if not self.node:
		#		# ghost_check_x =	lambda x: i_edge2b <= x	or x <=	edge1a
		#		# ghost_check_x =	lambda x,d: i_edges[d][-1] <= x	or x <=	edges[d][0]
		#		ghost_check_x =	lambda x,d: i_edges[d][2]+H*self.shifts_L[d] <= x < .75 or .25 < x <= i_edges[d][1]-H*self.shifts_R[d]
		#		
		#		ghost_check	= lambda loc: center_full(loc) and sum([ghost_check_x(x,i) for i,x in enumerate(loc)]) >= 1

		#	low_support_square = lambda loc: sum([far_out_1d(x,d) for (d,x) in enumerate(loc)])==self.dim

		#elif self.rtype	== 2: #	coarse center
		#	# mini = lambda x: (x	<= edge1b) or (x >=	edge2a)
		#	mini = lambda x,d: (x <= edges[d][1]) or (x >=	edges[d][2])
		#	check =	lambda loc:	sum([mini(x,i) for i,x in enumerate(loc)])	>= 1

		#	eout = lambda x: -H	< x	< 1
		#	emini =	lambda x: .25 <= x <= .75-H
		#	notcheck = lambda loc: sum([emini(x) for x in loc])	< self.dim
		#	outcheck = lambda loc: sum([eout(x)	for	x in loc]) == self.dim
		#	echeck = lambda	loc: notcheck(loc) and outcheck(loc)

		#	quadcheck =	lambda loc:	not_center_full(loc)
		#	if not self.node:
		#		# ghost_check_x =	lambda x: i_edge1b <= x	<= edge2a
		#		ghost_check_x =	lambda x,d: i_edges[d][1] <= x <= edges[d][2]
		#		ghost_check	= lambda loc: sum([ghost_check_x(x,i) for i,x in enumerate(loc)]) == self.dim
		#		
		#	low_support_square = lambda loc: sum([far_out_1d(x,d) for (d,x) in enumerate(loc)])==self.dim

		#else:
		#	raise ValueError('refinement type not supported')

		doms, checks = self._setup_coarse_info()
		check, echeck, periodic, dirichlet = checks[:4]
		low_support, quad, interface, ghost = checks[-4:]

		if self.dim	== 2:
			xdom,ydom = doms
			for	i,y	in enumerate(ydom):
				for	j,x	in enumerate(xdom):
					if check([x,y]):
						self.d_data[0].append([i,j])
						self.d_data[1].append([x,y])
						self.b_data[0].append(periodic([x,y]))
						self.b_data[1].append(dirichlet(i,j))
						self.i_data[2].append(low_support([x,y]))
					if echeck([x,y]):
						self.e_data[0].append([i,j])
						self.e_data[1].append([x,y])
						self.e_data[2].append(self._get_el_quads([x,y],H,quad))
					if interface([x,y]):
						self.i_data[0].append([i,j])
						if ghost([x,y]):
							nearest_point =	self._closest_point([x,y])
						else:
							nearest_point =	None
						self.i_data[1].append(nearest_point)

		d_info = self.d_data + self.b_data + [self.i_data[2]]
		return d_info,self.e_data,self.i_data[:2],[len(dom) for dom in doms]
		if self.dim	== 2:
			for	i,y	in enumerate(ydom):
				for	j,x	in enumerate(xdom):
					if check([x,y]):
						d_ind_list.append([i,j])
						d_loc_list.append([x,y])
						d_periodic.append(periodic_check_full([x,y]))
						d_dirichlet.append(dirichlet_check(i,j))
						d_square.append(low_support_square([x,y]))
					if echeck([x,y]):
						e_ind_list.append([i,j])
						e_loc_list.append([x,y])
						e_quads.append(self._get_el_quads([x,y],H,quadcheck))
					if interface_check([x,y]):
						int_ind_list.append([i,j])
						if ghost_check([x,y]):
							nearest_point =	self._closest_point([x,y],H)
						else:
							nearest_point =	None
						int_ghost_list.append(nearest_point)#ghost_check([x,y]))
		if self.dim	== 3:
			for	k,z	in enumerate(zdom):
				for	i,y	in enumerate(ydom):
					for	j,x	in enumerate(xdom):
						if check([x,y,z]):
							d_ind_list.append([i,j,k])
							d_loc_list.append([x,y,z])
							d_periodic.append(periodic_check_full([x,y,z]))
							d_dirichlet.append([dirichlet_check(i,j,k)])
							d_square.append(low_support_square([x,y,z]))
						if echeck([x,y,z]):
							e_ind_list.append([i,j,k])
							e_loc_list.append([x,y,z])
							e_quads.append(self._get_el_quads([x,y,z],H,quadcheck))

		d_info = [d_ind_list,d_loc_list,d_periodic,d_dirichlet,d_square]
		e_info = [e_ind_list,e_loc_list,e_quads]
		i_info = [int_ind_list,int_ghost_list]
		return d_info,e_info,i_info,[len(xdom),len(ydom)]
	

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
		quad =	lambda loc:	domain(loc[1-rdim]) and not loose_center(loc[rdim])

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
		if self.rtype % 2: # fine edges
			check, echeck, quad = self.edge_checks(H,edges,domain,loose_center)
			ghost = lambda loc: far_in(loc[rdim],rdim)
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
		rshade = rindex_to_shade[self.rtype]

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
		#	mini = lambda x,d: (x	>= edges[d][0]) and (x <= edges[d][-1])
		#	check =	lambda loc:	sum([mini(x,i) for i,x in enumerate(loc)])	== self.dim

		#	emini =	lambda x: .25-H	< x	< .75
		#	echeck = lambda	loc: sum([emini(x) for x in	loc]) == self.dim

		#	quadcheck =	lambda loc:	center_full(loc)
		#	dirichlet_check	= lambda i,j: False
		#	low_support_square = lambda loc: sum([far_out_1d(x,d) for (d,x) in enumerate(loc)])==self.dim
		mini = lambda x,d: edges[d][0]<= x <= edges[d][-1]
		check = lambda loc: self._all_d(mini,loc)
		emini =	lambda x: .25-H	< x	< .75
		echeck = lambda loc: self._all_d(emini,loc)
		quad =	lambda loc:	self._all_d(center,loc)

		if self.node:
			low_support = lambda loc: False
		else:
			low_support = lambda loc: self._all_d(far_out,loc)

		return check, echeck, quad, low_support

	def outside_checks(self,H,edges,domain,loose_center,far_in):
		#	mini = lambda x,d: (x	<= edges[d][1]) or (x >= edges[d][2])
		#	check =	lambda loc:	sum([mini(x,i) for i,x in enumerate(loc)])	>= 1

		#	eout = lambda x: -H	< x	< 1
		#	emini =	lambda x: .25 <= x <= .75-H
		#	notcheck = lambda loc: sum([emini(x) for x in loc])	< self.dim
		#	outcheck = lambda loc: sum([eout(x)	for	x in loc]) == self.dim
		#	echeck = lambda	loc: notcheck(loc) and outcheck(loc)

		#	quadcheck =	lambda loc:	not_center_full(loc)
		#	low_support_square = lambda loc: sum([far_in_1d(x,d) for (d,x) in enumerate(loc)])==self.dim
		minicheck = lambda x,d: edges[d][1] >= x or edges[d][2]<=x
		check = lambda loc: self._at_least_one(minicheck,loc)

		eoutside = lambda x: -H < x < 1
		einside = lambda x: .25 <= x <= .75-H
		echeck = lambda loc: self._not_all_d(einside,loc) and self._all_d(eoutside,loc)
		quad =	lambda loc:	self._all_d(domain,loc) and self._not_all_d(loose_center,loc)

		if self.node:
			low_support = lambda loc: False
		else:
			low_support = lambda loc: self._all_d(far_in,loc)

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
			ghost_x_a =	lambda x,d: i_edges[d][2]+H*self.shifts_L[d] <= x < .75
			ghost_x_b =	lambda x,d: .25 < x <= i_edges[d][1]-H*self.shifts_R[d]
			ghost_x = lambda x,d: ghost_x_a(x,d) or ghost_x_b(x,d)
			ghost	= lambda loc: self._all_d(center,loc) and self._at_least_one(ghost_x,loc)
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