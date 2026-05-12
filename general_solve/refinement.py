import numpy as	np
from general_solve import globals

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
	def	__init__(self,name,dofloc,N,dim,ords,zigzag=False):
		self.name =	name
		self.N = N
		self.h = 1/N
		self.dim = dim
		self.cell =	dofloc=='cell'
		self.node =	dofloc=='node'
		self.xside = dofloc=='xside'
		self.yside = dofloc=='yside'
		self.ords =	ords
		self.zigzag = zigzag
		 
		self.shifts_R =	[int((ord-1)/2)	for	ord	in self.ords]
		self.shifts_L =	[int(ord/2)	for	ord	in self.ords]
		self.shifts_T =	[max(ord-1,0) for ord in self.ords]

		self.supports_R = [int(ord/2) for ord in ords]
		self.supports_L = [int((ord-1)/2) for ord in ords]

		self.d_data = {i:[[],[]] for i in range(2)} # ind list and loc list
		self.e_data = {i:[[],[],[]] for i in range(2)} # ind list and loc list and quads
		self.b_data = {i:[[],[]] for i in range(2)} # periodic and dirichlet
		self.i_data = {i:[[],[],[]] for i in range(2)}
		self.g_data = {i:[[],[]] for i in range(2)}

	def get_interface_lines(self):
		return [],[]

	def get_patch_id(self,loc):
		return 0

	def	_get_el_quads(self,start,H,check):
		quads =	[]

		if self.dim	== 2:
			shifts = [[1/4,1/4],[3/4,1/4],[1/4,3/4],[3/4,3/4]]			
			# shifts = [[0,0],[1,0],[0,1],[1,1]] for some reason this does better sometimes
		if self.dim	== 3:
			shifts = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]			

		bumps = [1/4*(ord==0) for ord in self.ords]
		
		for	shift in shifts:
			corner = [orig_+H*(shft-bump) for (orig_,shft,bump) in zip(start,shift,bumps)]
			quads.append(check(corner))

		return quads

	def	_closest_point(self,loc,H=None):
		# must be overwritten
		return None

	def _all_d(self,func,loc):
		try:
			return sum([func(x)	for	x in loc]) == self.dim
		except:
			return sum([func(x,d) for d,x in enumerate(loc)]) == self.dim

	def _all_d_not(self,func,loc):
		try:
			return sum([func(x)	for	x in loc]) == 0
		except:
			return sum([func(x,d) for d,x in enumerate(loc)]) == 0

	
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
		if start !=0 and self.ords[0] == 0: start += H
		end	= 1 if (self.node or self.xside) else 1+H/2
		if end != 1 and self.ords[0] == 0: end -= H
		buff = self.cell+self.yside if self.ords[0]!=0 else -1
		xdom = np.linspace(start-H*self.shifts_L[0],
						  end+H*self.shifts_R[0],
						  myN+1+self.shifts_T[0]+buff)
		
		start =	0 if (self.node or self.yside) else 0-H/2
		if start !=0 and self.ords[1] == 0: start += H
		end	= 1 if (self.node or self.yside) else 1+H/2
		if end != 1 and self.ords[1] == 0: end -= H
		buff = self.cell+self.xside if self.ords[1]!=0 else -1
		ydom = np.linspace(start-H*self.shifts_L[1],
						  end+H*self.shifts_R[1],
						  myN+1+self.shifts_T[1]+buff)

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

			if self.ords[i]==0: offset = 0

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
			zsame = (i==1 and self.yside) or (i==0 and self.xside)
			if zsame and self.zigzag:
				shfts = [-H,H,-H,H]
				i_edges[i] = [val+shft for (val,shft) in zip(edges[i],shfts)]
			else:
				i_edges[i] = [val for val in edges[i]]
			if self.cell or same:
				# i_edges[i] = [val for val in edges[i]]
				extraL.append(0)
				extraR.append(0)
			else:
				if globals.LAG:
					i_edges[i] = [.25,.25,.75,.75]
				extraL.append(H*self.shifts_L[i])
				extraR.append(H*self.shifts_R[i])
			
		tmp_edges = i_edges if globals.LAG else edges
		far_in = lambda x,d: tmp_edges[d][1] <= x <= tmp_edges[d][2]
		far_out = lambda x,d: tmp_edges[d][0] >= x or x >= tmp_edges[d][3]

		block =	lambda x,d: tmp_edges[d][0]-extraL[d] <=	x <= tmp_edges[d][-1]+extraR[d]
		slice =	lambda x,d: (tmp_edges[d][0]<=x<=tmp_edges[d][1]) or (tmp_edges[d][2]<=x<=tmp_edges[d][-1])

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
						if interface([x,y]):
							self.i_data[L][0].append([i,j])
							if ghost([x,y]):
								if globals.LAG:
									nearest_point =	self._closest_point([x,y],H)
								else:
									nearest_point = True
							else:
								nearest_point =	None
							self.i_data[L][1].append(nearest_point)
					if echeck([x,y]):
						self.e_data[L][0].append([i,j])
						self.e_data[L][1].append([x,y])
						self.e_data[L][2].append(self._get_el_quads([x,y],H,quad))

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
	def	__init__(self,name,dofloc,N,dim,ords,zigzag=False):
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
	def	__init__(self,name,dofloc,N,dim,ords,zigzag=False):
		super().__init__(name,dofloc,N,dim,ords,zigzag)
		self.rtype = stripe_refinement_type[name]
		self.rdim = int(self.rtype/2) # vertical or horizontal stripe

		rindex_to_shade = {0:['vstripe','vedge'],#vfine
						   1:['vedge','vstripe'],#vco
						   2:['hstripe','hedge'],#hfine
						   3:['hedge','hstripe']}
		self.rshade = rindex_to_shade[self.rtype]

		
	def get_interface_lines(self):
		dofs = np.linspace(0,1,4*self.N+1)
		if self.rdim == 0:
			line0 = [(.25,y) for y in dofs]
			line1 = [(.75,y) for y in dofs]
			comps = [1,1]
		else:
			line0 = [(x,.25) for x in dofs]
			line1 = [(x,.75) for x in dofs]
			comps = [0,0]
		return [np.asarray(line0),np.asarray(line1)],comps


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
		mybnd = .25 if self.ords[rdim]==0 else .25-H
		emini =	lambda x: mybnd	< x	< .75
		emini_nonr = lambda x: -H < x < 1
		echeck = lambda loc: emini(loc[rdim]) and emini_nonr(loc[1-rdim])
		quad =	lambda loc:	domain(loc[1-rdim]) and center(loc[rdim])

		return check, echeck, quad

	def edge_checks(self,H,edges,domain,loose_center):
		rdim = int(self.rtype/2) # vertical or horizontal stripe
		check = lambda loc: edges[rdim][1] >= loc[rdim] or edges[rdim][2]<=loc[rdim]
				
		mybnd = .75 if self.ords[rdim]==0 else .75-H
		emini =	lambda x: -H < x < .25 or mybnd < x < 1
		emini_nonr = lambda x: -H < x < 1
		echeck = lambda loc: emini(loc[rdim]) and emini_nonr(loc[1-rdim])
		quad =	lambda loc:	self._all_d(domain,loc) and not loose_center(loc[rdim])

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

		if self.zigzag:
			if (rdim == 0 and self.xside) or (rdim==1 and self.yside):
				interface = lambda loc: loc[rdim] in [.25,.75]#myslice(loc[rdim],rdim)
				


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
			check, orig_echeck, quad = self.edge_checks(H,edges,domain,loose_center)
			myfar_in = lambda x,d: i_edges[d][1] <= x <= i_edges[d][2]
			orig_ghost = lambda loc: myfar_in(loc[rdim],rdim)
			myslice = lambda x,d: (i_edges[d][0]<=x<=edges[d][1]) or (edges[d][2]<=x<=i_edges[d][-1])

			if self.zigzag:
				# sloppy fix
				echeck = lambda loc: orig_echeck(loc) and (loc[rdim] not in [.25-H,.75])
			else:
				echeck = orig_echeck
		else: # fine stripe
			check,orig_echeck,quad = self.stripe_checks(H,edges,domain,center)
			orig_ghost = lambda loc: far_out(loc[rdim],rdim)
			myslice = lambda x,d: (edges[d][0]<=x<=i_edges[d][1]) or (i_edges[d][2]<=x<=edges[d][-1])

			if self.zigzag:
				# sloppy fix
				echeck = lambda loc: orig_echeck(loc) and (loc[rdim] not in [.25,.75-H])
			else:
				echeck = orig_echeck

		interface = lambda loc: slice(loc[rdim],rdim)

		low_support = lambda loc: False

		if self.zigzag:
			if (rdim==0 and self.xside) or (rdim==1 and self.yside):
				ghost = lambda loc: loc[rdim] in [.25,.75]
				interface = lambda loc: myslice(loc[rdim],rdim)
			else:
				ghost = orig_ghost
		else:
			ghost = orig_ghost

		

		if self.ords[rdim] == 0:
			ghost = lambda loc: False#interface(loc)
		# ghost = lambda loc: False

		checks = [check, echeck, periodic, dirichlet,
				  low_support, quad, interface, ghost]
		return doms, checks

class SquareRefinement(RefinementPattern):
	def	__init__(self,name,dofloc,N,dim,ords,zigzag=False):
		super().__init__(name,dofloc,N,dim,ords,zigzag)
		self.rtype = square_refinement_type[name]
		rindex_to_shade ={0:['in','out'],1:['out','in']}
		self.rshade = rindex_to_shade[self.rtype]

	def get_interface_lines(self):
		dofs = np.linspace(.25,.75,2*self.N+1)

		line0 = np.asarray([(.25,y) for y in dofs])
		line1 = np.asarray([(.75,y) for y in dofs])
		line2 = np.asarray([(x,.25) for x in dofs])
		line3 = np.asarray([(x,.75) for x in dofs])
		return [line0,line1,line2,line3],[1,1,0,0]
		
	
	def get_null_condensers(self):
		if self.rtype:
			cut = int(self.N/2)
			I = np.eye(cut)

			coarse_condense = np.zeros((cut,3*cut))
			coarse_condense[:,cut:-cut] = I[:]

			fine_condense = np.zeros((4*cut,3*cut))
			fine_condense[:cut,:cut] = I[:]
			fine_condense[-cut:,-cut:] = I[:]
			fine_condense[cut:-cut:2,cut:-cut] = I[:]
			fine_condense[cut+1:-cut:2,cut:-cut] = I[:]
		else:
			cut = int(self.N/4)
			I = np.eye(2*cut)

			coarse_condense = np.eye(self.N)

			fine_condense = np.zeros((self.N,self.N))
			fine_condense[::2,cut:-cut] = I[:]
			fine_condense[1::2,cut:-cut] = I[:]
		return coarse_condense.T, fine_condense.T, cut

	def get_patch_id(self, loc):
		check = lambda x: .25 <= x < .75
		in_center = self._all_d(check,loc)

		if in_center:
			return 1-self.rtype
		else:
			return self.rtype

	def _closest_point_side_centered(self,loc,H=None):
		if self.xside:
			comp = 1
		if self.yside:
			comp = 0
		side_vals =	[.25,.75]
		ops	= [abs(.25-loc[comp]),abs(.75-loc[comp])]
		side = side_vals[np.argmin(ops)]
		new_loc = [x for x in loc]
		new_loc[comp] = side
		return new_loc

	def _check_ax(self,ax,side_vals,sides,loc,H):
		shift = (side_vals[sides[ax]] - loc[ax])/H
		if shift < 0 and abs(shift) < 1+self.supports_L[ax]:
			return True
		if shift > 0 and shift < 1+self.supports_R[ax]:
			return True
		return False

	def _closest_point(self, loc, H=None):
		if .25 in loc or .75 in loc:
			if self._all_d(lambda x:.25 <= x <= .75,loc):
				return loc

		side_vals =	[.25,.75]
		new_loc = [x for x in loc]
		if self.rtype == 1: # fine edges: find point going out
			if self.xside or self.yside:
				comp = self.xside
				ops = [abs(s-loc[comp]) for s in side_vals]
				new_loc[comp] = side_vals[np.argmin(ops)]
				return new_loc
			else: # cell centered
				ops = [[abs(s-x) for s in side_vals] for x in loc]
				sides = [np.argmin(op) for op in ops]
				side_dists = [min(op) for op in ops]

				min_dist = min(side_dists)
				for cord,side_dist in enumerate(side_dists):
					if side_dist == min_dist:
						cont = self._check_ax(cord,side_vals,sides,loc,H)
						if cont:
							new_loc[cord] = side_vals[sides[cord]]
				return new_loc

		else: #fine center: find point going in
			in_or_out = [.25<x<.75 for x in loc]
			# print(in_or_out)
			if max(in_or_out): #one cord in inside (.25,.75)
				# print(loc)
				min_cord = np.argmin(in_or_out)
				side_dists = [abs(s-loc[min_cord]) for s in side_vals]
				new_loc[min_cord] = side_vals[np.argmin(side_dists)]
				return new_loc
			in_or_out = [.25<=x<=.75 for x in loc]
			if max(in_or_out): #one cord is on line .25 or .75
				shift = {.25:.25+H/2,.75:.75-H/2}
				min_cord = np.argmin(in_or_out)
				side_dists = [abs(s-loc[min_cord]) for s in side_vals]
				if min(side_dists) > H:
					return None
				elif min(side_dists) == H:
					tmp = side_vals[np.argmin(side_dists)]
					new_loc[min_cord] = shift[tmp]
					return new_loc
				new_loc[min_cord] = side_vals[np.argmin(side_dists)]
				# print(new_loc,loc,min_cord,side_dists,H)
				return new_loc
			# if we are here, both cords are outside [.25,.75]
			# print(new_loc,loc,min_cord,side_dists,H)
			shift = {0:.25+H/4,1:.75-H/4}
			ops = [[abs(s-x) for s in side_vals] for x in loc]
			sides = [np.argmin(op) for op in ops]
			side_dists = [min(op) for op in ops]

			max_dist = max(side_dists)
			for cord,side_dist in enumerate(side_dists):
				if side_dist == max_dist:
					new_loc[cord] = side_vals[sides[cord]]
				else:
					return None
			return new_loc


	def center_checks(self,H,edges,center,far_out):
		mini = lambda x,d: edges[d][0]<= x <= edges[d][-1]
		check = lambda loc: self._all_d(mini,loc)
		mybnd = [.25-H*(self.ords[d]!=0) for d in range(self.dim)]
		emini =	lambda x,d: mybnd[d] < x < .75
		echeck = lambda loc: self._all_d(emini,loc)
		quad =	lambda loc:	self._all_d(center,loc)

		### let's change things so that low support gives us the corners
		corner = lambda x,d: x < edges[d][1] or x > edges[d][-2]
		low_support = lambda loc: self._all_d(corner,loc)

		return check, echeck, quad, low_support

	def outside_checks(self,H,edges,domain,loose_center,far_in):
		minicheck = lambda x,d: edges[d][1] >= x or edges[d][2]<=x
		check = lambda loc: self._at_least_one(minicheck,loc)

		eoutside = lambda x: -H < x < 1
		mybnd = [.75-H*(self.ords[d]!=0) for d in range(self.dim)]
		einside = lambda x,d: .25 <= x <= mybnd[d]
		echeck = lambda loc: self._not_all_d(einside,loc) and self._all_d(eoutside,loc)
		quad =	lambda loc:	self._all_d(domain,loc) and self._not_all_d(loose_center,loc)

		low_support = lambda loc: False

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

		if self.zigzag and self.xside:
			check2 = lambda	loc: block(loc[1],1) and loc[0] in [.25,.75]
		else:
			check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		if self.zigzag and self.yside:
			check1 = lambda	loc: block(loc[0],0) and loc[1] in [.25,.75]
		else:
			check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
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
			check, orig_echeck, quad, low_support = self.center_checks(
						H,edges,center,far_out)

			in_i_edge = lambda x,d: x==i_edges[d][0] or x==i_edges[d][-1]
			out_i_edge = lambda x,d: i_edges[d][0] <= x <= i_edges[d][-1]
			orig_ghost	= lambda loc: self._all_d(out_i_edge,loc) and self._at_least_one(in_i_edge,loc)
			myslice = lambda x,d: (edges[d][0]<=x<=i_edges[d][1]) or (i_edges[d][2]<=x<=edges[d][-1])

			if self.zigzag:
				# sloppy fix
				eline_check = lambda x: x not in [.25,.75-H]
				echeck = lambda loc: orig_echeck(loc) and eline_check(loc[0]) and eline_check(loc[1])
			else:
				echeck = orig_echeck

		else: # fine edges
			check, orig_echeck, quad, low_support = self.outside_checks(
						H,edges,domain,loose_center,far_in)
			ghost_x =	lambda x,d: i_edges[d][1] <= x <= i_edges[d][2]
			orig_ghost	= lambda loc: self._all_d(ghost_x,loc)
			myslice = lambda x,d: (i_edges[d][0]<=x<=edges[d][1]) or (edges[d][2]<=x<=i_edges[d][-1])

			if self.zigzag:
				# sloppy fix
				eline_check = lambda x: x not in [.25-H,.75]
				echeck = lambda loc: orig_echeck(loc) and eline_check(loc[0]) and eline_check(loc[1])
			else:
				echeck = orig_echeck

		if self.zigzag and self.yside:
			check1 = lambda	loc: block(loc[0],0) and myslice(loc[1],1)
		else:
			check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
		if self.zigzag and self.xside:
			check2 = lambda	loc: block(loc[1],1) and myslice(loc[0],0)
		else:
			check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		interface	= lambda loc: check1(loc) or check2(loc)

		# sloppy fix
		if self.zigzag:
			zerdim = 0 if self.xside else 1
			ghost = lambda loc: orig_ghost(loc) or loc[zerdim] in [.25,.75]
		else:
			ghost = orig_ghost

		checks = [check, echeck, periodic, dirichlet,
				  low_support, quad, interface, ghost]
		return doms, checks