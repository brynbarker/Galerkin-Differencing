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
		self.e_data = {i:[[],[],[],[]] for i in range(2)} # ind list and loc list and quads
		self.b_data = {i:[[],[]] for i in range(2)} # periodic and dirichlet
		self.i_data = {i:[[],[],[]] for i in range(2)}
		self.g_data = {i:[[],[]] for i in range(2)}

	def get_interface_lines(self):
		return [],[],[]

	def get_patch_id(self,loc):
		return 0

	def	_get_el_quads(self,start,H,check,tri=False):
		quads =	[]

		if self.dim	== 2:
			shifts = [[1/4,1/4],[3/4,1/4],[1/4,3/4],[3/4,3/4]]			
			# shifts = [[0,0],[1,0],[0,1],[1,1]] for some reason this does better sometimes
		if self.dim	== 3:
			shifts = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]			

		bumps = [1/4*(ord==0) for ord in self.ords]
		if tri:
			shifts = [[-1/4,-1/4],[1/4,-1/4],[-1/4,1/4],[1/4,1/4]]
			bumps = [0,0,0,0]
			new_start = [cord for cord in start]
			new_start[self.yside] += H
		
		for	shift in shifts:
			corner = [orig_+H*(shft-bump) for (orig_,shft,bump) in zip(start,shift,bumps)]
			quads.append(check(corner))
			if tri:
				corner = [orig_+H*(shft-bump) for (orig_,shft,bump) in zip(new_start,shift,bumps)]
				quads.append(check(corner))

		if tri:
			return True in quads
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

		if self.zigzag:
			Lshft = .25+self.h/2 if self.rtype%2==0 else .25-self.h/2
			Rshft = .75-self.h/2 if self.rtype%2==0 else .75+self.h/2
			zoffIN = H if self.rtype%2==0 else 0
			zoffOUT = 0 if self.rtype%2==0 else H 
		edges =	{}
		for	i in range(self.dim):
			zsame = self.zigzag and ((i==0 and self.xside) or (i==1 and self.yside))
			if i==0:
				offset = (self.cell+self.yside)*H/2
			if i==1:
				offset = (self.cell+self.xside)*H/2
			Lstart,Rstart = [.25,.75] if not zsame else [Lshft,Rshft]

			if self.ords[i]==0: offset = 0
			offsetIN,offsetOUT = offset,offset
			if zsame: 
				offsetIN,offsetOUT = zoffIN,zoffOUT
			print(zsame)

			edge1a = Lstart - offsetOUT - H*self.shifts_L[i]
			edge1b = Lstart + offsetIN + H*self.shifts_R[i]
			edge2a = Rstart - offsetIN - H*self.shifts_L[i]
			edge2b = Rstart + offsetOUT + H*self.shifts_R[i]

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
				# shfts = [-H,H,-H,H]
				i_edges[i] = [.25-self.h/2,self.h/2+.25,.75-self.h/2,.75+self.h/2]
				# i_edges[i] = [val+shft for (val,shft) in zip(edges[i],shfts)]
				i_edges[i] = [Lshft,Lshft,Rshft,Rshft]
			elif self.zigzag:
				i_edges[i] = [.25,.25,.75,.75]
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
		Lline_support = lambda x,d: x-H-extraR[d] < i_edges[d][0] < x+H+extraL[d]
		Rline_support = lambda x,d: x-H-extraR[d] < i_edges[d][3] < x+H+extraL[d]
		line_support = lambda x,d: Lline_support(x,d) or Rline_support(x,d)

		block =	lambda x,d: tmp_edges[d][0]-extraL[d] <=	x <= tmp_edges[d][-1]+extraR[d]
		slice =	lambda x,d: (tmp_edges[d][0]<=x<=tmp_edges[d][1]) or (tmp_edges[d][2]<=x<=tmp_edges[d][-1])

		funcs = [center,loose_center, domain, line_support, far_out,
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
		check, echeck, periodic, z_interface = checks[:4]
		e_extra, quad, interface, ghost = checks[-4:]

		if self.dim	== 2:
			xdom,ydom = doms
			for	i,y	in enumerate(ydom):
				for	j,x	in enumerate(xdom):
					if check([x,y]):
						self.d_data[L][0].append([i,j])
						self.d_data[L][1].append([x,y])
						self.b_data[L][0].append(periodic([x,y]))
						# self.i_data[L][2].append(low_support([x,y]))
						# r_inter,z_inter = interface([x,y]),z_interface([x,y])
						if interface([x,y]):
						# if r_inter or z_inter:
							# rval = [i,j] if r_inter else None
							# zval = [i,j] if z_inter else None
							self.i_data[L][0].append([i,j])
							# self.i_data[L][2].append(zval)#z_interface([x,y]))
							if ghost([x,y]):
								nearest_point =	self._closest_point([x,y],H)
							else:
								nearest_point =	None
							self.i_data[L][1].append(nearest_point)
					if echeck([x,y]):
						self.e_data[L][0].append([i,j])
						self.e_data[L][1].append([x,y])
						if e_extra([x,y]):
							self.e_data[L][2].append(self._get_el_quads([x,y],H,quad,tri=True))
						else:
							self.e_data[L][2].append(self._get_el_quads([x,y],H,quad))
						self.e_data[L][3].append(e_extra([x,y]))

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
		
		d_info = self.d_data[L] + [self.b_data[L][0]]#,self.i_data[L][2]]
		return d_info,self.e_data[L],self.i_data[L],[len(dom) for dom in doms]

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
		center,loose_center, domain, line_support, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]

		check =	lambda x: True

		emini =	lambda x: -H < x < 1
		echeck = lambda	loc: self._all_d(emini,loc)

		interface	= lambda x:	True
		quad =	lambda loc:	self._all_d(domain,loc)
		# low_support = lambda loc: False
		e_extra = lambda *args: False

		ghost = lambda loc: False

		z_interface = lambda *args: False
		checks = [check, echeck, periodic, z_interface,
				  e_extra, quad, interface, ghost]
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
		sg = 1 if self.rtype%2 == 0 else -1
		xsh = self.xside*self.h/2*sg
		ysh = self.yside*self.h/2*sg
		lines,comps,dirs = [],[],[]
		if self.rdim == 0:
			lines.append(np.asarray([(.25+xsh,y) for y in dofs]))
			lines.append(np.asarray([(.75-xsh,y) for y in dofs]))
			comps += [1,1]
			if self.rtype < 2:
				dirs += [1,-1]
			else:
				dirs += [-1,1]
		elif self.rdim == 1:
			lines.append(np.asarray([(x,.25+ysh) for x in dofs]))
			lines.append(np.asarray([(x,.75-ysh) for x in dofs]))
			comps += [0,0]
			if self.rtype < 2:
				dirs += [1,-1]
			else:
				dirs += [-1,1]
		return lines, comps, dirs



	def get_patch_id(self, loc):
		in_stripe = .25 <= loc[self.rdim] < .75

		stripe_id = 1 - (self.rtype%2)

		if in_stripe:
			return stripe_id
		else:
			return 1-stripe_id


	def _closest_point(self, loc, H=None):
		if not globals.LAG: return True
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
		center,loose_center, domain, line_support, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]

		rdim = int(self.rtype/2) # vertical or horizontal stripe
		if self.rtype % 2: # coarse stripe
			check, old_echeck, quad = self.stripe_checks(H,edges,domain,center)
			if self.zigzag:
				mybnd = .25 if self.ords[rdim]==0 else .25-H
				emini =	lambda x: mybnd-H/2	< x	< .75+H/2
				emini_nonr = lambda x: -H < x < 1
				echeck = lambda loc: emini(loc[rdim]) and emini_nonr(loc[1-rdim])
				quad =	lambda loc:	domain(loc[1-rdim]) and center(loc[rdim])
				# sloppy fix
				not_e_extra = lambda loc: echeck(loc) and (loc[rdim] not in [.25-H,.75])
				e_extra = lambda loc: not not_e_extra(loc)
		else: # coarse edges
			check, old_echeck, quad = self.edge_checks(H,edges,domain,loose_center)
			if self.zigzag:
				# sloppy fix
				mybnd = .75 if self.ords[rdim]==0 else .75-H
				emini =	lambda x: -H < x < .25+H/2 or mybnd-H/2 < x < 1
				emini_nonr = lambda x: -H < x < 1
				echeck = lambda loc: emini(loc[rdim]) and emini_nonr(loc[1-rdim])
				not_e_extra = lambda loc: echeck(loc) and (loc[rdim] not in [.25,.75-H])
				e_extra = lambda loc: not not_e_extra(loc)

		if not globals.LAG:
			interface = lambda loc: line_support(loc[rdim],rdim)
		else:
			interface = lambda loc: slice(loc[rdim],rdim)

		# low_support = lambda loc: False
		# e_extra = lambda loc: False
		ghost = lambda loc: False

		z_interface = lambda *args: False
		if self.zigzag:
			if (rdim == 0 and self.xside) or (rdim==1 and self.yside):
				# interface = lambda loc: loc[rdim] in [.25,.75]#myslice(loc[rdim],rdim)
				z_interface = lambda loc: loc[rdim] in [.25,.75]#myslice(loc[rdim],rdim)
				


		checks = [check, echeck, periodic, z_interface,
				  e_extra, quad, interface, ghost]
		return doms, checks

	def _setup_fine_info(self):
		H = self.h/2
		doms,edge_dicts,funcs = super()._setup_info(coarse=False)
		edges, i_edges = edge_dicts
		center,loose_center, domain, line_support, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]
		print(edges)
		print(i_edges)


		z_interface = lambda *args: False
		e_extra = lambda *args: False
		rdim = int(self.rtype/2) # vertical or horizontal stripe
		# here is what far in is doing
		# let's look at the values for edges[rdim][1,2]
		#far_in = lambda x,d: edges[d][1] <= x <= edges[d][2]
		if self.rtype % 2: # fine edges
			check, old_echeck, quad = self.edge_checks(H,edges,domain,loose_center)
			myfar_in = lambda x,d: i_edges[d][1] <= x <= i_edges[d][2]
			orig_ghost = lambda loc: myfar_in(loc[rdim],rdim)
			myslice = lambda x,d: (i_edges[d][0]<=x<=edges[d][1]) or (edges[d][2]<=x<=i_edges[d][-1])

			if self.zigzag:
				# sloppy fix
				not_e_extra = lambda loc: old_echeck(loc) and (loc[rdim] not in [.25-H,.75])
				# tmp_e_extra = lambda loc: not not_e_extra(loc)

				# e_extra_check = lambda loc: loc[rdim] in [.25-2*H,.75+H]
				# e_extra = lambda loc: echeck(loc) and e_extra_check(loc)
			# else:
			# 	echeck = orig_echeck
		else: # fine stripe
			check,old_echeck,quad = self.stripe_checks(H,edges,domain,center)
			orig_ghost = lambda loc: far_out(loc[rdim],rdim)
			myslice = lambda x,d: (edges[d][0]<=x<=i_edges[d][1]) or (i_edges[d][2]<=x<=edges[d][-1])

			if self.zigzag:
				# sloppy fix
				not_e_extra = lambda loc: old_echeck(loc) and (loc[rdim] not in [.25,.75-H])
				# tmp_e_extra = lambda loc: not not_e_extra(loc)

				# e_extra_check = lambda loc: loc[rdim] in [.25+H,.75-2*H]
				# e_extra = lambda loc: echeck(loc) and e_extra_check(loc)
			# else:
			# 	echeck = orig_echeck

		if not globals.LAG:
			interface = lambda loc: line_support(loc[rdim],rdim)
		else:
			interface = lambda loc: slice(loc[rdim],rdim)

		if self.zigzag:
			echeck = lambda loc: not_e_extra(loc)
			if (rdim==0 and self.xside) or (rdim==1 and self.yside):
				ghost = lambda loc: loc[rdim] in [.25,.75]
				z_interface = lambda loc: myslice(loc[rdim],rdim)

				# z_interface = lambda loc: loc[rdim] in [.25,.75]#ghost
			else:
				ghost = orig_ghost
		else:
			echeck = old_echeck
			ghost = orig_ghost
			# if not globals.LAG and self.cell:
			# 	ghost = lambda *args: True

		e_extra = lambda loc: False
		

		if self.ords[rdim] == 0:
			ghost = lambda loc: False#interface(loc)
		# ghost = lambda loc: False
		 
		checks = [check, echeck, periodic, z_interface,
				  e_extra, quad, interface, ghost]
		return doms, checks

class SquareRefinement(RefinementPattern):
	def	__init__(self,name,dofloc,N,dim,ords,zigzag=False):
		super().__init__(name,dofloc,N,dim,ords,zigzag)
		self.rtype = square_refinement_type[name]
		rindex_to_shade ={0:['in','out'],1:['out','in']}
		self.rshade = rindex_to_shade[self.rtype]

	def get_interface_lines(self):
		dofs = np.linspace(.25,.75,2*self.N+1)

		sg = 1 if self.rtype == 0 else -1
		xsh = self.xside*self.h/2*sg
		ysh = self.yside*self.h/2*sg
		line0 = np.asarray([(.25+xsh,y) for y in dofs])
		line1 = np.asarray([(.75-xsh,y) for y in dofs])
		line2 = np.asarray([(x,.25+ysh) for x in dofs])
		line3 = np.asarray([(x,.75-ysh) for x in dofs])

		lines = [line0,line1,line2,line3]
		comps = [1,1,0,0]

		if self.rtype == 0:
			dirs = [1,-1,1,-1]
		else:
			dirs = [-1,1,-1,1]

		return lines, comps, dirs
		
	
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
		if not globals.LAG: return True
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
		# low_support = lambda loc: self._all_d(corner,loc)

		return check, echeck, quad#, low_support

	def outside_checks(self,H,edges,domain,loose_center):
		minicheck = lambda x,d: edges[d][1] >= x or edges[d][2]<=x
		check = lambda loc: self._at_least_one(minicheck,loc)

		eoutside = lambda x: -H < x < 1
		mybnd = [.75-H*(self.ords[d]!=0) for d in range(self.dim)]
		einside = lambda x,d: .25 <= x <= mybnd[d]
		echeck = lambda loc: self._not_all_d(einside,loc) and self._all_d(eoutside,loc)
		quad =	lambda loc:	self._all_d(domain,loc) and self._not_all_d(loose_center,loc)

		# low_support = lambda loc: False

		return check, echeck, quad#, low_support

	def _setup_coarse_info(self):
		H = self.h
		doms,edge_dicts,funcs = super()._setup_info()
		edges, i_edges = edge_dicts
		center,loose_center, domain, line_support, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]

		if self.rtype == 1: # coarse center
			check, echeck, quad = self.center_checks(
						H,edges,center,far_out)
		else: # coarse edges
			check, echeck, quad = self.outside_checks(
						H,edges,domain,loose_center)

		check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
		check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		if not globals.LAG:
			interface = lambda loc: self._at_least_one(line_support,loc)
		else:
			interface	= lambda loc: check1(loc) or check2(loc)

		if self.zigzag:
			if self.xside:
				zcheck2 = lambda	loc: block(loc[1],1) and loc[0] in [.25,.75]
			else:
				zcheck2 = check2
			if self.yside:
				zcheck1 = lambda	loc: block(loc[0],0) and loc[1] in [.25,.75]
			else:
				zcheck1 = check1
			z_interface	= lambda loc: check1(loc) or check2(loc)
		else:
			z_interface = lambda *args: False

		ghost = lambda loc: False

		e_extra = lambda *args: False
		checks = [check, echeck, periodic, z_interface,
				  e_extra, quad, interface, ghost]
		return doms, checks


	def _setup_fine_info(self):
		H = self.h/2
		doms,edge_dicts,funcs = super()._setup_info(coarse=False)
		edges, i_edges = edge_dicts
		center,loose_center, domain, line_support, far_out = funcs[:5]
		periodic, dirichlet, block, slice = funcs[-4:]

		z_interface = lambda *args: False
		e_extra = lambda *args: False
		if self.rtype == 0: # fine center
			check, echeck, quad = self.center_checks(
						H,edges,center,far_out)

			in_i_edge = lambda x,d: x==i_edges[d][0] or x==i_edges[d][-1]
			out_i_edge = lambda x,d: i_edges[d][0] <= x <= i_edges[d][-1]
			orig_ghost	= lambda loc: self._all_d(out_i_edge,loc) and self._at_least_one(in_i_edge,loc)
			myslice = lambda x,d: (edges[d][0]<=x<=i_edges[d][1]) or (i_edges[d][2]<=x<=edges[d][-1])

			if self.zigzag:
				# sloppy fix
				eline_check = lambda x: x not in [.25,.75-H]
				not_e_extra = lambda loc: echeck(loc) and eline_check(loc[0]) and eline_check(loc[1])
				e_extra = lambda loc: not not_e_extra(loc)

				# e_extra_line = lambda x: x in [.25+H,.75-2*H]
				# e_extra_check = lambda loc: e_extra_line(loc[0]) or e_extra_line(loc[1])

				# e_extra = lambda loc: echeck(loc) and e_extra_check(loc)
			# else:
			# 	echeck = orig_echeck

		else: # fine edges
			check, echeck, quad = self.outside_checks(
						H,edges,domain,loose_center)
			ghost_x =	lambda x,d: i_edges[d][1] <= x <= i_edges[d][2]
			orig_ghost	= lambda loc: self._all_d(ghost_x,loc)
			myslice = lambda x,d: (i_edges[d][0]<=x<=edges[d][1]) or (edges[d][2]<=x<=i_edges[d][-1])

			if self.zigzag:
				# sloppy fix
				eline_check = lambda x: x not in [.25-H,.75]
				not_e_extra = lambda loc: echeck(loc) and eline_check(loc[0]) and eline_check(loc[1])
				e_extra = lambda loc: not not_e_extra(loc)

				# e_extra0 = lambda x: x in [.25-2*H,.75+H]
				# e_extra1 = lambda x: .25-2*H <= x <= .75+H
				# e_extra_check0 = lambda loc: e_extra0(loc[0]) and e_extra1(loc[1])
				# e_extra_check1 = lambda loc: e_extra0(loc[1]) and e_extra1(loc[0])
				# e_extra_check = lambda loc: e_extra_check0(loc) or e_extra_check1(loc)

				# e_extra = lambda loc: echeck(loc) and e_extra_check(loc)
			# else:
			# 	echeck = orig_echeck

		check1 = lambda	loc: block(loc[0],0) and slice(loc[1],1)
		check2 = lambda	loc: block(loc[1],1) and slice(loc[0],0)
		if not globals.LAG:
			interface = lambda loc: self._at_least_one(line_support,loc)
		else:
			interface	= lambda loc: check1(loc) or check2(loc)
		if self.zigzag:
			if self.yside:
				zcheck1 = lambda	loc: block(loc[0],0) and myslice(loc[1],1)
			else:
				zcheck1 = check1
			if self.xside:
				zcheck2 = lambda	loc: block(loc[1],1) and myslice(loc[0],0)
			else:
				zcheck2 = check2
		z_interface	= lambda loc: zcheck1(loc) or zcheck2(loc)

		# sloppy fix
		if self.zigzag:
			zerdim = 0 if self.xside else 1
			ghost = lambda loc: orig_ghost(loc) or loc[zerdim] in [.25,.75]
			# z_interface = lambda loc: loc[zerdim] in [.25,.75]
		else:
			ghost = orig_ghost
			# if not globals.LAG and self.cell:
			# 	ghost = lambda *args: True


		checks = [check, echeck, periodic, z_interface,
				  e_extra, quad, interface, ghost]
		return doms, checks