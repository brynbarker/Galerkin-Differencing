import numpy as np

refinement_type = {'uniform':0,
				   'finecenter':1,
				   'coarsecenter':2}

class RefinementPattern:
	def __init__(self,name,dofloc,N,dim):
		self.name = name
		self.N = N
		self.h = 1/N
		self.dim = dim
		self.rtype = refinement_type[name]
		self.cell = dofloc=='cell'
		self.node = dofloc=='node'

	def _get_el_quads(self,start,H,check):
		quads = []

		if self.dim == 2:
			shifts = [[0,0],[1,0],[0,1],[1,1]]			
		if self.dim == 3:
			shifts = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]			
		
		for shift in shifts:
			corner = [og+H*shft for (og,shft) in zip(start,shift)]
			quads.append(check(corner))

		return quads

	def _closest_point(self,loc):
		side_vals = [.25,.75]
		ops = [(abs(.25-x),abs(.75-x)) for x in loc]
		sides = [np.argmin(op) for op in ops]
		vals = [min(op) for op in ops]

		mindist = min(vals)
		axes = [i for i in range(self.dim) if vals[i]==mindist]
		nearest_point = loc.copy()
		for ax in axes:
			nearest_point[ax] = side_vals[sides[ax]]
		return nearest_point


	def get_coarse_info(self):
		H = self.h
		start = 0 if self.node else -H/2
		end = 1 if self.node else 1+H/2
		dom = np.linspace(start-H,
				  		  end+H,
						  self.N+3+self.cell)
		d_ind_list,d_loc_list = [],[]
		e_ind_list,e_loc_list = [],[]
		e_quads = []
		d_periodic = []
		d_dirichlet = []
		int_ind_list,int_ghost_list = [],[]

		edge1a = 0.25 - (self.cell*H/2) - H
		edge1b = 0.25 + (self.cell*H/2) + H
		edge2a = 0.75 - (self.cell*H/2) - H
		edge2b = 0.75 + (self.cell*H/2) + H

		# for quad checks
		center_1d = lambda x: .25 <= x <= .75
		loose_center_1d = lambda x: .25 < x < .75
		loose_center_full = lambda loc: sum([loose_center_1d(x) for x in loc]) == self.dim
		center_full = lambda loc: sum([center_1d(x) for x in loc]) == self.dim
		domain_1d = lambda x: 0 <= x <= 1
		domain_full = lambda loc: sum([domain_1d(x) for x in loc]) == self.dim
		not_center_full = lambda loc: domain_full(loc) and not loose_center_full(loc)

		# for dof search
		periodic_check = lambda x: 0 <= x < 1
		periodic_check_full = lambda loc: sum([periodic_check(x) for x in loc]) != self.dim
		dirichlet_check = lambda i,j: i in [0,len(dom)-1] or j in [0,len(dom)-1]

		i_edge1a = .25 if self.node else .25-3*H/2
		i_edge1b = .25 if self.node else .25+3*H/2
		i_edge2a = .75 if self.node else .75-3*H/2
		i_edge2b = .75 if self.node else .75+3*H/2
		block = lambda x: i_edge1a<= x <= i_edge2b
		slice = lambda x: (i_edge1a<=x<=i_edge1b) or (i_edge2a<=x<=i_edge2b)
		check1 = lambda loc: block(loc[0]) and slice(loc[1])
		check2 = lambda loc: block(loc[1]) and slice(loc[0])
		interface_check = lambda loc: check1(loc) or check2(loc)
		ghost_check = lambda loc: False

		if self.rtype == 0: # uniform
			check = lambda x: True
			emini = lambda x: -H < x < 1
			echeck = lambda loc: sum([emini(x) for x in loc]) == self.dim 
			interface_check = lambda x: True
			quadcheck = lambda loc: domain_full(loc)

		elif self.rtype == 1: # fine center
			mini = lambda x: (x <= edge1b) or (x >= edge2a)
			check = lambda loc: sum([mini(x) for x in loc]) >= 1

			eout = lambda x: -H < x < 1
			emini = lambda x: .25 <= x <= .75-H
			notcheck = lambda loc: sum([emini(x) for x in loc]) < self.dim
			outcheck = lambda loc: sum([eout(x) for x in loc]) == self.dim
			echeck = lambda loc: notcheck(loc) and outcheck(loc)

			quadcheck = lambda loc: not_center_full(loc)

		elif self.rtype == 2: # coarse center
			mini = lambda x: (x >= edge1a) and (x <= edge2b)
			check = lambda loc: sum([mini(x) for x in loc]) == self.dim

			emini = lambda x: .25-H < x < .75
			echeck = lambda loc: sum([emini(x) for x in loc]) == self.dim

			quadcheck = lambda loc: center_full(loc)

		else:
			raise ValueError('refinement type not supported')

		if self.dim == 2:
			for i,y in enumerate(dom):
				for j,x in enumerate(dom):
					if check([x,y]):
						d_ind_list.append([i,j])
						d_loc_list.append([x,y])
						d_periodic.append(periodic_check_full([x,y]))
						d_dirichlet.append(dirichlet_check(i,j))
					if echeck([x,y]):
						e_ind_list.append([i,j])
						e_loc_list.append([x,y])
						e_quads.append(self._get_el_quads([x,y],H,quadcheck))
					if interface_check([x,y]):
						int_ind_list.append([i,j])
						if ghost_check([x,y]):
							nearest_point = self._closest_point([x,y])
							print(True)
						else:
							nearest_point = None
						int_ghost_list.append(nearest_point)#ghost_check([x,y]))
						# int_loc_list.append([x,y])
		if self.dim == 3:
			for k,z in enumerate(dom):
				for i,y in enumerate(dom):
					for j,x in enumerate(dom):
						if check([x,y,z]):
							d_ind_list.append([i,j,k])
							d_loc_list.append([x,y,z])
							d_periodic.append(periodic_check_full([x,y,z]))
							d_dirichlet.append([dirichlet_check(i,j,k)])
						if echeck([x,y,z]):
							e_ind_list.append([i,j,k])
							e_loc_list.append([x,y,z])
							e_quads.append(self._get_el_quads([x,y,z],H,quadcheck))
		
		d_info = [d_ind_list,d_loc_list,d_periodic,d_dirichlet]
		e_info = [e_ind_list,e_loc_list,e_quads]
		i_info = [int_ind_list,int_ghost_list]
		return d_info,e_info,i_info,len(dom)

	def get_fine_info(self):
		H = self.h/2
		start = 0 if self.node else -H/2
		end = 1 if self.node else 1+H/2
		dom = np.linspace(start-H,
				  		  end+H,
						  2*self.N+3+self.cell)
		d_ind_list,d_loc_list = [],[]
		e_ind_list,e_loc_list = [],[]
		int_ind_list,int_ghost_list = [],[]
		e_quads = []
		d_periodic = []
		d_dirichlet = []

		edge1a = 0.25 - (self.cell*H/2) - H
		edge1b = 0.25 + (self.cell*H/2) + H
		edge2a = 0.75 - (self.cell*H/2) - H
		edge2b = 0.75 + (self.cell*H/2) + H
		
		center_1d = lambda x: .25 <= x <= .75
		center_full = lambda loc: sum([center_1d(x) for x in loc]) == self.dim
		loose_center_1d = lambda x: .25 < x < .75
		loose_center_full = lambda loc: sum([loose_center_1d(x) for x in loc]) == self.dim
		domain_1d = lambda x: 0 <= x <= 1
		domain_full = lambda loc: sum([domain_1d(x) for x in loc]) == self.dim

		not_center_full = lambda loc: domain_full(loc) and not loose_center_full(loc)

		periodic_check = lambda x: 0 <= x < 1
		periodic_check_full = lambda loc: sum([periodic_check(x) for x in loc]) != self.dim
		dirichlet_check = lambda i,j: i in [0,len(dom)-1] or j in [0,len(dom)-1]

		i_edge1a = .25 if self.node else .25-3*H/2
		i_edge1b = .25 if self.node else .25+3*H/2
		i_edge2a = .75 if self.node else .75-3*H/2
		i_edge2b = .75 if self.node else .75+3*H/2
		block = lambda x: i_edge1a<= x <= i_edge2b
		slice = lambda x: (i_edge1a<=x<=i_edge1b) or (i_edge2a<=x<=i_edge2b)
		check1 = lambda loc: block(loc[0]) and slice(loc[1])
		check2 = lambda loc: block(loc[1]) and slice(loc[0])
		interface_check = lambda loc: check1(loc) or check2(loc)
		if self.node:
			ghost_check = lambda loc: True
		else:
			ghost_check_x = lambda x: i_edge1b <= x <= edge2a
			ghost_check = lambda loc: sum([ghost_check_x(x) for x in loc]) == self.dim

		if self.rtype == 0: # uniform
			return [[],[]],[[],[],[]],[[],[]],0

		elif self.rtype == 1: # finecenter
			mini = lambda x: (x >= edge1a) and (x <= edge2b)
			check = lambda loc: sum([mini(x) for x in loc]) == self.dim

			emini = lambda x: .25-H < x < .75
			echeck = lambda loc: sum([emini(x) for x in loc]) == self.dim

			quadcheck = lambda loc: center_full(loc)

		elif self.rtype == 2: # coarse center
			mini = lambda x: (x <= edge1b) or (x >= edge2a)
			check = lambda loc: sum([mini(x) for x in loc]) >= 1

			eout = lambda x: -H < x < 1
			emini = lambda x: .25 <= x <= .75-H
			notcheck = lambda loc: sum([emini(x) for x in loc]) < self.dim
			outcheck = lambda loc: sum([eout(x) for x in loc]) == self.dim
			echeck = lambda loc: notcheck(loc) and outcheck(loc)

			quadcheck = lambda loc: not_center_full(loc)

		else:
			raise ValueError('refinement type not supported')

		if self.dim == 2:
			for i,y in enumerate(dom):
				for j,x in enumerate(dom):
					if check([x,y]):
						d_ind_list.append([i,j])
						d_loc_list.append([x,y])
						d_periodic.append(periodic_check_full([x,y]))
						d_dirichlet.append(dirichlet_check(i,j))
					if echeck([x,y]):
						e_ind_list.append([i,j])
						e_loc_list.append([x,y])
						e_quads.append(self._get_el_quads([x,y],H,quadcheck))
					if interface_check([x,y]):
						int_ind_list.append([i,j])
						if ghost_check([x,y]):
							nearest_point = self._closest_point([x,y])
						else:
							nearest_point = None
						int_ghost_list.append(nearest_point)#ghost_check([x,y]))
		if self.dim == 3:
			for k,z in enumerate(dom):
				for i,y in enumerate(dom):
					for j,x in enumerate(dom):
						if check([x,y,z]):
							d_ind_list.append([i,j,k])
							d_loc_list.append([x,y,z])
							d_periodic.append(periodic_check_full([x,y,z]))
							d_dirichlet.append([dirichlet_check(i,j,k)])
						if echeck([x,y,z]):
							e_ind_list.append([i,j,k])
							e_loc_list.append([x,y,z])
							e_quads.append(self._get_el_quads([x,y,z],H,quadcheck))

		d_info = [d_ind_list,d_loc_list,d_periodic,d_dirichlet]
		e_info = [e_ind_list,e_loc_list,e_quads]
		i_info = [int_ind_list,int_ghost_list]
		return d_info,e_info,i_info,len(dom)
	