import numpy as	np
import matplotlib.pyplot as	plt
from general_solve.element import Element,TriElement,TrapElement
from general_solve.dof import DoF
from general_solve import globals

refinement_type	= {'uniform':0,
				   'finecenter':1,
				   'coarsecenter':2}

class Patch:
	def	__init__(self,N,dim,refinement_info,dtype,ords,level=0,ghost_off=False,eshft=0):
		self.N = (1+level)*N
		self.h = 1/self.N
		self.dim = dim
		self.info =	refinement_info[:-1]
		self.lens =	refinement_info[-1]
		self.dofs =	{}
		self.elements =	{}
		self.periodic_pairs	= {}
		self.cell =	dtype=='cell'
		self.node =	dtype=='node'
		self.xside = dtype == 'xside'
		self.yside = dtype == 'yside'
		self.ords =	ords
		self.Ls	= [int(ord/2) for ord in self.ords]

		self.alt_dof = {}
		self.alt_el	= {}
		self.level = level
		self.corners = []

		self.size =	len(self.info[0][0])

		self.comp =	None
		if sum(ords) ==	1:
			self.comp =	ords.index(1)
			self.sum_arr = np.zeros((self.lens[1-self.comp],self.size))

		self.ghost_off = ghost_off

		self._setup(eshft)

		if self.comp is	not	None:
			keeps =	np.sum(self.sum_arr,axis=1)!=0
			self.sum_arr = self.sum_arr[keeps,:]


	def	get_dof(self,non_lookup_id):
		lookup_id =	self.alt_dof[non_lookup_id]
		dof	= self.dofs[lookup_id]
		assert dof.ID == non_lookup_id
		return dof

	def	get_el(self,non_lookup_id):
		lookup_id =	self.alt_el[non_lookup_id]
		el = self.elements[lookup_id]
		assert el.ID ==	non_lookup_id
		return el 
		
	def	_get_lookup_id_from_ind(self,ind):
		if self.dim	== 2:
			i,j	= ind
			return i*self.lens[0]+j
		elif self.dim == 3:
			i,j,k =	ind
			return k*self.lens[0]*self.lens[1]+i*self.lens[0]+j

	def	_get_lookup_id_from_loc(self,loc,tri=False):
		shftx =	0 if (self.node	or self.xside) else	1/2
		if shftx!=0	and	self.ords[0]==0: shftx-=1
		shfty =	0 if (self.node	or self.yside) else	1/2
		if shfty!=0	and	self.ords[1]==0: shfty-=1
		shft = [shftx,shfty]
		if self.dim	== 2:
			[j,i] =	[int(x/self.h+shft[ind]+self.Ls[ind]) for ind,x	in enumerate(loc)]
			ind	= [i,j]
		elif self.dim == 3:
			[j,i,k]	= [int(x/self.h+shft+self.Ls[ind]) for ind,x in	enumerate(loc)]
			ind	= [i,j,k]

		flp	= -1 if	tri	else 1
		return flp*self._get_lookup_id_from_ind(ind)

	def	_get_element_from_loc(self,loc_in):
		loc	= np.copy(loc_in)
		loc	= [x - (x==1)*1e-12	for	x in loc]
		# if self.yside: loc[0]+=self.h/2
		# if self.xside: loc[1]+=self.h/2
		
		# loc =	[x + (x==0)*1e-12 for x	in loc]
		el_lookup_id = self._get_lookup_id_from_loc(loc)
		e =	self.elements[el_lookup_id]
		e.check_loc(loc_in)
		return e

	def	_get_periodic_pair(self,loc):
		def	get_shift(x):
			if x < 0:
				return 1
			elif x >= 1:
				return -1
			return 0
		shifts = [get_shift(x) for x in	loc]
		pair_loc = [x+shft for (x,shft)	in zip(loc,shifts)]
		return self._get_lookup_id_from_loc(pair_loc)

	def	_setup(self,eshft=0):
		d_info,e_info,i_info = self.info

		for	id in range(self.size):
			ind,loc,per	= d_info[0][id],d_info[1][id],d_info[2][id]#,d_info[3][id]
			newdof = DoF(id,self.dim,ind,loc,self.h,self.ords)

			if self.comp is	not	None:
				index =	newdof.j if	self.comp else newdof.i
				self.sum_arr[index,id] = 1

			lookup_id =	self._get_lookup_id_from_ind(ind)
			self.dofs[lookup_id] = newdof
			self.alt_dof[id] = lookup_id
			if per:
				pair_lookup_id = self._get_periodic_pair(loc)
				self.periodic_pairs[lookup_id] = pair_lookup_id
			# if low:
			#	self.corners.append(lookup_id)
		# print(self.sum_arr)

		el_count = len(e_info[0])
		trap_els,tri_els = [None],[]
		for	id in range(el_count):
			ind,loc,quads,extra	= e_info[0][id],e_info[1][id],e_info[2][id],e_info[3][id]
			if extra:
				newel =	TrapElement(id,self.dim,ind,np.array(loc),self.h,self.ords)
				newel.set_corners(self.xside,self.yside)

				if quads:
					triel =	TriElement(len(tri_els)+el_count,
									self.dim,ind,np.array(loc),
									self.h,self.ords)
					triel.set_corners(self.xside,self.yside)

					el_lookup =	self._get_lookup_id_from_ind(triel.ind_shift)
					if el_lookup in	self.elements:
						tmp_trap = self.elements[el_lookup]
						if not tmp_trap.regular	and	not	tmp_trap.tri:
							tmp_trap.add_tri(triel)
					# if trap_els[-1] is not None:
					#	trap_els[-1].add_tri(triel)
					newel.add_tri(triel)
					tri_els.append(triel)



				trap_els.append(newel)
			else:
				newel =	Element(id,self.dim,ind,np.array(loc),self.h,self.ords)
				newel.set_support(quads)
			dof_lookup_id =	self._get_lookup_id_from_loc(loc)
			strt = dof_lookup_id-self.Ls[0]-self.Ls[1]*self.lens[0]#1-self.xlen
			newel.add_dofs(strt,self.lens[0])
			el_lookup_id = self._get_lookup_id_from_ind(ind)
			check_id = self._get_lookup_id_from_loc(loc)
			assert(check_id==el_lookup_id)
			self.elements[el_lookup_id]	= newel
			if extra and quads:
				self.elements[-el_lookup_id] = triel

		for	e in self.elements.values():
			if eshft > 0:
				e.set_global_ID(eshft)
			if e.regular or not e.tri:
				e.update_dofs(self.dofs)
		for	e in tri_els:
			# el_lookup_id = self._get_lookup_id_from_ind(e.ind)
			# el_lookup_id_shift = self._get_lookup_id_from_ind(e.ind_shift)
			# for lookup_id	in [el_lookup_id_shift,el_lookup_id]:
			#	if lookup_id in	self.elements:
			#		tmp_trap = self.elements[lookup_id]
			#		if not tmp_trap.regular	and	not	tmp_trap.tri:
			#			tmp_trap.add_tri(e)
			# self.elements[-el_lookup_id] = e
			# if eshft > 0:
			# 	e.set_global_ID(eshft)
			e.update_dofs(self.dofs)

		self.zigzag	= [trap_els[1:],tri_els]

		for dof in self.dofs.values():
			dof.update()
		self._setup_interface()

	def	_setup_interface(self):
		inds,ghosts,zinds =	self.info[-1]
		self.interface_dofs	= []
		self.interface_ghosts =	[]
		self.interface_points =	[]
		self.zigzag_interface =	[]
		self.zigzag_ghosts = []

		if self.ghost_off:
			self.ghost_count = 0
			return
		for	(ind,ghost_loc)	in zip(inds,ghosts):#,zinds):
			# ind =	rind if	rind is	not	None else zind
			dof_lookup_id =	self._get_lookup_id_from_ind(ind)
			dof	= self.dofs[dof_lookup_id]
			# if zind is not None:
			#	self.zigzag_interface.append(dof_lookup_id)
			if globals.LAG:
				if ghost_loc is	not	None:
					self.interface_ghosts.append(dof_lookup_id)
					self.interface_points.append(ghost_loc)
				else:
					self.interface_dofs.append(dof_lookup_id)
			else:
				if self.level == 0:
					self.interface_dofs.append(dof_lookup_id)
				else:
					self.interface_ghosts.append(dof_lookup_id)
				# if rind is not None:
				#	if self.level == 0:
				#		# self.interface_dofs.append(dof_lookup_id)
				#		if zind	is None:
				#			self.interface_dofs.append(dof_lookup_id)
				#	elif zind is not None and ghost_loc	is None:
				#		self.interface_ghosts.append(dof_lookup_id)
				#	elif zind is None and ghost_loc	is not None:
				#		self.interface_ghosts.append(dof_lookup_id)
				#	elif zind is not None and ghost_loc	is not None:
				#		pass
				#	else:
				#		self.interface_dofs.append(dof_lookup_id)
				# if zind is not None and ghost_loc	is not None:
				#	dof_id = self.dofs[dof_lookup_id].ID
				#	self.zigzag_ghosts.append(dof_id)

		self.ghost_count = len(self.interface_ghosts)

		if globals.DEBUG:
			ttls = ['interface dofs','interface	ghosts','zigzag	interface','zigzag ghosts']
			fig,ax = plt.subplots(1,4,figsize=(10,3))
			for	i in range(4):
				ax[i].plot([.25,.25,.75,.75,.25],[0,1,1,0,0],'k')
				ax[i].plot([0,0,1,1,0],[0,1,1,0,0],'k')
				ax[i].set_title(ttls[i])
			for	g in self.interface_dofs:
				gdof = self.dofs[g]
				ax[0].plot(gdof.x,gdof.y,'o')
			for	g in self.interface_ghosts:
				gdof = self.dofs[g]
				ax[1].plot(gdof.x,gdof.y,'o')
			for	g in self.zigzag_interface:
				gdof = self.dofs[g]
				ax[2].plot(gdof.x,gdof.y,'o')
			for	zzg	in self.zigzag_ghosts:
				zzgdof = self.get_dof(zzg)
				ax[3].plot(zzgdof.x,zzgdof.y,'o')
				
			plt.show()

	def	evaluate_interface_lines(self,lines):
		evals =	np.zeros((len(self.interface_dofs),len(lines)))
		for	i,dof_id in	enumerate(self.interface_dofs):
			dof	= self.dofs[dof_id]
			for	j,loc in enumerate(lines):
				evals[i,j] = dof.phi(loc,glob=True)
		return evals


	def	evaluate_interface_points(self,eval_points):
		evals =	np.zeros((len(eval_points),len(self.interface_dofs)))

		for	i,loc in enumerate(eval_points):
			for	j,dof_id in	enumerate(self.interface_dofs):
				dof	= self.dofs[dof_id]
				evals[i,j] = dof.phi(loc,glob=True)
		return evals

	def	evaluate_interface_ghosts(self,lines=None):
		if len(self.interface_ghosts) == 0:
			return None

		if globals.LAG:
			tmp	= len(self.interface_ghosts)
			ghost_arr =	np.zeros((tmp,tmp))
			for	i,loc in enumerate(self.interface_points):
				for	j,dof_id in	enumerate(self.interface_ghosts):
					dof	= self.dofs[dof_id]
					val	= dof.phi(loc,glob=True)
					ghost_arr[i,j] = val
			return ghost_arr

		else:
			evals =	np.zeros((len(self.interface_ghosts),len(lines)))
			for	i,dof_id in	enumerate(self.interface_ghosts):
				dof	= self.dofs[dof_id]
				for	j,loc in enumerate(lines):
					evals[i,j] = dof.phi(loc,glob=True)
			return evals


	def	vis(self,rtype=None):
		# fig =	plt.figure(figsize=(10,10))
		plt.plot([0,1,1,0,0],[0,0,1,1,0],'k')
		if rtype is	not	None:
			if rtype=='stripe':
				plt.plot([.25,.25,.75,.75],[0,1,1,0],'k')
			elif rtype=='square':
				plt.plot([.25,.75,.75,.25,.25],[.25,.25,.75,.75,.25],'k')
		for	lookup_id in self.dofs:
			count =	0
			dof	= self.dofs[lookup_id]
			filltype='full'
			m =	'.'
			if lookup_id in	self.interface_dofs:
				m =	'o'
				count += 1
			if lookup_id in	self.interface_ghosts:
				m =	'o'
				filltype='none'
				count += 1
			if count > 1:
				m =	'*'
			if count ==	0:
				m =	'k'+m
			plt.plot(dof.x,dof.y,m,fillstyle=filltype)
		plt.show()


	def	vis_interface_eval_points(self,rtype=None):
		if len(self.interface_ghosts) == 0:
			print('no ghosts on	this patch')
			return
		plt.plot([0,1,1,0,0],[0,0,1,1,0],'k')
		if rtype is	not	None:
			if rtype=='stripe':
				plt.plot([.25,.25,.75,.75],[0,1,1,0],'k')
			elif rtype=='square':
				plt.plot([.25,.75,.75,.25,.25],[.25,.25,.75,.75,.25],'k')
		for	j,(g,pt) in	enumerate(zip(self.interface_ghosts,self.interface_points)):
			gdof = self.dofs[g]
			color =	'C'+str(j%10)
			plt.plot([gdof.x],[gdof.y],'o',fillstyle='none',ms=10,c=color)
			plt.plot([pt[0]],[pt[1]],'.',c=color)
			plt.plot([gdof.x,pt[0]],[gdof.y,pt[1]],c=color)
		plt.show()