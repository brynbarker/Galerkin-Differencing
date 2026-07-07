from general_solve import shape_functions as sf


class DoF:
	def __init__(self,ID,dim,inds,loc,h,ords):#=[3,3]):
		self.ID = ID
		self.dim = dim
		self.loc = loc
		self.ind  = inds
		self.ords = ords

		self.i,self.j = inds
		self.x,self.y = loc
		self.k,self.z = None, None

		self.interface = False

		self.update_phi = False
		# self.phi = lambda xy,el=None,glob=True: sf.phi_2d_eval(self.ords,xy[0],xy[1],
		# 						          h,self.x,self.y)
		
		# self.dphi = lambda xy: sf.dphi_2d_eval(self.ords,xy[0],xy[1],
		# 						            h,self.x,self.y)

		self.h = h
		self.elements = {}
		self.duplicates = False
		self.other_els = []
		self.ref_shifts = {}
		# self.zz_elements = []

	def phi(self,xy,el=None,glob=True,force=False,check=None):
		if not force and self.update_phi:
			return self.updated_phi(xy,el,glob,check)

		if glob:
			return sf.phi_2d_eval(self.ords,xy[0],xy[1],self.h,self.x,self.y)
		else:
			return sf.phi_2d(self.ords,xy[0],xy[1],1)

	def dphi(self,xy,el=None,glob=True,force=False,check=None):
		if not force and self.update_phi:
			return self.updated_dphi(xy,el,glob,check)

		if glob:
			return sf.dphi_2d_eval(self.ords,xy[0],xy[1],self.h,self.x,self.y)
		else:
			return sf.dphi_2d(self.ords,xy[0],xy[1],1)

	def updated_phi(self,xy,el,glob,check=None):
		if el is None:
			for e_global_id in self.elements:
				if self.elements[e_global_id].check_loc(xy):
					el = self.elements[e_global_id]
					if check is not None:
						if not el.global_ID==check[0]:
							print(check,(el.global_ID,el.x,el.y,el.regular,el.tri),xy)
					break

		if el is None:
			return 0

		if el.regular:
			return self.phi(xy,glob=glob,force=True)
		
		i,j = (el.x-self.x)/self.h, (el.y-self.y)/self.h
		if glob:
			xi,eta = el.phi_input_global(xy[0],xy[1])
		else:
			xi,eta = el.phi_input_local(xy[0],xy[1])
		return self.phi([xi+i,eta+j],glob=False,force=True)

	def updated_dphi(self,xy,el,glob):
		if el is None:
			for e_global_id in self.elements:
				if self.elements[e_global_id].check_loc(xy):
					el = self.elements[e_global_id]
					break

		if el is None:
			return 0

		if el.regular:
			return self.dphi(xy,glob=glob,force=True)
		
		i,j = (el.x-self.x)/self.h, (el.y-self.y)/self.h
		if glob:
			xi,eta = el.phi_input_global(xy[0],xy[1])
		else:
			xi,eta = el.phi_input_local(xy[0],xy[1])
		return self.phi([xi+i,eta+j],glob=False,force=True)

	def add_element(self,e,shfts=None):
		if e.global_ID not in self.elements:
			self.elements[e.global_ID] = e
			if not e.regular:
				xi0 = min(1,int((self.x-e.x)/e.h))
				eta0 = min(1,int((self.y-e.y)/e.h))
				self.ref_shifts[e.global_ID] = [xi0,eta0]
				self.interface = True
		elif self.elements[e.global_ID] != e:
			other_e = self.elements[e.global_ID]
			print((e.regular,other_e.regular),(e.ID,other_e.ID))

			self.duplicates = True
			self.other_els.append(e)
		else:
			if not e.regular:
				self.ref_shifts[e.global_ID].append(True)

	def remove_element(self,e_global_ID):
		del self.elements[e_global_ID]

	def update(self):
		if self.interface:
			self.update_phi = True

	def no_set_phi(self):
		self.update_phi = True
		return
		ind_to_shifts = {0:[1,0],1:[0,0],2:[-1,0],3:[1,-1],4:[0,-1],5:[-1,-1]}
		# e_inds = {}
		# for e_id in self.elements:
		# 	e = self.elements[e_id]
		# 	ind_first = e.dof_list.index(self)
		# 	ind = [ind_first]
		# 	if e.dof_list.count(self) > 1:
		# 		ind_second = e.dof_list[ind_first+1:].index(self)
		# 		ind.append(ind_second+ind_first+1)
		# 	e_inds[e_id] = ind

		orig_phi = self.phi
		orig_dphi = self.dphi

		def phi(loc,el=None,loc_id=None,glob=False):
			if el is None:
				for e_global_id in self.elements:
					if self.elements[e_global_id].check_loc(loc):
						el = self.elements[e_global_id]
						break

			if el is None:
				return 0

			if el.regular and el.h==self.h:
				if glob:
					return orig_phi(loc)
				return orig_phi([loc[0]+el.x,loc[1]+el.y])

			elif el.regular:
				print('this should not happen')
				return orig_phi(loc)
				# new_x = 2*loc[0]-self.x
				# new_y = 2*loc[1]-self.y
				# if self.y < el.y:
				# 	return orig_phi([new_x,new_y-self.h/2])
				# elif self.y > el.y+el.h:
				# 	return orig_phi([new_x,new_y+self.h/2])
				# else:
				# 	val0 = orig_phi([new_x,new_y-self.h/2])
				# 	val1 = orig_phi([new_x,new_y+self.h/2])
				# 	return val0+val1

			i = (el.x-self.x)/self.h
			j = (el.y-self.y)/self.h

			if glob:
				xi,eta = el.phi_input_global(loc[0],loc[1])
				# xi,eta = el.inv_transform(loc[0],loc[1])
			else:
				xi,eta = el.phi_input_local(loc[0],loc[1])
			return orig_phi([self.x+self.h*(xi+i),self.y+self.h*(eta+j)])
			# 	xi,eta = loc
			# nu,rho = el.phi_input_local(xi,eta)
			# return orig_phi([nu+el.x0,rho+el.y0])
			
			# double = False
			# if loc_id is None:
			# 	loc_id = el.dof_list.index(self)
			# 	if el.dof_list.count(self)>1:
			# 		double = True

			# xi_shift,eta_shift = self.ref_shifts[el.global_ID]#ind_to_shifts[loc_id]#[el.ID][0]
			# xi0,eta0 = el.inv_transform(self.x,self.y)
			if el.tri:
				nu,rho = el.phi_input_local(xi,eta)
				return orig_phi(nu+el.x0,rho+el.y0)

				# mod,prod = eta0[1]-eta0[0],eta[0]*eta0[0]+eta[1]*eta0[1]
				# if el.map_type % 2 == 0:
				# 	if sum(eta0) < 0:
				# 		eta = mod*prod
				# 	elif sum(eta0) == 0:
				# 		eta = 0
				# 		return bspline3(xi,1)+(xi-1)*bspline3(xi+1,1)+(1-xi)*bspline3(xi-1,1)
				# 	else:
				# 		eta = mod*(prod-1)
				# 	eta0 = 0
				# else:
				# 	if sum(xi0) < 0:
				# 		xi = 1+(xi[0]*xi0[0]+xi[1]*xi0[1])
				# 	elif sum(xi0) == 0:
				# 		xi = 0
				# 	else:
				# 		xi = xi[0]*xi0[0]+xi[1]*xi0[1]
				# 	xi0 = 0
				# return bspline3(xi-xi0,1)*bspline2(eta-eta0,1)
				# return bspline2(eta-eta0,1)
			return phi_2d(self.ords,xi-xi0,eta-eta0,1)

			# if double:
			# 	second_id = el.dof_list[loc_id+1:].index(self)+loc_id+1

			# 	xi_shift,eta_shift = ind_to_shifts[second_id]#[el.ID][0]
			# 	output += phi_2d(self.ords,xi+xi_shift,eta+eta_shift,1)

			return output

			# if len(e_shifts[el.ID]) > 1:
			# 	xi_shift,eta_shift = e_shifts[el.ID][1]
			# 	output += phi_2d(self.ords,xi+xi_shift,eta+eta_shift,1)
			
			# return output

		self.phi = phi

		def dphi(loc,el=None,loc_id=None,glob=False):
			if el is None:
				for e_global_id in self.elements:
					if self.elements[e_global_id].check_loc(loc):
						el = self.elements[e_global_id]
						break

			if el is None:
				return 0

			if el.regular and el.h==self.h:
				return orig_dphi(loc)
			elif el.regular:
				new_x = 2*loc[0]-self.x
				new_y = 2*loc[1]-self.y
				if self.y < el.y:
					return orig_dphi([new_x,new_y-self.h/2])*2
				elif self.y > el.y+el.h:
					return orig_dphi([new_x,new_y+self.h/2])*2
				else:
					val0 = orig_dphi([new_x,new_y-self.h/2])*2
					val1 = orig_dphi([new_x,new_y+self.h/2])*2
					return val0+val1

			if glob:
				xi,eta = el.inv_transform(loc[0],loc[1])
			else:
				xi,eta = loc
			
			nu,rho = el.phi_input_local(xi,eta)
			return orig_phi(nu+el.x0,rho+el.y0)
			double = False
			if loc_id is None:
				loc_id = el.dof_list.index(self)
				if el.dof_list.count(self)>1:
					double = True


			xi_shift,eta_shift = ind_to_shifts[loc_id]#[el.ID][0]
			output = phi_2d(self.ords,xi+xi_shift,eta+eta_shift,1)

			if double:
				second_id = el.dof_list[loc_id+1:].index(self)+loc_id+1

				xi_shift,eta_shift = ind_to_shifts[second_id]#[el.ID][0]
				output += phi_2d(self.ords,xi+xi_shift,eta+eta_shift,1)

			return output

			# if len(e_shifts[el.ID]) > 1:
			# 	xi_shift,eta_shift = e_shifts[el.ID][1]
			# 	output += phi_2d(self.ords,xi+xi_shift,eta+eta_shift,1)
			
			# return output

		# self.phi = phi



