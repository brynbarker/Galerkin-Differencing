import numpy as np
import matplotlib.pyplot as plt
from general_solve.refinement import UniformRefinement
from general_solve.refinement import StripeRefinement
from general_solve.refinement import SquareRefinement
from general_solve.patch import Patch

from general_solve import globals

refinement_class = {'uniform':UniformRefinement,
				   'stripe':StripeRefinement,
				   'square':SquareRefinement}
refinement_index = {'uniform':0,
				   'stripe':1,
				   'square':2}

def fill_quad(xs,y0,y1,q_id,diffs):
	if len(xs) == 2:
		if q_id == 0:
			new_xs = xs#[xs[0],xs[0]+diffs[0]/2]
			new_y0 = y0#[y0[0],sum(y0)/2]
			new_y1 = [min(y0)+diffs[1]/2]*2
		# if q_id == 1:
		# 	new_xs = [xs[0]+diffs[0]/2,xs[1]]
		# 	new_y0 = [sum(y0)/2,y0[1]]
		# 	new_y1 = [min(y0)+diffs[1]/2]*2
		if q_id == 2:
			new_xs = xs#[xs[0],xs[0]+diffs[0]/2]
			new_y0 = [min(y0)+diffs[1]/2]*2
			new_y1 = y1#[y1[0],sum(y1)/2]
		# if q_id == 3:
		# 	new_xs = [xs[0]+diffs[0]/2,xs[1]]
		# 	new_y0 = [min(y0)+diffs[1]/2]*2
		# 	new_y1 = [sum(y1)/2,y1[1]]
	else:
		# if q_id == 0:
		# 	new_xs = [xs[0],(xs[0]+xs[-1])/2]
		# 	new_y0 = [y0[0],y0[1]]
		# 	new_y1 = [(y0[0]+y1[0])/2,(y0[1]+y1[1])/2]
		if q_id == 1:
			new_xs = xs[:2]+[(xs[0]+xs[-1])/2]
			new_y0 = y0[:-1]#[y0[-2],y0[-1]]
			new_y1 = y1[:-1]#[(y0[-2]+y1[-2])/2,(y0[-1]+y1[-1])/2]
		# if q_id == 2:
		# 	new_xs = [xs[0],(xs[0]+xs[-1])/2]
		# 	new_y0 = [(y0[0]+y1[0])/2,(y0[1]+y1[1])/2]
		# 	new_y1 = [y1[0],y1[1]]
		if q_id == 3:
			new_xs = [(xs[0]+xs[-1])/2]+xs[-2:]
			new_y0 = y0[1:]#[(y0[-2]+y1[-2])/2,(y0[-1]+y1[-1])/2]
			new_y1 = y1[1:]#[y1[-2],y1[-1]]
	return new_xs,new_y0,new_y1




class Mesh:
	def __init__(self,N,dim,ords,dofloc='node',
			     rtype='uniform',rname=None,ghost_off=False,
				 zigzag=False):
		self.N = N 
		self.h = 1/N
		self.dim = dim
		self.rtype = rtype
		self.rindex = refinement_index[rtype]
		rClass = refinement_class[rtype]
		self.refinement = rClass(rname,dofloc,N,dim,ords,zigzag)
		self.dofloc = dofloc
		self.zigzag = zigzag

		coarse_info = self.refinement.get_coarse_info()
		fine_info = self.refinement.get_fine_info()
		coarse_patch = Patch(N,dim,coarse_info,dofloc,ords,level=0)

		e_id_shift = len(coarse_patch.elements)
		fine_patch = Patch(N,dim,fine_info,dofloc,ords,level=1,ghost_off=ghost_off,eshft=e_id_shift)
		self.patches = [coarse_patch,fine_patch]

		self.all_elements = []
		for p in self.patches:
			for e_id in p.elements:
				self.all_elements.append(p.elements[e_id])

		self.zigzag_elements = []

		self.dof_id_shift = len(coarse_patch.dofs)

	def add_zigzag_elements(self,traps,tris):
		self.trapezoid_elements = traps
		self.triangle_elements = tris
		self.zigzag_elements = traps+tris

	def set_quadrature(self,integrator):
		self.pts = np.array(integrator.points)
		self.wts = np.array(integrator.weights)
 
	def get_interface_lines(self):
		pts,wts = ((self.pts+1)/2)*self.h/4, self.wts*self.h/8
		interface_lines,comps,dirs = self.refinement.get_interface_lines()
		all_points,all_weights,all_shifts = [[],[]],[],[[],[]]

		for line,comp,dir in zip(interface_lines,comps,dirs):
			points = []
				
			val = line[0][1-comp]
			nodes = [nd[comp] for nd in line]
			for left in nodes[:-1]:
				points += list(pts+left)

			weights = list(wts) * len(nodes[:-1])
			all_points[comp] += points
			other_points = [val]*len(points)
			 
			all_points[1-comp] += other_points
			all_weights += weights

			all_shifts[comp] += [0]*len(points)
			all_shifts[1-comp] += [dir*1e-15]*len(points)

		point_arr = np.asarray(all_points).T
		weight_arr = np.asarray(all_weights)
		shift_arr = np.asarray(all_shifts).T

		if globals.DEBUG:
			plt.plot([0,0,1,1,0],[0,1,1,0,0],'k')
			plt.xticks([0,self.h,.25,.5,.75,1])
			plt.yticks([0,self.h,.25,.5,.75,1])
			plt.plot(point_arr[:,0],point_arr[:,1],'.')
			tmp = point_arr+1e15*self.h*shift_arr
			plt.plot(tmp[:,0],tmp[:,1],'*')
			
			plt.show()

		return point_arr,weight_arr,shift_arr

	def loc_to_el(self,loc):
		loc_patch_id = self.refinement.get_patch_id(loc)
		shift = self.dof_id_shift if loc_patch_id else 0
		try:
			return self.patches[loc_patch_id]._get_element_from_loc(loc), shift
		except:
			# zigzag element
			count = 0
			for e in self.zigzag_elements:
				count += 1
				if e.check_loc(loc):
					return e,shift
			raise ValueError('cant find the element for this point ({},{})'.format(loc[0],loc[1]))

	
	def collapse_null_space(self):
		sum_arrs = [p.sum_arr for p in self.patches]
		if self.rtype == 'uniform':
			return sum_arrs[0],None
		elif self.rtype == 'stripe':
			if self.refinement.rdim == self.patches[0].comp:
				sum_arrs[1] = sum_arrs[1][::2]+sum_arrs[1][1::2]
				return np.hstack(sum_arrs),None
			else:
				shapes = [arr.shape for arr in sum_arrs]
				zero_0 = np.zeros((shapes[0][0],shapes[1][1]))
				zero_1 = np.zeros((shapes[1][0],shapes[0][1]))
				return np.block([[sum_arrs[0],zero_0],
					 			 [zero_1,sum_arrs[-1]]]), None
		else:
			coarse_map, fine_map, cut = self.refinement.get_null_condensers()
			new_c = coarse_map @ sum_arrs[0]
			new_f = fine_map @ sum_arrs[-1]

			return np.hstack([new_c,new_f]),cut

	def view(self,split=True):
		if split:
			fig,axes = plt.subplots(1,2,figsize=(20,10))
		else:
			fig,ax = plt.subplots(1,figsize=(10,10))
		extra = max(.5,max(self.patches[0].ords)-1)

		for level in range(2):
			if split:
				ax = axes[level]
			H = self.h/(1+level)
			dom = np.linspace(0,1,(1+level)*self.N+1)
			ext_dom = np.linspace(-2*H,1+2*H,(1+level)*self.N+5)
			eps = H/8
			for x in dom:
				ax.plot([x,x],[0,1],'grey')
				ax.plot([0,1],[x,x],'grey')
			for x in ext_dom:
				ax.plot([x,x],[-2*H,1+2*H],'grey',ls=':')
				ax.plot([-2*H,1+2*H],[x,x],'grey',ls=':')

			for id in self.patches[level].elements:
				el = self.patches[level].elements[id]
				xs,y0,y1 = el.fill(eps)
				ax.fill_between(xs,y0,y1,alpha=.5)
				# ax.fill_between([el.x+eps,el.x+H-eps],
				# 	   el.y+eps,el.y+H-eps,alpha=.5)

			for id in self.patches[level].dofs:
				dof = self.patches[level].dofs[id]
				ax.plot(dof.x,dof.y,'k.')

			if split:
				c = 'lightgrey'
				if self.refinement.rshade[level] == 'all':
					ax.fill_between([0,1],0,1,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'in':
					ax.fill_between([.25,.75],.25,.75,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'out':
					ax.fill_between([0,1],0,.25,color=c,alpha=.5)
					ax.fill_between([0,1],.75,1,color=c,alpha=.5)
					ax.fill_between([0,.25],.25,.75,color=c,alpha=.5)
					ax.fill_between([.75,1],.25,.75,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'vstripe':
					ax.fill_between([.25,.75],0,1,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'hstripe':
					ax.fill_between([0,1],.25,.75,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'vedge':
					ax.fill_between([0,.25],0,1,color=c,alpha=.5)
					ax.fill_between([.75,1],0,1,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'hedge':
					ax.fill_between([0,1],0,.25,color=c,alpha=.5)
					ax.fill_between([0,1],.75,1,color=c,alpha=.5)



			if split or level==0:
				ax.set_aspect('equal')
				ax.set_xlim(-H*extra,1+H*extra)
				ax.set_ylim(-H*extra,1+H*extra)

		# for zz_el in self.zigzag_elements:
		# 	xs,y0,y1 = zz_el.fill(eps)
		# 	ax.fill_between(xs,y0,y1,alpha=.5)


		plt.show()

	def view_detailed(self,large=True,split=False):
		if self.dim ==2:
			quad_bounds = [[0,.5,0,.5],
				[.5,1,0,.5],[0,.5,.5,1],[.5,1,.5,1]]
		if self.dim ==3:
			quad_bounds = [[0,.5,0,.5,0,.5],
				[.5,1,0,.5,0,.5],[0,.5,.5,1,0,.5],[.5,1,.5,1,0,.5],
				[0,.5,0,.5,.5,1],[.5,1,0,.5,.5,1],
				[0,.5,.5,1,.5,1],[.5,1,.5,1,.5,1]]
		if split:
			fgsz = (20,10) if large else (6,3)
			fig,axes = plt.subplots(1,2,figsize=fgsz)
		else:
			fgsz = (10,10) if large else (3,3)
			fig,ax = plt.subplots(1,figsize=fgsz)

		for level in range(2):
			if split: ax = axes[level]
			H = self.h/(1+level)
			dom = np.linspace(0,1,(1+level)*self.N+1)
			ext_dom = np.linspace(-2*H,1+2*H,(1+level)*self.N+5)
			eps = 0
			for x in dom:
				ax.plot([x,x],[0,1],'grey',lw=.7)
				ax.plot([0,1],[x,x],'grey',lw=.7)
			for x in ext_dom:
				ax.plot([x,x],[-2*H,1+2*H],'grey',ls=':',lw=.7)
				ax.plot([-2*H,1+2*H],[x,x],'grey',ls=':',lw=.7)

			for id in self.patches[level].elements:
				el = self.patches[level].elements[id]
				x0,x1,y0,y1 = el.bounds
				ax.plot(el.to_plot[0],el.to_plot[1],'k',lw=.4)
				lens = [max(v)-min(v) for v in el.to_plot]
				# lens = np.array(el.bounds[1::2])-np.array(el.bounds[::2])
				for j,(quad,q_bool) in enumerate(zip(quad_bounds,el.quads)):
					if q_bool:
						if el.regular:
							quad_bound = []
							for ind in range(self.dim):
								diff = lens[ind]
								strt = el.bounds[2*ind]
								for shft in quad[2*ind:2*ind+2]:
									quad_bound.append(strt+shft*diff)
							x0,x1,y0,y1 = quad_bound
							ax.fill_between([x0+eps,x1-eps],
								y0+eps,y1-eps,alpha=.4)
						else:
							xs,y0,y1 = el.fill(eps)
							# print(el.tri,el.map_type)
							# print(xs,y0,y1)
							if el.tri:
								nxs,ny0,ny1 = xs,y0,y1
							else:
								nxs,ny0,ny1 = fill_quad(xs,y0,y1,j,lens)
							# print( nxs,ny0,ny1)
							# print()
							ax.fill_between(nxs,ny0,ny1,alpha=.4)


			for id in self.patches[level].dofs:
				dof = self.patches[level].dofs[id]
				ax.plot(dof.x,dof.y,'k.',ms=2)

			if split:
				c = 'lightgrey'
				if self.refinement.rshade[level] == 'all':
					ax.fill_between([0,1],0,1,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'in':
					ax.fill_between([.25,.75],.25,.75,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'out':
					ax.fill_between([0,1],0,.25,color=c,alpha=.5)
					ax.fill_between([0,1],.75,1,color=c,alpha=.5)
					ax.fill_between([0,.25],.25,.75,color=c,alpha=.5)
					ax.fill_between([.75,1],.25,.75,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'vstripe':
					ax.fill_between([.25,.75],0,1,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'hstripe':
					ax.fill_between([0,1],.25,.75,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'vedge':
					ax.fill_between([0,.25],0,1,color=c,alpha=.5)
					ax.fill_between([.75,1],0,1,color=c,alpha=.5)
				elif self.refinement.rshade[level] == 'hedge':
					ax.fill_between([0,1],0,.25,color=c,alpha=.5)
					ax.fill_between([0,1],.75,1,color=c,alpha=.5)

			ax.set_aspect('equal')
		
		# for zz_el in self.zigzag_elements:
		# 	xs,y0,y1 = zz_el.fill(0)
		# 	if split: ax = axes[1]
		# 	ymid = [(y0[i]+y1[i])/2 for i in range(len(y0))]
		# 	if len(xs) == 2:
		# 		xmid = sum(xs)/2
		# 		y0mid = sum(y0)/2
		# 		y1mid = sum(y1)/2
		# 		ymidmid = sum(ymid)/2
		# 	else:
		# 		xmid = xs[1]
		# 		y0mid = y0[1]
		# 		y1mid = y1[1]
		# 		ymidmid = ymid[1]
		# 	xlims = [[xs[0],xmid],[xmid,xs[-1]]]
		# 	y0lims = [[y0[0],y0mid],[y0mid,y0[-1]],[ymid[0],ymidmid],[ymidmid,ymid[-1]]]
		# 	y1lims = [[ymid[0],ymidmid],[ymidmid,ymid[-1]],[y1[0],y1mid],[y1mid,y1[-1]]]

		# 	for quad_id in range(4):
		# 		if zz_el.quads[quad_id]:
		# 			xind = quad_id % 2
		# 			this_x,this_y0,this_y1 = xlims[xind],y0lims[quad_id],y1lims[quad_id]

		# 			ax.fill_between(this_x,this_y0,this_y1,alpha=.5)

		plt.show()
	
	def vis_dof_sol(self,U,locs=None,log=True,lines=False):
		
		fig,ax = plt.subplots(figsize=(10,10))

		c_vals = []

		for level in range(2):

			H = self.h/(1+level)
			dom = np.linspace(0,1,(1+level)*self.N+1)
			ext_dom = np.linspace(-2*H,1+2*H,(1+level)*self.N+5)
			for x in dom:
				ax.plot([x,x],[0,1],'grey',zorder=0)
				ax.plot([0,1],[x,x],'grey',zorder=0)
			for x in ext_dom:
				ax.plot([x,x],[-2*H,1+2*H],'grey',ls=':',zorder=0)
				ax.plot([-2*H,1+2*H],[x,x],'grey',ls=':',zorder=0)
		if self.rindex==2:
			ax.plot([.25,.75,.75,.25,.25],[.25,.25,.75,.75,.25],'k')
		elif self.rindex == 1:
			ops = [[.25,.25],[0,1]]
			ax.plot(ops[self.refinement.rtype>2],ops[self.refinement.rtype<2],'k')	
			ops = [[.75,.75],[0,1]]
			ax.plot(ops[self.refinement.rtype>2],ops[self.refinement.rtype<2],'k')	

		if locs is None:
			locs = []
			for index,e in enumerate(self.all_elements):
				c_vals.append(U[index])
				locs.append(e.mid)
		else:
			c_vals = list(U)

		x,y = np.array(locs).T
		if log:
			logvals = [np.log(cval) for cval in c_vals]
			plot = ax.scatter(x,y,c=logvals,cmap='jet',zorder=1)	
		else:
			plot = ax.scatter(x,y,c=c_vals,cmap='jet',zorder=1)	

		if lines:
			for e in self.all_elements:
				ax.plot(e.to_plot[0],e.to_plot[1],'white',lw=.5)
		fig.colorbar(plot,location='right')

		ax.set_aspect('equal')
		plt.show()

	def evaluate_on_each_element(self,func1,func2=None):
		if func2 is not None:
			func = lambda x: abs(func1(x)-func2(x))
		else:
			func = func1

		fig,ax = plt.subplots(1,2,figsize=(20,10))

		c_vals = {0:[],1:[]}
		locs = {0:[],1:[]}

		cbar_loc = ['left','right']

		for level in range(2):
			if True:#try:
				H = self.h/(1+level)
				dom = np.linspace(0,1,(1+level)*self.N+1)
				ext_dom = np.linspace(-2*H,1+2*H,(1+level)*self.N+5)
				eps = H/8
				for x in dom:
					ax[level].plot([x,x],[0,1],'grey')
					ax[level].plot([0,1],[x,x],'grey')
				for x in ext_dom:
					ax[level].plot([x,x],[-2*H,1+2*H],'grey',ls=':')
					ax[level].plot([-2*H,1+2*H],[x,x],'grey',ls=':')

				for id in self.patches[level].elements:
					e = self.patches[level].elements[id]
					center = [e.x+H/2,e.y+H/2]
					c_vals[level].append(func(center))
					locs[level].append(center)

				x,y = np.array(locs[level]).T
				plot = ax[level].scatter(x,y,c=c_vals[level],cmap='jet')	
				fig.colorbar(plot,location=cbar_loc[level])

				ax[level].set_aspect('equal')
			else:#except:
				pass

		plt.show()

		
	def vis_patches_together(self):
		# fig = plt.figure(figsize=(10,10))
		plt.plot([0,1,1,0,0],[0,0,1,1,0],'k')
		if self.rtype=='stripe':
			if self.refinement.rtype < 2:
				plt.plot([.25,.25,.75,.75],[0,1,1,0],'k')
			else:
				plt.plot([0,1,1,0],[.25,.25,.75,.75],'k')
		elif self.rtype=='square':
			plt.plot([.25,.75,.75,.25,.25],[.25,.25,.75,.75,.25],'k')

		for p_id,p in enumerate(self.patches):
			rm = 'o' if p_id else 's'
			for lookup_id in p.dofs:
				count = 0
				dof = p.dofs[lookup_id]
				filltype='full'
				m = '.'
				if lookup_id in p.interface_dofs:
					m = rm
					count += 1
				if lookup_id in p.interface_ghosts:
					m = rm
					filltype='none'
					count += 1
				if count > 1:
					m = '*'
				if count == 0:
					m = 'k'+m
				plt.plot(dof.x,dof.y,m,fillstyle=filltype)
		plt.show()