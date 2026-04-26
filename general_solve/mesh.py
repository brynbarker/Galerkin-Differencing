import numpy as np
import matplotlib.pyplot as plt
from general_solve.refinement import UniformRefinement
from general_solve.refinement import StripeRefinement
from general_solve.refinement import SquareRefinement
from general_solve.patch import Patch

refinement_class = {'uniform':UniformRefinement,
				   'stripe':StripeRefinement,
				   'square':SquareRefinement}
refinement_index = {'uniform':0,
				   'stripe':1,
				   'square':2}


class PseudoMesh:
	def __init__(self,N,dim=2,rtype='uniform',rname=None):
		self.rindex = refinement_index[rtype]
		rClass = refinement_class[rtype]
		refinement = rClass(rname,'node',N,dim,[1,1])
		
		fine_info = refinement.get_fine_info()

		coarse_info = refinement.get_coarse_info()
		H = 1/N
		quad_shifts_x = [H/4,3*H/4,H/4,3*H/4]
		quad_shifts_y = [H/4,H/4,3*H/4,3*H/4]
		self.coarse_quads = []
		for id in range(len(coarse_info[1][0])):
			loc,quads = coarse_info[1][1][id],coarse_info[1][2][id]
			x,y = loc
			my_quads = {}
			for quad_id,quad in enumerate(quads):
				if quad:
					new_x = x + quad_shifts_x[quad_id]
					new_y = y + quad_shifts_y[quad_id]
					my_quads[quad_id] = (new_x,new_y)
			self.coarse_quads.append(my_quads)
		

		fine_info = refinement.get_fine_info()
		H = 1/N/2
		quad_shifts_x = [H/4,3*H/4,H/4,3*H/4]
		quad_shifts_y = [H/4,H/4,3*H/4,3*H/4]
		self.fine_quads = []
		for id in range(len(fine_info[1][0])):
			loc,quads = fine_info[1][1][id],fine_info[1][2][id]
			x,y = loc
			my_quads = {}
			for quad_id,quad in enumerate(quads):
				if quad:
					new_x = x + quad_shifts_x[quad_id]
					new_y = y + quad_shifts_y[quad_id]
					my_quads[quad_id] = (new_x,new_y)
			self.fine_quads.append(my_quads)

class Mesh:
	def __init__(self,N,dim,ords,dofloc='node',
			     rtype='uniform',rname=None,ghost_off=False):#,ords=[3,3]):
		self.N = N 
		self.h = 1/N
		self.dim = dim
		self.rtype = rtype
		self.rindex = refinement_index[rtype]
		rClass = refinement_class[rtype]
		self.refinement = rClass(rname,dofloc,N,dim,ords)
		self.dofloc = dofloc

		coarse_info = self.refinement.get_coarse_info()
		fine_info = self.refinement.get_fine_info()
		coarse_patch = Patch(N,dim,coarse_info,dofloc,ords,level=0)
		fine_patch = Patch(N,dim,fine_info,dofloc,ords,level=1,ghost_off=ghost_off)
		self.patches = [coarse_patch,fine_patch]

		self.dof_id_shift = len(coarse_patch.dofs)

	def loc_to_el(self,loc):
		loc_patch_id = self.refinement.get_patch_id(loc)
		shift = self.dof_id_shift if loc_patch_id else 0
		return self.patches[loc_patch_id]._get_element_from_loc(loc), shift

	
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

	def view(self):
		fig,ax = plt.subplots(2,1,figsize=(10,10))

		for level in range(2):
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
				el = self.patches[level].elements[id]
				ax[level].fill_between([el.x+eps,el.x+H-eps],
					   el.y+eps,el.y+H-eps,alpha=.5)

			for id in self.patches[level].dofs:
				dof = self.patches[level].dofs[id]
				ax[level].plot(dof.x,dof.y,'k.')

			c = 'lightgrey'
			if self.refinement.rshade[level] == 'all':
				ax[level].fill_between([0,1],0,1,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'in':
				ax[level].fill_between([.25,.75],.25,.75,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'out':
				ax[level].fill_between([0,1],0,.25,color=c,alpha=.5)
				ax[level].fill_between([0,1],.75,1,color=c,alpha=.5)
				ax[level].fill_between([0,.25],.25,.75,color=c,alpha=.5)
				ax[level].fill_between([.75,1],.25,.75,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'vstripe':
				ax[level].fill_between([.25,.75],0,1,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'hstripe':
				ax[level].fill_between([0,1],.25,.75,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'vedge':
				ax[level].fill_between([0,.25],0,1,color=c,alpha=.5)
				ax[level].fill_between([.75,1],0,1,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'hedge':
				ax[level].fill_between([0,1],0,.25,color=c,alpha=.5)
				ax[level].fill_between([0,1],.75,1,color=c,alpha=.5)




			ax[level].set_aspect('equal')

		plt.show()

	def view_detailed(self,large=True):
		if self.dim ==2:
			quad_bounds = [[0,.5,0,.5],
				[.5,1,0,.5],[0,.5,.5,1],[.5,1,.5,1]]
		if self.dim ==3:
			quad_bounds = [[0,.5,0,.5,0,.5],
				[.5,1,0,.5,0,.5],[0,.5,.5,1,0,.5],[.5,1,.5,1,0,.5],
				[0,.5,0,.5,.5,1],[.5,1,0,.5,.5,1],
				[0,.5,.5,1,.5,1],[.5,1,.5,1,.5,1]]
		fgsz = (20,10) if large else (6,3)
		fig,ax = plt.subplots(1,2,figsize=fgsz)

		for level in range(2):
			H = self.h/(1+level)
			dom = np.linspace(0,1,(1+level)*self.N+1)
			ext_dom = np.linspace(-2*H,1+2*H,(1+level)*self.N+5)
			eps = 0*H/10
			for x in dom:
				ax[level].plot([x,x],[0,1],'grey',lw=.7)
				ax[level].plot([0,1],[x,x],'grey',lw=.7)
			for x in ext_dom:
				ax[level].plot([x,x],[-2*H,1+2*H],'grey',ls=':',lw=.7)
				ax[level].plot([-2*H,1+2*H],[x,x],'grey',ls=':',lw=.7)

			for id in self.patches[level].elements:
				el = self.patches[level].elements[id]
				x0,x1,y0,y1 = el.bounds
				ax[level].plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0],'k',lw=.4)
				lens = np.array(el.bounds[1::2])-np.array(el.bounds[::2])
				for quad,q_bool in zip(quad_bounds,el.quads):
					if q_bool:
						quad_bound = []
						for ind in range(self.dim):
							diff = lens[ind]
							strt = el.bounds[2*ind]
							for shft in quad[2*ind:2*ind+2]:
								quad_bound.append(strt+shft*diff)
						x0,x1,y0,y1 = quad_bound
						ax[level].fill_between([x0+eps,x1-eps],
					   		y0+eps,y1-eps,alpha=.4)

			for id in self.patches[level].dofs:
				dof = self.patches[level].dofs[id]
				ax[level].plot(dof.x,dof.y,'k.',ms=2)

			c = 'lightgrey'
			if self.refinement.rshade[level] == 'all':
				ax[level].fill_between([0,1],0,1,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'in':
				ax[level].fill_between([.25,.75],.25,.75,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'out':
				ax[level].fill_between([0,1],0,.25,color=c,alpha=.5)
				ax[level].fill_between([0,1],.75,1,color=c,alpha=.5)
				ax[level].fill_between([0,.25],.25,.75,color=c,alpha=.5)
				ax[level].fill_between([.75,1],.25,.75,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'vstripe':
				ax[level].fill_between([.25,.75],0,1,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'hstripe':
				ax[level].fill_between([0,1],.25,.75,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'vedge':
				ax[level].fill_between([0,.25],0,1,color=c,alpha=.5)
				ax[level].fill_between([.75,1],0,1,color=c,alpha=.5)
			elif self.refinement.rshade[level] == 'hedge':
				ax[level].fill_between([0,1],0,.25,color=c,alpha=.5)
				ax[level].fill_between([0,1],.75,1,color=c,alpha=.5)

			ax[level].set_aspect('equal')

		plt.show()
	
	def vis_dof_sol(self,U,true_list=None,shrunk=None,log=True,split=True):
		
		if split:
			fig,axes = plt.subplots(1,2,figsize=(20,10))
		else:
			fig,ax = plt.subplots(figsize=(10,10))

		c_vals = []#{0:[],1:[]}
		locs = []#{0:[],1:[]}

		id_shift = len(self.patches[0].dofs)
		cbar_loc = ['left','right']

		for level in range(2):
			if split: 
				ax = axes[level]
				c_vals,locs = [],[]
			try:
				H = self.h/(1+level)
				dom = np.linspace(0,1,(1+level)*self.N+1)
				ext_dom = np.linspace(-2*H,1+2*H,(1+level)*self.N+5)
				eps = H/8
				for x in dom:
					ax.plot([x,x],[0,1],'grey',zorder=0)
					ax.plot([0,1],[x,x],'grey',zorder=0)
				for x in ext_dom:
					ax.plot([x,x],[-2*H,1+2*H],'grey',ls=':',zorder=0)
					ax.plot([-2*H,1+2*H],[x,x],'grey',ls=':',zorder=0)
				if self.rindex==2:
					ax.plot([.25,.75,.75,.25,.25],[.25,.25,.75,.75,.25],'k')

				for id in self.patches[level].dofs:
					dof = self.patches[level].dofs[id]
					global_id = dof.ID+id_shift*level
					if true_list is None or global_id in true_list:
						if true_list is None: u_val = U[global_id]
						else:
							true_index = true_list.index(global_id)
							u_val = U[true_index]
						c_vals.append(u_val)
						locs.append(dof.loc)


				tmp = np.array(locs).T
				if tmp.size > 0:
					x,y = tmp
				else:
					x,y = [],[]
				if split or level:
					if log:
						logvals = [np.log(cval) for cval in c_vals]
						plot = ax.scatter(x,y,c=logvals,cmap='jet',zorder=1)	
					else:
						plot = ax.scatter(x,y,c=c_vals,cmap='jet',zorder=1)	
					fig.colorbar(plot,location=cbar_loc[level])

					ax.set_aspect('equal')
				else:
					plot = ax.plot(x,y,'ko',ms=15,alpha=.5,fillstyle='none')

			except:
				print(uhoh)
				pass

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

		