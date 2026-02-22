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

class Mesh:
	def __init__(self,N,dim,ords,dofloc='node',rtype='uniform',rname=None):#,ords=[3,3]):
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
		coarse_patch = Patch(N,dim,coarse_info,dofloc,ords,level=0)#,ords=ords)
		fine_patch = Patch(N,dim,fine_info,dofloc,ords,level=1)#,ords=ords)
		self.patches = [coarse_patch,fine_patch]#{0:coarse_patch, 1:fine_patch}

		self.dof_id_shift = len(coarse_patch.dofs)

	def loc_to_el(self,loc):
		get_shift = [0,self.dof_id_shift]
		loc_patch_id = self.refinement.get_patch_id(loc)
		shift = self.dof_id_shift if loc_patch_id else 0
		return self.patches[loc_patch_id]._get_element_from_loc(loc), shift


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
	
	def vis_dof_sol(self,U,true_list=None):
		
		fig,ax = plt.subplots(1,2,figsize=(20,10))

		c_vals = {0:[],1:[]}
		locs = {0:[],1:[]}

		id_shift = len(self.patches[0].dofs)
		cbar_loc = ['left','right']

		for level in range(2):
			try:
				H = self.h/(1+level)
				dom = np.linspace(0,1,(1+level)*self.N+1)
				ext_dom = np.linspace(-2*H,1+2*H,(1+level)*self.N+5)
				eps = H/8
				for x in dom:
					ax[level].plot([x,x],[0,1],'grey',zorder=0)
					ax[level].plot([0,1],[x,x],'grey',zorder=0)
				for x in ext_dom:
					ax[level].plot([x,x],[-2*H,1+2*H],'grey',ls=':',zorder=0)
					ax[level].plot([-2*H,1+2*H],[x,x],'grey',ls=':',zorder=0)

				for id in self.patches[level].dofs:
					dof = self.patches[level].dofs[id]
					global_id = dof.ID+id_shift*level
					if true_list is None or global_id in true_list:
						c_vals[level].append(U[global_id])
						locs[level].append(dof.loc)

				x,y = np.array(locs[level]).T
				plot = ax[level].scatter(x,y,c=c_vals[level],cmap='jet',zorder=1)	
				fig.colorbar(plot,location=cbar_loc[level])

				ax[level].set_aspect('equal')
			except:
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

		