import numpy as np
import matplotlib.pyplot as plt
from paper_1.refinement import RefinementPattern
from paper_1.patch import Patch

refinement_type = {'uniform':0,
				   'finecenter':1,
				   'coarsecenter':2}

class Mesh:
	def __init__(self,N,dim,dofloc='node',rtype='uniform'):
		self.N = N 
		self.h = 1/N
		self.dim = dim
		self.refinement = RefinementPattern(rtype,dofloc,N,dim)
		self.rindex = refinement_type[rtype]
		self.dofloc = dofloc

		coarse_info = self.refinement.get_coarse_info()
		fine_info = self.refinement.get_fine_info()
		coarse_patch = Patch(N,dim,coarse_info,dofloc,level=0)
		fine_patch = Patch(N,dim,fine_info,dofloc,level=1)
		self.patches = [coarse_patch,fine_patch]#{0:coarse_patch, 1:fine_patch}

		self.dof_id_shift = len(coarse_patch.dofs)

	def loc_to_el(self,loc):
		get_shift = [0,self.dof_id_shift]
		if self.rindex == 0:
			return self.patches[0]._get_element_from_loc(loc), 0
		else:# self.rindex == 1:
			center = self.rindex % 1
			if sum([.25<=x<.75 for x in loc])==self.dim:
				return self.patches[center]._get_element_from_loc(loc),get_shift[center]
			else:
				return self.patches[1-center]._get_element_from_loc(loc),get_shift[1-center]


	def view(self):
		fig,ax = plt.subplots(2,1,figsize=(10,10))

		rindex_to_shade ={0:['none','all'],
				  		  1:['in','out'],2:['out','in']}
		rshade = rindex_to_shade[self.rindex]


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
			if rshade[level] == 'all':
				ax[level].fill_between([0,1],0,1,color=c,alpha=.5)
			elif rshade[level] == 'in':
				ax[level].fill_between([.25,.75],.25,.75,color=c,alpha=.5)
			elif rshade[level] == 'out':
				ax[level].fill_between([0,1],0,.25,color=c,alpha=.5)
				ax[level].fill_between([0,1],.75,1,color=c,alpha=.5)
				ax[level].fill_between([0,.25],.25,.75,color=c,alpha=.5)
				ax[level].fill_between([.75,1],.25,.75,color=c,alpha=.5)



			ax[level].set_aspect('equal')

		plt.show()

	def view_detailed(self):
		if self.dim ==2:
			quad_bounds = [[0,.5,0,.5],
				[.5,1,0,.5],[0,.5,.5,1],[.5,1,.5,1]]
		if self.dim ==3:
			quad_bounds = [[0,.5,0,.5,0,.5],
				[.5,1,0,.5,0,.5],[0,.5,.5,1,0,.5],[.5,1,.5,1,0,.5],
				[0,.5,0,.5,.5,1],[.5,1,0,.5,.5,1],
				[0,.5,.5,1,.5,1],[.5,1,.5,1,.5,1]]
		fig,ax = plt.subplots(2,1,figsize=(10,10))

		rindex_to_shade ={0:['none','all'],
				  		  1:['in','out'],2:['out','in']}
		rshade = rindex_to_shade[self.rindex]


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
			if rshade[level] == 'all':
				ax[level].fill_between([0,1],0,1,color=c,alpha=.5)
			elif rshade[level] == 'in':
				ax[level].fill_between([.25,.75],.25,.75,color=c,alpha=.5)
			elif rshade[level] == 'out':
				ax[level].fill_between([0,1],0,.25,color=c,alpha=.5)
				ax[level].fill_between([0,1],.75,1,color=c,alpha=.5)
				ax[level].fill_between([0,.25],.25,.75,color=c,alpha=.5)
				ax[level].fill_between([.75,1],.25,.75,color=c,alpha=.5)

			ax[level].set_aspect('equal')

		plt.show()
	
	def vis_dof_sol(self,U):
		
		fig,ax = plt.subplots(2,1,figsize=(10,10))

		c_vals = {0:[],1:[]}
		locs = {0:[],1:[]}

		id_shift = len(self.patches[0].dofs)
		cbar_loc = ['left','right']

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

			for id in self.patches[level].dofs:
				dof = self.patches[level].dofs[id]
				c_vals[level].append(U[dof.ID-id_shift*level])
				locs[level].append(dof.loc)

			x,y = np.array(locs[level]).T
			plot = ax[level].scatter(x,y,c=c_vals[level],cmap='jet')	
			fig.colorbar(plot,location=cbar_loc[level])

			ax[level].set_aspect('equal')

		plt.show()
