from fractions import Fraction
from corner_classes_2d import CornerRefineSolver
import numpy as np
import matplotlib.pyplot as plt

def extract_stencil(mat_in,ind,solver,scale=1,hanging_only=False,no_dup=False,all_stencils=None):
	mat = mat_in.copy()
	ind_full = solver.true_dofs[ind]
	og = solver.mesh.dofs[ind_full]

	stencil = []
	stencil.append([0,0,mat[ind,ind]*scale])
	for j,v in enumerate(mat[ind]):
		if abs(v)>1e-12 and j!=ind:
			j_full = solver.true_dofs[j]
			dof = solver.mesh.dofs[j_full]
			if abs(dof.x-og.x)>.5 or abs(dof.y-og.y)>.5:
				return None
			stencil.append([(dof.x-og.x)/solver.h,(dof.y-og.y)/solver.h,v*scale])
	stencil = np.array(stencil)

	if abs(np.sum(stencil[:,-1])) > 1e-12:
		return None

	check_hanging = np.allclose(stencil[1:,-1],-scale/3)
	if hanging_only and check_hanging:
		return None
	
	scale_mat = np.zeros_like(stencil)
	scale_mat[:,:-1] = 1/scale
	scale_mat[:,-1] = solver.h

	if no_dup:
		for sten in all_stencils.values():
			if np.allclose(sten[0].shape,stencil.shape) and np.allclose(sten[0][:,:-1],stencil[:,:-1]):
				return None
				
	return np.array(stencil), scale_mat, check_hanging, og.h==solver.h, ((.5-og.x)/solver.h,(.5-og.y)/solver.h)

def load_2d_stencils():
	u = lambda x,y: 5
	f = lambda x,y: 0
	ref = CornerRefineSolver(8,u,f)
	ref.laplace()
	CTKC = ref.C.T@ref.K@ref.C

	mystencils = {}
	stencil_count = 0
	found_basic = False

	for ind in range(len(ref.true_dofs)):
		output = extract_stencil(-CTKC,ind,ref,scale=3,hanging_only=found_basic,no_dup=True,all_stencils=mystencils)
		if output is not None:
			keep = True
			stencil, scales,basic,level,center = output
			if not found_basic and basic:
				found_basic = True
			for sten in mystencils.values():
				if np.allclose(sten[0].shape,stencil.shape) and np.allclose(sten[0][:,:-1],stencil[:,:-1]):
					keep = False
					break
			if keep:
				mystencils[stencil_count] = (stencil,scales,level,center)
				stencil_count += 1
	return mystencils

def disp_stencil(stencil,fname=None):
	fig,ax = plt.subplots(1,figsize=(7,20))

	data,scales,level,center = stencil
	minx,maxx = np.min(data[:,0]),np.max(data[:,0])
	miny,maxy = np.min(data[:,1]),np.max(data[:,1])

	ymod,xmod = 0,0
	if level:
		top = 1 in (data[:,1]-.5)%2
		ymod = 1 if top else 0
		xmod = 2 not in abs(data[:,0])

	for x_shift in np.arange(int(minx-.5),int(maxx+1.5)):
		c = ['lightgrey' if x_shift%2 else 'k', 'lightgrey' if (x_shift+xmod)%2 else 'k']
		plt.plot([x_shift,x_shift],[miny-3/4,maxy+3/4],c[level])

	for y_shift in np.arange(int(miny-1.5),int(maxy+1.5)):
		c = ['k' if y_shift%2 else 'lightgrey','k' if (y_shift-ymod)%2 else 'lightgrey']
		offset = .5 if level else 0
		plt.plot([minx-.5,maxx+.5],[y_shift+offset,y_shift+offset],c[level])

	plt.plot([center[0],center[0],maxx+1],[maxy+1,center[1],center[1]],lw=5,color='k')
	plt.title(center)
	assert scales[0,0]==1/3

	for j,vals in enumerate(data):
		x,y,v = vals
		c = 'r' if j==0 else 'b'
		ax.plot(x,y,marker='o',color=c)
		ax.annotate(str(Fraction(round(v,6))*Fraction(1,3)),(x,y),(x+1/16,y+1/16))
	ax.set_aspect('equal', adjustable='box')

	plt.xlim(minx-1/2,maxx+1/2)
	plt.ylim(miny-3/4,maxy+3/4)
	if fname is not None:
		plt.savefig(fname,dpi=300)
	plt.show()


