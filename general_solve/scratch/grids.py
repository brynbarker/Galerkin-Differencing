import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self,ID,j,i,x,y,h):
        self.ID = ID
        self.j = j
        self.i = i
        self.x = x
        self.y = y
        self.h = h
        self.elements = {}

    def add_element(self,e):
        if e.ID not in self.elements.keys():
            self.elements[e.ID] = e

class Element:
    def __init__(self,ID,j,i,x,y,h):
        self.ID = ID
        self.j = j
        self.i = i
        self.x = x
        self.y = y
        self.h = h
        self.dof_ids = []
        self.dof_list = []
        self.fine = False
        self.interface = False
        self.side = None
        self.dom = [x,x+h,y,y+h]
        self.plot = [[x,x+h,x+h,x,x],
                     [y,y,y+h,y+h,y]]

    def add_dofs(self,strt,xlen,sz):
        if len(self.dof_ids) != 0:
            return
        for ii in range(sz):
            for jj in range(sz):
                self.dof_ids.append(strt+xlen*ii+jj)
        return

    def update_dofs(self,dofs):
        if len(self.dof_list) != 0:
            return
        for dof_id in self.dof_ids:
            dof = dofs[dof_id]
            dof.add_element(self)
            self.dof_list.append(dof)
        return

    def set_fine(self):
        self.fine = True
    def set_interface(self,which,p):
        self.interface = True
        

        if p == 3:
            self.side = which
 
            if which == 0:
                self.dom[3] = self.y+self.h/2
                for ind in [2,3]:
                    self.plot[1][ind] -= self.h/2
            else:
                self.dom[2] = self.y+self.h/2
                for ind in [0,1,4]:
                    self.plot[1][ind] += self.h/2

class Mesh:
    def __init__(self,N,p):
        self.N = N # number of fine elements 
                   # from x=0.5 to x=1.0
        self.p = p
        self.h = 0.5/N
        self.dofs = {}
        self.elements = []
        self.boundaries = []
        self.periodic = {}
        self.periodic_ghost = []

        self.dof_count = 0
        self.el_count = 0
        
        self.n_els = []

        self.interface = {}

        c_specs, f_specs = self._set_specs()
        
        self._make_coarse(c_specs)
        self._make_fine(f_specs)

        self._update_elements()

    def _set_specs(self):
        raise ValueError('virtual needs to be overwritten')

    def _make_coarse(self,specs):
        H = self.h*2

        xdom, ydom = specs['xdom'],specs['ydom']
        xlen,ylen = len(xdom),len(ydom)

        dof_id,e_id = 0,0
        for i,y in enumerate(ydom):
            for j,x in enumerate(xdom):
                self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

                if (0<=x<.5) and (0<=y<1.):
                    strt = dof_id-specs['strt']
                    element = Element(e_id,j,i,x,y,H)
                    element.add_dofs(strt,xlen)
                    self.elements.append(element)
                    e_id += 1

                if x==specs['bnd']:
                    self.boundaries.append(dof_id)
                elif y < specs['ymin'] or y > specs['ymax']:
                    self.periodic[0].append(dof_id)
                
                if (x == specs['xinter']) and (0 <= y < 1):
                    self.interface[0].append(dof_id)

                if j in specs['int_inds'] and (0<=y<1):
                    offset_lab = specs['int_inds_map'][j]
                    self.interface_offset[offset_lab].append(dof_id)

                dof_id += 1

        self.n_coarse_dofs = dof_id
        self.n_coarse_els = e_id

    def _make_fine(self,specs):
        H = self.h

        xdom, ydom = specs['xdom'],specs['ydom']
        xlen,ylen = len(xdom),len(ydom)

        dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
        for i,y in enumerate(ydom):
            for j,x in enumerate(xdom):
                self.dofs[dof_id] = Node(dof_id,j,i,x,y,H)

                if (0.5<=x<1.) and (0<=y<1.):
                    strt = dof_id-specs['strt']
                    element = Element(e_id,j,i,x,y,H)
                    element.add_dofs(strt,xlen)
                    element.set_fine()
                    self.elements.append(element)
                    e_id += 1

                if x==specs['bnd']:
                    self.boundaries.append(dof_id)
                elif y < specs['ymin'] or y > specs['ymax']:
                    self.periodic[1].append(dof_id)
                if (x == specs['xinter']) and (0 <= y < 1):
                    self.interface[1].append(dof_id)
                if j in specs['int_inds'] and (0<=y<1):
                    offset_lab = specs['int_inds_map'][j]
                    self.interface_offset[offset_lab].append(dof_id)

                dof_id += 1

    def _update_elements(self):
        for e in self.elements:
            e.update_dofs(self.dofs)

class SideCenteredUGrid(Mesh):
    def __init__(self,N,p):
        super().__init__(N,p)

    def _set_specs(self):
        c_specs = {}
        f_specs = {}
        if self.p == 1:
            H = self.h*2
            xdom = np.linspace(0,0.5,int(self.N/2)+1)
            ydom = np.linspace(0-H/2,1+H/2,self.N+2)
            xlen,ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = 0
            c_specs['int_inds'] = []
            c_specs['int_inds_map'] = {}
            c_specs['ymin'] = H
            c_specs['ymax'] = 1-H
            c_specs['xinter'] = 0.5
            c_specs['bnd'] = 0.0

            H = self.h
            xdom = np.linspace(0.5,1,self.N+1)
            ydom = np.linspace(0-H/2,1+H/2,self.N*2+2)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = 0
            f_specs['int_inds'] = []
            f_specs['int_inds_map'] = {}
            f_specs['ymin'] = H
            f_specs['ymax'] = 1-H
            f_specs['xinter'] = 0.5
            f_specs['bnd'] = 1.0
        if self.p == 3:
            H = self.h*2
            xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
            ydom = np.linspace(0-3*H/2,1+3*H/2,self.N+4)
            xlen,ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = xlen+1
            c_specs['int_inds'] = [xlen-4,xlen-3,xlen-2,xlen-1]
            c_specs['int_inds_map'] = {xlen-4:0,xlen-3:1,xlen-2:2,xlen-1:3}
            c_specs['ymin'] = 2*H
            c_specs['ymax'] = 1-2*H
            c_specs['xinter'] = 0.5
            c_specs['bnd'] = 0.0

            H = self.h
            xdom = np.linspace(0.5-H,1+H,self.N+3)
            ydom = np.linspace(0-3*H/2,1+3*H/2,self.N*2+4)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = xlen+1
            f_specs['int_inds'] = [0,1,2,3]
            f_specs['int_inds_map'] = {0:4,1:5,2:6,3:7}
            f_specs['ymin'] = 2*H
            f_specs['ymax'] = 1-2*H
            f_specs['xinter'] = 0.5
            f_specs['bnd'] = 1.0
        return c_specs, f_specs

class SideCenteredVGrid(Mesh):
    def __init__(self,N,p):
        super().__init__(N,p)

    def _set_specs(self):
        c_specs = {}
        f_specs = {}
        if self.p == 1:
            H = self.h*2
            xdom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)
            ydom = np.linspace(0,1,self.N+1)
            xlen,ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = 0
            c_specs['int_inds'] = [xlen-2,xlen-1]
            c_specs['int_inds_map'] = {xlen-2:0,xlen-1:1}
            c_specs['ymin'] = H/2
            c_specs['ymax'] = 1-H/2
            c_specs['xinter'] = None
            c_specs['bnd'] = -H/2

            H = self.h
            xdom = np.linspace(0.5-H/2,1+H/2,self.N+2)
            ydom = np.linspace(0,1,self.N*2+1)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = 0
            f_specs['int_inds'] = [0,1]
            f_specs['int_inds_map'] = {0:2,1:3}
            f_specs['ymin'] = H/2
            f_specs['ymax'] = 1-H/2
            f_specs['xinter'] = None
            f_specs['bnd'] = 1+H/2
        if self.p == 3:
            H = self.h*2
            xdom = np.linspace(0-3*H/2,0.5+3*H/2,int(self.N/2)+4)
            ydom = np.linspace(0-H,1+H,self.N+3)
            xlen,ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = xlen+1
            c_specs['int_inds'] = [xlen-4,xlen-3,xlen-2,xlen-1]
            c_specs['int_inds_map'] = {xlen-4:0,xlen-3:1,xlen-2:2,xlen-1:3}
            c_specs['ymin'] = 3*H/2
            c_specs['ymax'] = 1-3*H/2
            c_specs['xinter'] = None
            c_specs['bnd'] = -H/2
            
            H = self.h
            xdom = np.linspace(0.5-3*H/2,1+3*H/2,self.N+4)
            ydom = np.linspace(0-H,1+H,self.N*2+3)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = xlen+1
            f_specs['int_inds'] = [0,1,2,3]
            f_specs['int_inds_map'] = {0:4,1:5,2:6,3:7}
            f_specs['ymin'] = 3*H/2
            f_specs['ymax'] = 1-3*H/2
            f_specs['xinter'] = None
            f_specs['bnd'] = 1+H/2
        return c_specs, f_specs

class CellCenteredGrid(Mesh):
    def __init__(self,N,p):
        super().__init__(N,p)

    def _set_specs(self):
        c_specs = {}
        f_specs = {}
        if self.p == 1:
            H = self.h*2
            xdom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)
            ydom = np.linspace(0-H/2,1+H/2,self.N+2)
            xlen,ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = 0
            c_specs['int_inds'] = [xlen-2,xlen-1]
            c_specs['int_inds_map'] = {xlen-2:0,xlen-1:1}
            c_specs['ymin'] = H
            c_specs['ymax'] = 1-H
            c_specs['xinter'] = None
            c_specs['bnd'] = -H/2

            H = self.h
            xdom = np.linspace(0.5-H/2,1+H/2,self.N+2)
            ydom = np.linspace(0-H/2,1+H/2,self.N*2+2)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = 0
            f_specs['int_inds'] = [0,1]
            f_specs['int_inds_map'] = {0:2,1:3}
            f_specs['ymin'] = H
            f_specs['ymax'] = 1-H
            f_specs['xinter'] = None
            f_specs['bnd'] = 1+H/2
        if self.p == 3:
            H = self.h*2
            xdom = np.linspace(0-3*H/2,0.5+3*H/2,int(self.N/2)+4)
            ydom = np.linspace(0-3*H/2,1+3*H/2,self.N+4)
            xlen,ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = xlen+1
            c_specs['int_inds'] = [xlen-4,xlen-3,xlen-2,xlen-1]
            c_specs['int_inds_map'] = {xlen-4:0,xlen-3:1,xlen-2:2,xlen-1:3}
            c_specs['ymin'] = 2*H
            c_specs['ymax'] = 1-2*H
            c_specs['xinter'] = None
            c_specs['bnd'] = -H/2
            
            H = self.h
            xdom = np.linspace(0.5-3*H/2,1+3*H/2,self.N+4)
            ydom = np.linspace(0-3*H/2,1+3*H/2,self.N*2+4)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = xlen+1
            f_specs['int_inds'] = [0,1,2,3]
            f_specs['int_inds_map'] = {0:4,1:5,2:6,3:7}
            f_specs['ymin'] = 2*H
            f_specs['ymax'] = 1-2*H
            f_specs['xinter'] = None
            f_specs['bnd'] = 1+H/2
        return c_specs, f_specs


class NodeCenteredGrid(Mesh):
    def __init__(self,N,p):
        super().__init__(N,p)

    def _set_specs(self):
        c_specs = {}
        f_specs = {}
        if self.p == 1:
            H = self.h*2
            xdom = np.linspace(0,0.5,int(self.N/2)+1)
            ydom = np.linspace(0,1,self.N+1)
            xlen,ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = 0
            c_specs['int_inds'] = []
            c_specs['int_inds_map'] = {}
            c_specs['ymin'] = H
            c_specs['ymax'] = 1-H
            c_specs['xinter'] = 0.5
            c_specs['bnd'] = 0.0

            H = self.h
            xdom = np.linspace(0.5,1.,self.N+1)
            ydom = np.linspace(0,1,2*self.N+1)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = 0
            f_specs['int_inds'] = []
            f_specs['int_inds_map'] = {}
            f_specs['ymin'] = H
            f_specs['ymax'] = 1-H
            f_specs['xinter'] = 0.5
            f_specs['bnd'] = 1.0
        if self.p == 3:
            H = self.h*2
            xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
            ydom = np.linspace(-H,1+H,self.N+3)
            xlen, ylen = len(xdom),len(ydom)
            c_specs['xdom'] = xdom
            c_specs['ydom'] = ydom
            c_specs['strt'] = xlen+1 
            c_specs['int_inds'] = [xlen-4,xlen-3,xlen-2,xlen-1]
            c_specs['int_inds_map'] = {xlen-4:0,xlen-3:1,xlen-2:2,xlen-1:3}
            c_specs['ymin'] = 2*H
            c_specs['ymax'] = 1-2*H
            c_specs['xinter'] = 0.5
            c_specs['bnd'] = 0.0
            
            H = self.h
            xdom = np.linspace(0.5-H,1.+H,self.N+3)
            ydom = np.linspace(-H,1+H,2*self.N+3)
            xlen,ylen = len(xdom),len(ydom)
            f_specs['xdom'] = xdom
            f_specs['ydom'] = ydom
            f_specs['strt'] = xlen+1
            f_specs['int_inds'] = [0,1,2,3]
            f_specs['int_inds_map'] = {0:4,1:5,2:6,3:7}
            f_specs['ymin'] = 2*H
            f_specs['ymax'] = 1-2*H
            f_specs['xinter'] = 0.5
            f_specs['bnd'] = 1.0
        return c_specs, f_specs