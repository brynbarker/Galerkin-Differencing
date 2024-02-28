import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self,j,i):
        self.j = j
        self.i = i

class Element:
    def __init__(self,j,i):
        self.j = j
        self.i = i
        self.nodes = []
        self.stiffness = None

class Mesh:
    def __init(self,N,xlim=[0,1],ylim=[0,1]):
        self.N = N
        self.xlim = xlim
        self.ylim = ylim