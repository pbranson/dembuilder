# -*- coding: utf-8 -*-
"""
    Kriging interpolation library
"""
from scipy import spatial
import numpy as np
import os
import multiprocessing



import pdb

class KrigingOptions(object):
    def __init__(self,varmodel,sill,nugget,vrange):
        self.varmodel = varmodel
        self.sill = sill
        self.nugget = nugget
        self.vrange = vrange

class WeightInputs(object):
    def __init__(self, dist, xin, yin, krigOptions):
        self.dist = dist
        self.xin = xin
        self.yin = yin
        self.krigOptions = krigOptions

class kriging(object):
    
    """ Class for kriging interpolation"""
    
    ### Properties ###
    maxdist = 1000
    NNear = 12
    
    # Variogram paramters
    varmodel = 'spherical'
    nugget = 0.1
    sill = 0.8
    vrange = 250.0

    
    verbose = True
    
    def __init__(self,XYin,XYout,rescale=False,**kwargs):
        self.__dict__.update(kwargs)
        if rescale:
            XYin = np.array(XYin)
            XYout = np.array(XYout)
            self.scaleMin = np.min(XYin, axis=0)
            XYin[:, 0] = XYin[:, 0] - self.scaleMin[0]
            XYin[:, 1] = XYin[:, 1] - self.scaleMin[1]
            XYout[:, 0] = XYout[:, 0] - self.scaleMin[0]
            XYout[:, 1] = XYout[:, 1] - self.scaleMin[1]
            self.scale = np.max(XYin, axis=0)
            XYin[:, 0] = XYin[:, 0] / self.scale[0]
            XYin[:, 1] = XYin[:, 1] / self.scale[1]
            XYout[:, 0] = XYout[:, 0] / self.scale[0]
            XYout[:, 1] = XYout[:, 1] / self.scale[1]
        self.XYin = XYin #zip(XYin[:,0],XYin[:,1])
        self.XYout = XYout #zip(XYout[:,0],XYout[:,1])
        self.krigOptions =  KrigingOptions(self.varmodel, self.nugget, self.sill, self.vrange)
        self._buildWeights()
        
    def __call__(self,Zin):
        """
        Calls the interpolation function with the scalar in Zin
        """
        self.Z = np.zeros((self.Nc,))
        for ii in range(0,self.Nc):
            self.Z[ii] = np.dot(self.W[:,ii],Zin[self.ind[ii,:]])
            
        return self.Z
                
    def _buildWeights(self):
        """ Calculates the kriging weights for all of the points in the grid"""
        # Compute the spatial tree
        kd = spatial.cKDTree(self.XYin)
        
        # Perform query on all of the points in the grid
        dist, self.ind=kd.query(self.XYout,
                distance_upper_bound=self.maxdist,
                k=self.NNear)
        
        self.Nc = np.size(self.ind,axis=0)
        print('%d interpolation points.'%self.Nc)
        # Now loop through and get the weights for each point
        self.W = np.zeros((self.NNear,self.Nc))

        # Print percentages
        p0=0
        pstep=5
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        
        wInputs=[]
        for ii in range(0,self.Nc):
        
            wInputs.append(WeightInputs(dist[ii,:],self.XYin[self.ind[ii,:],0],self.XYin[self.ind[ii,:],1],self.krigOptions))
            
            
            # if self.verbose:
                # pfinish = float(ii)/float(self.Nc)*100.0
                # if  pfinish> p0:
                    # print '%3.1f %% complete...'%pfinish
                    # p0+=pstep
                                
            # W = self.getWeights(dist[ii,:],\
                # self.XYin[self.ind[ii,:],0],\
                # self.XYin[self.ind[ii,:],1])
            
        self.W=pool.map(getWeights,wInputs)
        self.W=np.squeeze(self.W).transpose()
        # print(self.W.shape)
            # self.W[:,ii] = W.T 


        
def getWeights(weightInputs):

    krigOptions=weightInputs.krigOptions
    dist=weightInputs.dist
    xin=weightInputs.xin
    yin=weightInputs.yin

    """ Calculates the kriging weights point by point"""

    eps = 1e-10
    Ns = len(dist)

    # Construct the LHS matrix C
    C=np.ones((Ns+1,Ns+1))
    for i in range(0,Ns):
        C[i,i]=0
        for j in range(i+1,Ns):
            D = np.sqrt((xin[i]-xin[j])**2+(yin[i]-yin[j])**2)
            C[i,j] = semivariogram(krigOptions,D+eps)
            C[j,i] = C[i,j]

    C[Ns,Ns]=0

    ###
    # Old method
    ###
    # Calculate the inverse of C
    #Cinv = np.linalg.inv(C)

    # Loop through each model point and calculate the vector D
    gamma = np.ones((Ns+1,1))

    for j in range(0,Ns):
        gamma[j,0]= semivariogram(krigOptions,dist[j]+eps)

    # Solve the matrix to get the weights
    #W = np.dot(Cinv,gamma)
    #W = W[:-1,:]

    W = np.linalg.solve(C,gamma)

    #print np.size(gamma,axis=0),np.size(gamma,axis=1)
    return W[:-1]
    #
        #return 1.0/float(Ns)*np.ones((Ns,1))
        
def semivariogram(krigOptions,D):
    """ Semivariogram functions"""
    if krigOptions.varmodel == 'spherical':
        if D > krigOptions.vrange:
            F = krigOptions.sill
        else:
            tmp = D/krigOptions.vrange
            F = krigOptions.nugget + (krigOptions.sill-krigOptions.nugget)*(1.5*tmp - 0.5*tmp**3)
    return F


        
