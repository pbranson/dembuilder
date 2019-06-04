# -*- coding: utf-8 -*-
"""
Create bathy for Perth Metro wave model developed by BoM
Data sources include: 
    2016 Coastal LiDAR
    2012 GA Multibeam
    2009 Coastal LiDar    
    2009 GA Topo/bathy
    
M Cuttler
June 2019

"""
#%% load modules

#import python modules
import sys
import numpy as np
from matplotlib import pyplot as plt

#import p branson modules
import dembuilder as db

sys.path.append('../')

#%% build bathy from individual island lidar and from 200m-resolution Pilbara coastal bathy

bbox=np.zeros(4)

#limits of Pilbara_200m_Composite_Linear.tif
bbox[0] = 212463 #left
bbox[1] = 6183152 #bottom
bbox[2] = 398218 #right
bbox[3] = 6569213 #top

newRaster = db.Raster(bbox=bbox, resolution=200, epsgCode=28350)

#code treats first bathy added as highest priority, so add islands first

#if .xyz, need to be tab delimited
filepath = 'P:\HANSEN_UWA-UNSW_Linkage\Analysis\Bathy\GriddingForBOM\GlobalMapper_analysis\For_dembuilder'
bathyfiles = ['lidar2016clip2_10m.xyz', 'lidar2009.xyz', 'ga2012clip2.xyz', 'ga2009_clip3.xyz']

for bathy in bathyfiles:    
    newSampleReader = db.SamplePointReader(os.path.join(filepath, bathy))     
    newSamples = newSampleReader.load()    
    newSamples.generateBoundary(type=db.BoundaryPolygonType.ConcaveHull,threshold=250)
    #interpolate to raster ojbect
    print('Interpolating samples to raster...')
    newSamples.resample(newRaster,method=db.ResampleMethods.BlockAvg)     
    
#show bathy
newRaster.plot()


#%% export new raster

#need to add code to assign projection info to .tff
newRaster.saveToFile('PilbaraIslands.tif')

newRaster.getSamples().saveXYZ('PilbaraIslands.xyz')

