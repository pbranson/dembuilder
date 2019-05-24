# -*- coding: utf-8 -*-
"""
Create Pilbara bathy from 200m resolution bathy (from P Branson) 
and from 2018 coastal and island lidar

M Cuttler
May 2019

"""
#%% load modules

#import python modules
import sys
import numpy as np
from matplotlib import pyplot as plt

#import p branson modules
import dembuilder as db

sys.path.append('../')

#%% build base bathy from 200m-resolution Pilbara coastal bathy

bbox=np.zeros(4)

#limits of Pilbara_200m_Composite_Linear.tif
bbox[0] = 217000 #left
bbox[1] = 7550000 #bottom
bbox[2] = 291000 #right
bbox[3] = 7620000 #top

newRaster = db.Raster(bbox=bbox, resolution=200, epsgCode=28350)


sampleReader = db.SamplePointReader('Pilbara_200m_Composite_Linear.tif',cropTo=bbox)
samples = sampleReader.load()

#Before any loaded samples can be used to fill the raster, you must specify the type of boundary to use to limit extrapolation.
    #options for boundary type
    #[<BoundaryPolygonType.Box: 0>,
    # <BoundaryPolygonType.ConvexHull: 1>,
    # <BoundaryPolygonType.ConcaveHull: 2>]
    
#box boundary
samples.generateBoundary(type=db.BoundaryPolygonType.Box)

#interpolate bathy to raster
    #re-sampling methods
    #[<ResampleMethods.BlockAvg: 0>,
    # <ResampleMethods.Linear: 1>,
    # <ResampleMethods.Cubic: 2>,
    # <ResampleMethods.SmoothCubic: 3>,
    # <ResampleMethods.BsplineLSQ: 4>,
    # <ResampleMethods.BsplineSmooth: 5>,
    # <ResampleMethods.Rbf: 6>,
    # <ResampleMethods.Kriging: 7>,
    # <ResampleMethods.NaturalNeighbour: 8>]
    
samples.resample(newRaster,method=db.ResampleMethods.BlockAvg)

#show bathy
newRaster.plot()

#%% add island to bathy

#load files, and then resample to interpolate with previous bathy
    #re-sampling methods
    #[<ResampleMethods.BlockAvg: 0>,
    # <ResampleMethods.Linear: 1>,
    # <ResampleMethods.Cubic: 2>,
    # <ResampleMethods.SmoothCubic: 3>,
    # <ResampleMethods.BsplineLSQ: 4>,
    # <ResampleMethods.BsplineSmooth: 5>,
    # <ResampleMethods.Rbf: 6>,
    # <ResampleMethods.Kriging: 7>,
    # <ResampleMethods.NaturalNeighbour: 8>]
    
#if .xyz, need to be tab delimited
bathyfiles = ['Ashburton_5m.xyz', 'Eva-Y_5m.xyz', 'Fly_5m.xyz', 'Locker_5m.xyz','Observation_5m.xyz']
#bathyfiles = ['Ashburton_5m.xyz']
for bathy in bathyfiles:    
    newSampleReader = db.SamplePointReader(bathy)     
    newSamples = newSampleReader.load()    
    newSamples.generateBoundary(type=db.BoundaryPolygonType.ConcaveHull,threshold=250)
    #interpolate to raster ojbect
    newSamples.resample(newRaster,method=db.ResampleMethods.BlockAvg)     

newRaster.plot_island()

#%% export new raster

#need to add code to assign projection info to .tff
newRaster.saveToFile('PilbaraIslands.tif')

newRaster.getSamples().saveXYZ('PilbaraIslands.xyz')

