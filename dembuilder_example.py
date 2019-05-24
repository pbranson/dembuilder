# -*- coding: utf-8 -*-
"""
Testing dembuilder for Pilbara Islands bathymetry

M Cuttler
May 2019

"""
#%% Example from P Branson notebook

#import python modules
import sys
import numpy as np

#import p branson modules
import dembuilder as db

sys.path.append('../')

bbox=np.zeros(4)
bbox[0] = 220000 #left
bbox[1] = 7570000 #bottom
bbox[2] = 270000 #right
bbox[3] = 7620000 #top
newRaster = db.Raster(bbox=bbox, resolution=500, epsgCode=28350)


sampleReader = db.SamplePointReader('Pilbara_200m_Composite_Linear.tif',cropTo=bbox)
samples = sampleReader.load()

samples

[t for t in db.BoundaryPolygonType]


samples.generateBoundary(type=db.BoundaryPolygonType.Box)


samples.plot()

x = np.random.rand(100) * 10000 + 240000
y = np.random.rand(100) * 10000 + 7590000
z = np.random.rand(100) * 10 - 15 
randomSamples = db.SamplePoints(x,y,z)
randomSamples.plot()

randomSamples.generateBoundary(type=db.BoundaryPolygonType.ConcaveHull,threshold=1500)

[t for t in db.ResampleMethods]


randomSamples.resample(newRaster,method=db.ResampleMethods.Linear)


newRaster.plot()
samples.resample(newRaster,method=db.ResampleMethods.BlockAvg)
newRaster.plot()
newRaster.saveToFile('tempRaster.tiff')
newRaster.getSamples().saveXYZ('tempRaster.xyz')
loadedRaster = db.Raster.loadFromFile('tempRaster.tiff')
loadedRaster.plot()

