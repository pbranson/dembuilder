# -*- coding: utf-8 -*-
"""
Example script demonstrating functionality of dembuilder
Code copied from P Branson's jupyter notebook - basically just so MC doesn't have to keep flipping between 
google chrome and spyder

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

#define bounding box of new raster
bbox=np.zeros(4)
bbox[0] = 220000 #left
bbox[1] = 7570000 #bottom
bbox[2] = 270000 #right
bbox[3] = 7620000 #top
newRaster = db.Raster(bbox=bbox, resolution=500, epsgCode=28350)

#read bathy file and crop to bounding box
sampleReader = db.SamplePointReader('Pilbara_200m_Composite_Linear.tif',cropTo=bbox)
samples = sampleReader.load()

#disp samples (x,y,z, arrays)
samples

#show options for boundary types
[t for t in db.BoundaryPolygonType]

#use Box boundary
samples.generateBoundary(type=db.BoundaryPolygonType.Box)

#plot samples
samples.plot()

#define random samples to incorporate into bathy; could also load another bathy file (.tif, .xyz, .mat, .nc)
x = np.random.rand(100) * 10000 + 240000
y = np.random.rand(100) * 10000 + 7590000
z = np.random.rand(100) * 10 - 15

#create samples from x,y,z arrays defined above 
randomSamples = db.SamplePoints(x,y,z)
#show random samples
randomSamples.plot()

#generate a concave hull boundary and thresold distance to 1500 
randomSamples.generateBoundary(type=db.BoundaryPolygonType.ConcaveHull,threshold=1500)

#show resampling methods
[t for t in db.ResampleMethods]

#resample using linear interpolation
randomSamples.resample(newRaster,method=db.ResampleMethods.Linear)

#plot new raster with interpolated bathy (only has randomSamples)
newRaster.plot()

#include bathy from Pilbara_200m_composite
samples.resample(newRaster,method=db.ResampleMethods.BlockAvg)

#display composite bathy dataset
newRaster.plot()

#save raster to .tif
newRaster.saveToFile('tempRaster.tiff')
#export raster as .xyz
newRaster.getSamples().saveXYZ('tempRaster.xyz')

#can also load existing rasters 
loadedRaster = db.Raster.loadFromFile('tempRaster.tiff')
loadedRaster.plot()

