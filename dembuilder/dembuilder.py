# -*- coding: utf-8 -*-

"""
    Interpolate sounding data onto a regular grid
"""

from scipy.interpolate import griddata
import numpy as np

from dembuilder.kriging import kriging

from shapely.ops import cascaded_union, polygonize, unary_union
from scipy.spatial import Delaunay
from scipy.interpolate import Rbf
from scipy.interpolate import LSQBivariateSpline
from scipy.interpolate import SmoothBivariateSpline
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from matplotlib import path
# from matplotlib.mlab import griddata
from pyproj import Proj, transform
import math
import shapely.geometry as geometry
import scipy.stats as stats
import gdal, osr
import os      
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
import pylab as pl
# from mpl_toolkits.natgrid import _natgrid


#Enumeration for valid file formats of sample data
class SamplePointFormat(Enum):
    Unknown, Netcdf, Xyz, Hdf, Matlab, GeoTIFF = range(6)

#Factory Class to read sample data and produce SampleData objects    
class SamplePointReader(object):
    
    def __init__(self,filename,coordConvert=False,sourceProj='+init=epsg:4326',targetProj='+init=epsg:28350',format=SamplePointFormat.Unknown,approxSpatialResolution=0,cropTo=None):

        self.coordConvert = coordConvert
        self.sourceProj = sourceProj
        self.targetProj = targetProj
        self.cropTo = cropTo

        if os.path.isfile(filename):
            self.filename = filename
        else:
            raise IOError('File not found %s' % filename)
        
        if type(format) != SamplePointFormat:
            raise ValueError('Invalid input for samplePoint format')
        else:
            self.format = format
            
        print(filename)
        
        if (format == SamplePointFormat.Unknown):
            splitName, splitExt = os.path.splitext(self.filename)
            splitExt=splitExt[1:]
            if (splitExt.lower() in "mat".split()):
                self.format = SamplePointFormat.Matlab
            elif (splitExt.lower() in "hdf hdf5 h5".split()):
                self.format = SamplePointFormat.Hdf
            elif (splitExt.lower() in "xyz csv txt".split()):
                self.format = SamplePointFormat.Xyz
            elif (splitExt.lower() in "nc".split()):
                self.format = SamplePointFormat.Netcdf
            elif (splitExt.lower() in "tif".split()):
                self.format = SamplePointFormat.GeoTIFF
            else:
                print(splitExt)
                raise ValueError('Unknown input file format and no format argument not specified')
                
        self.approxSpatialResolution = approxSpatialResolution
    
    def load(self,variableName=None,headerLines=1,delimiter=None,delim_whitespace=True):
        print(self.format)
        if self.format == SamplePointFormat.Matlab:
            print('Loading matlab file')
            x, y, z = self._loadDataMatlab(variableName)
        elif self.format == SamplePointFormat.Hdf:
            x, y, z = self._loadDataHdf(variableName)
        elif self.format == SamplePointFormat.Xyz:
            x, y, z = self._loadDataXyz(headerLines=headerLines,delimiter=delimiter,delim_whitespace=delim_whitespace)
        elif self.format == SamplePointFormat.Netcdf:
            x, y, z = self._loadDataNetcdf(variableName)
        elif self.format == SamplePointFormat.GeoTIFF:
            x, y ,z = self._loadDataGeoTIFF()

        if self.coordConvert:

            inProj = Proj(self.sourceProj)
            outProj = Proj(self.targetProj)
            x, y = transform(inProj, outProj, x, y)

            #return SamplePoints(x2, y2, z)

        if self.cropTo is None:
            return SamplePoints(x, y, z)
        else:
            inds = (x > self.cropTo[0]) & (y > self.cropTo[1]) & (x < self.cropTo[2]) & (y < self.cropTo[3])
            return SamplePoints(x[np.where(inds)], y[np.where(inds)], z[np.where(inds)])




    def _loadDataMatlab(self,variableName):
        from scipy import io
        self.rawData = io.loadmat(self.filename)
        
        x=self.rawData[variableName][:,0]
        y=self.rawData[variableName][:,1]
        z=self.rawData[variableName][:,2]
        
        return x,y,z
        
    def _loadDataXyz(self,headerLines, delimiter=None, delim_whitespace=True):
        import pandas as pd
        
        self.rawData = pd.read_csv(self.filename, delim_whitespace=delim_whitespace, header=headerLines, names=['X','Y','Z'])
        
        x=self.rawData['X'].values
        y=self.rawData['Y'].values
        z=self.rawData['Z'].values
        
        return x,y,z

    def _loadDataNetcdf(self,variableName,convertLL=True):
        import xarray as xr

        ds = xr.open_dataset(self.filename)

        #self.rawData = pd.read_csv(self.filename, delim_whitespace=True, header=1, names=['X', 'Y', 'Z'])
	
        # z = ds.z.data.ravel()
        z = ds[variableName].data.ravel()
        x, y = np.meshgrid(ds.lon.data, ds.lat.data)
        x = x.ravel()
        y = y.ravel()

        return x, y, z

    def _loadDataGeoTIFF(self):
        ds=gdal.Open(self.filename)
        geoTrans = ds.GetGeoTransform()
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = ds.GetGeoTransform()

        band = ds.GetRasterBand(1)
        elevation = band.ReadAsArray().astype(np.float)
        z = elevation.ravel()
        noDataVal = band.GetNoDataValue()

        x_coords = x_size * np.arange(0,band.XSize) + upper_left_x + (x_size / 2)  # add half the cell size
        y_coords = y_size * np.arange(0,band.YSize) + upper_left_y + (y_size / 2)  # to centre the point

        x, y = np.meshgrid(x_coords, y_coords)
        x = x.ravel()
        y = y.ravel()

        if noDataVal is None:
            mask = np.isnan(z)
        else:
            mask = (z == noDataVal) | (z == 0.0)

        x = np.delete(x, np.where(mask))
        y = np.delete(y, np.where(mask))
        z = np.delete(z, np.where(mask))

        rasterSRS = osr.SpatialReference()
        rasterSRS.ImportFromWkt(ds.GetProjectionRef())
        self.sourceProj = rasterSRS.ExportToProj4()
        print('Source projection: ' + self.sourceProj)

        return x, y, z

    def save(self,filename):
        np.savez_compressed(filename,self.x,self.y,self.z)

class BoundaryPolygonType(Enum):
    Box, ConvexHull, ConcaveHull = range(3)    
    
class ResampleMethods(Enum):
    BlockAvg, Linear, Cubic, SmoothCubic, BsplineLSQ, BsplineSmooth, Rbf, Kriging, NaturalNeighbour = range(9)

# class for sample point data         
class SamplePoints(object):
   
    def __init__(self,x,y,z):
        self.x = x 
        self.y = y 
        self.z = z 
        
    def __len__(self):
        return len(self.x)
    
    def __repr__(self):
#         return f'Total samples: {len} \n x={x} \n y={y} \n z={z}'.format(len=str(len(self)),x=self.x.__repr__,y=print(self.y),z=print(self.z))
        return 'Total samples: {l:d} \n x={x} \n y={y} \n z={z}'.format(l=len(self),x=self.x.__repr__(),y=self.y.__repr__(),z=self.z.__repr__())
    
    def getBoundingBox(self):
        self.boundingBox = getattr(self,'boundingBox',np.array([min(self.x),min(self.y),max(self.x),max(self.y)]))
        return self.boundingBox
        
    def generateBoundary(self,type,threshold=250):
        if (type == BoundaryPolygonType.Box):
            bbox=self.getBoundingBox()
            # x=[bbox[0] bbox[0] bbox[2] bbox[2]]
            # y=[bbox[1] bbox[3] bbox[3] bbox[1]]
            # self.boundary = geometry.Polygon(zip(x,y))
            self.boundary = geometry.box(bbox[0],bbox[1],bbox[2],bbox[3])
        
        else:
            self.points = geometry.MultiPoint(list(zip(self.x,self.y)))
            if (type == BoundaryPolygonType.ConvexHull):
                self.boundary = self.points.convex_hull
            if (type == BoundaryPolygonType.ConcaveHull):
                self.boundary, self.triangulation = alpha_shape(self.points,threshold)
        self.boundaryType = type
    
    def triangulate(self):
        self.triangulation = Delaunay(list(zip(self.x,self.y)))
    
    def averageSpacing(self):
        tri=getattr(self,'triangulation',self.triangulate())
        circum_radii = []
        coords=tri.points
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]
            # Lengths of sides of triangle
            a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
            c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
            # Semiperimeter of triangle
            s = (a + b + c)/2.0
            # Area of triangle by Heron's formula
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)
            circum_radii.append(circum_r)

        circum_radii=np.array(circum_radii)
        return np.mean(circum_radii)

    def crop(self,box):
        inds = (self.x > box[0]) & (self.y > box[1]) & (self.x < box[2]) & (self.y < box[3])
        self.x = np.delete(self.x, np.where(~inds))
        self.y = np.delete(self.y, np.where(~inds))
        self.z = np.delete(self.z, np.where(~inds))

    # def cropLevel(self,zmin=None,zmax=None):
    #     if zmin not None:
    #         inds = (self.z > zmin)

    def remove(self, inds):
        self.x = np.delete(self.x, np.where(inds))
        self.y = np.delete(self.y, np.where(inds))
        self.z = np.delete(self.z, np.where(inds))

    def appendSamples(self,x,y,z):
        self.x = np.concatenate((self.x,x),axis=0)
        self.y = np.concatenate((self.y,y),axis=0)
        self.z = np.concatenate((self.z,z),axis=0)
        
    def resample(self,raster,method,maxDist=1500,statistic='mean'):
        #if we are just block averaging, use myBinnedSamples to add to the raster where it was nan
        if (method == ResampleMethods.BlockAvg):
            myBinnedSamples, yedges, xedges, myBinNums = stats.binned_statistic_2d(self.y, self.x, self.z,
                                                                                   statistic=statistic,
                                                                                   bins=[raster.yBinEdges,
                                                                                         raster.xBinEdges],
                                                                                   expand_binnumbers=True)
            raster.z = np.where(np.isnan(raster.z),myBinnedSamples,raster.z)
        #otherwise things are a little more complicated!
        else:
            #determine which points in this Samples should be excluded as already covered in the raster
            myBinnedSamples, xedges, yedges, myBinNums = stats.binned_statistic_2d(self.x, self.y, self.z,
                                                                                   statistic='mean',
                                                                                   bins=[raster.xBinEdges,
                                                                                         raster.yBinEdges],
                                                                                   expand_binnumbers=True)
            
            # swap x/y to be y dimension 0 x dimension 1 so consistent with plotting convention and meshgrid
            # annoyingly the myBinNums represents the index in the edge array not the output myBinnedSamples array,
            # with out of bounds data in the outer cells of the edge array
            # clip this off, re-index and set out of bounds data to a binnumber of -1
            myBinNums = myBinNums - 1
            x_inds = myBinNums[0, :]
            y_inds = myBinNums[1, :]
            internalBins = np.where(
                (x_inds >= 0) & (y_inds >= 0) & (x_inds < len(xedges) - 1) & (y_inds < len(yedges) - 1))
            internalBinNums = -1 * np.ones(self.z.shape, dtype='int64')
            # re-ravel the 2D index, however need to use column-major order for consistency in the layout of raster.z
            internalBinNums[internalBins] = np.ravel_multi_index((x_inds[internalBins], y_inds[internalBins]),
                                                           (len(xedges)-1, len(yedges)-1),order='F')
            # internalBinNums=np.delete(internalBinNums,np.where(internalBinNums==-1))

            rasterBinNums = np.where(~np.isnan(raster.z.ravel()))[0]
            requiredSampleBins = np.setdiff1d(internalBinNums,rasterBinNums)
            requiredSamplesIndexes = np.where(np.in1d(internalBinNums,requiredSampleBins))
            requiredSamplesX = self.x[requiredSamplesIndexes]
            requiredSamplesY = self.y[requiredSamplesIndexes]
            requiredSamplesZ = self.z[requiredSamplesIndexes]
            print('Using %i SamplePoints to fill %i raster points with in the SamplePoints boundary' % (np.size(requiredSamplesX),np.size(requiredSampleBins)))
            #now that we have the 'new' samples provided by this Sample object (excluding those already covered by the raster)
            #bring in the rasters samples so that edges between data are kept consistent with smooth/linear transitions and included in the kriging/interpolation process
            combinedSampleSet = raster.getSamples()
            combinedSampleSet.appendSamples(requiredSamplesX,requiredSamplesY,requiredSamplesZ)

            # import pylab as pl
            # fig = pl.figure(figsize=(10, 10))
            # ax = fig.add_subplot(111)
            # margin = 250
            # x_min, y_min, x_max, y_max = raster.bbox
            # ax.set_xlim([x_min - margin, x_max + margin])
            # ax.set_ylim([y_min - margin, y_max + margin])
            # #pl.pcolor(raster.x-raster.resolution/2, raster.y-raster.resolution/2, raster.z, vmin=-20, vmax=0)
            # pl.pcolor(xedges,yedges,myBinnedSamples.transpose(),vmin=-20,vmax=0)
            # pl.scatter(requiredSamplesX, requiredSamplesY, 80, requiredSamplesZ, 's', vmin=-20, vmax=0)
            # pl.scatter(combinedSampleSet.x, combinedSampleSet.y, 80, combinedSampleSet.z, '.', vmin=-20, vmax=0)
            # pl.show()
            
            #Work out what points in the raster fall within the boundary of this Samples object and should be filled by the objects Samples
            try:
                boundaryType = self.boundaryType
                x=raster.x.ravel()
                y=raster.y.ravel()
                z=raster.z.copy().ravel()
                bbox=self.getBoundingBox()
                if (self.boundaryType == BoundaryPolygonType.Box):
                    inds = (x > bbox[0]) & (y > bbox[1]) & (x < bbox[2]) & (y < bbox[3]) & (np.isnan(z))
                elif (self.boundary.type == 'Polygon'):
                    # matplotlib path has function for array of points, shapely Polygon only seems to work on individual points...
                    boundaryPath = path.Path(np.stack(self.boundary.exterior.xy).transpose())
                    inds = (x > bbox[0]) & (y > bbox[1]) & (x < bbox[2]) & (y < bbox[3]) & boundaryPath.contains_points(np.stack((x,y),axis=0).transpose())
                elif (self.boundary.type == 'MultiPolygon'):
                    inds = np.empty(x.shape[0])
                    inds[:] = False
                    for polygon in self.boundary:
                        boundaryPath = path.Path(np.stack(polygon.boundary.exterior.xy).transpose())
                        inds = inds | boundaryPath.contains_points(np.stack((x,y),axis=0).transpose())
                    #inds=np.where(inds)
                rasterSamplePointsX = x[np.where(inds)]
                rasterSamplePointsY = y[np.where(inds)]
                #myBoundary=getattr(self,'boundary',geometry.MultiPoint(zip(self.x,self.y)))
                #self.boundary=getattr(self,'boundary',geometry.MultiPoint(zip(self.x,self.y)))
            except AttributeError:
                raise AttributeError("Boundary not identified, run getBoundary first and specify a type.")
            
            #Finally do the resampling with the specified method
            if (method == ResampleMethods.Linear):
                Z = griddata(list(zip(combinedSampleSet.x, combinedSampleSet.y)), combinedSampleSet.z,
                             list(zip(rasterSamplePointsX, rasterSamplePointsY)), method='linear')
            elif (method == ResampleMethods.Cubic):
                Z = griddata(list(zip(combinedSampleSet.x, combinedSampleSet.y)), combinedSampleSet.z,
                             list(zip(rasterSamplePointsX, rasterSamplePointsY)), method='cubic')
            elif (method == ResampleMethods.SmoothCubic):
                F = CloughTocher2DInterpolator(list(zip(combinedSampleSet.x, combinedSampleSet.y)), combinedSampleSet.z,
                             rescale=True)
                Z = F(rasterSamplePointsX, rasterSamplePointsY)
            elif (method == ResampleMethods.Kriging):
                points = np.stack((combinedSampleSet.x,combinedSampleSet.y),axis=1)
                grd = np.stack((rasterSamplePointsX,rasterSamplePointsY)).transpose()
                self.Finterp = kriging(points,grd,rescale=True,maxdist=maxDist,NNear=7)
                Z = self.Finterp(combinedSampleSet.z)
            elif (method == ResampleMethods.BsplineLSQ):
                F = LSQBivariateSpline(combinedSampleSet.x,combinedSampleSet.y,combinedSampleSet.z,raster.xBinCentres,raster.yBinCentres)
                Z = F(rasterSamplePointsX,rasterSamplePointsY,grid=False)
            elif (method == ResampleMethods.BsplineSmooth):
                F = SmoothBivariateSpline(combinedSampleSet.x, combinedSampleSet.y, combinedSampleSet.z)
                Z = F(rasterSamplePointsX, rasterSamplePointsY,grid=False)
            elif (method == ResampleMethods.NaturalNeighbour):
                print('** THis doesn''t work for me (RON) at the moment - error "undefined symbol: _intel_fast_memcpy"')
		        #_natgrid.seti(b'ext', 0)
                #_natgrid.setr(b'nul', np.nan)
                #zi = np.empty((raster.xBinCentres.shape[0],raster.yBinCentres.shape[0]), np.float64)
                #xp = np.require(combinedSampleSet.x, requirements=['C'])
                #yp = np.require(combinedSampleSet.y, requirements=['C'])
                #zp = np.require(combinedSampleSet.z, requirements=['C'])
                #xi = np.require(raster.xBinCentres, requirements=['C'])
                #yi = np.require(raster.yBinCentres, requirements=['C'])
                #_natgrid.natgridd(xp, yp, zp, xi, yi, zi)

                #Z = zi[inds.reshape(zi.shape)]
            elif (method == ResampleMethods.Rbf):                
                coords=zip(self.x,self.y)
                tri = Delaunay(coords)
                rbfi = Rbf(self.x,self.y,self.z)
                Z = rbfi(x,y)
        
            z[np.where(inds)]=Z
            raster.z = np.where(np.isnan(raster.z), np.reshape(z,raster.z.shape), raster.z)

        #return grid

    def save(self, filename):
        np.savez_compressed(filename, self.x, self.y, self.z)

    def saveXYZ(self, filename):
        output = np.stack((self.x,self.y,self.z), axis=1)
        np.savetxt(filename, output, delimiter="   ")

    def plot(self,vmin=-20,vmax=0,margin=250):

        fig = pl.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        x_min, y_min, x_max, y_max = self.getBoundingBox()
        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        sc = ax.scatter(self.x, self.y, 20, self.z, vmin=vmin, vmax=vmax)
        ax.axis('image')
        pl.colorbar(sc)
        # pl.show()

        
class Raster(object):
    #Assumes that the coordinates are cell centres
    def __init__(self,bbox,resolution,epsgCode):
    
        self.bbox = bbox
        x=np.arange(bbox[0],bbox[2]+0.1,resolution)
        y=np.arange(bbox[1],bbox[3]+0.1,resolution)

        self.x, self.y = np.meshgrid(x,y)
        self.z = np.empty(self.y.shape)
        self.z.fill(np.nan)

        self.xBinCentres = x
        self.yBinCentres = y
        self.xBinEdges = np.arange(bbox[0]-resolution/2,bbox[2]+resolution/2+0.1,resolution)
        self.yBinEdges = np.arange(bbox[1]-resolution/2,bbox[3]+resolution/2+0.1,resolution)
                
        self.resolution = resolution
        self.epsgCode = epsgCode
        
    def getSamples(self):
    
        X=self.x.ravel()
        Y=self.y.ravel()
        Z=self.z.ravel()
        
        mask = np.isnan(Z)
        samplesX = np.delete(X,np.where(mask))
        samplesY = np.delete(Y,np.where(mask))
        samplesZ = np.delete(Z,np.where(mask))
        
        return SamplePoints(samplesX,samplesY,samplesZ)

    def smooth(self, sigma):
        self.z = gaussian_filter(self.z, sigma)
    
    def getMissing(self):
    
        X=self.x.ravel()
        Y=self.y.ravel()
        Z=self.z.ravel()
        
        mask = np.isnan(Z)
        yetToBeFoundX = np.delete(X,np.where(~mask))
        yetToBeFoundY = np.delete(Y,np.where(~mask))
        yetToBeFoundZ = np.delete(Z,np.where(~mask))
        
        return SamplePoints(yetToBeFoundX,yetToBeFoundY,yetToBeFoundZ)

    def saveToFile(self,filename):
        cols=self.x.shape[1]
        rows=self.y.shape[0]
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
        outRaster.SetGeoTransform((self.bbox[0]-self.resolution/2, self.resolution, 0, self.bbox[1]-self.resolution/2, 0, self.resolution))
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(self.z)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(self.epsgCode)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()

    @classmethod
    def loadFromFile(cls,filename):
        ds = gdal.Open(filename)
        geoTrans = ds.GetGeoTransform()
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = ds.GetGeoTransform()

        band = ds.GetRasterBand(1)

        # noDataVal = band.GetNoDataValue()

        x_coords = x_size * np.arange(0, band.XSize) + upper_left_x + (x_size / 2)  # add half the cell size
        y_coords = y_size * np.arange(0, band.YSize) + upper_left_y + (y_size / 2)  # to centre the point

        rasterSRS = osr.SpatialReference()
        rasterSRS.ImportFromWkt(ds.GetProjectionRef())
        epsgCode = rasterSRS.ExportToProj4()

        bbox = np.zeros(4)
        bbox[0] = np.min(x_coords)
        bbox[1] = np.min(y_coords)
        bbox[2] = np.max(x_coords)
        bbox[3] = np.max(y_coords)

        newRaster = cls(bbox,x_size,epsgCode)
        newRaster.z = band.ReadAsArray().astype(np.float)

        # if noDataVal is None:
        #     mask = np.isnan(z)
        # else:
        #     mask = (z == noDataVal) | (z == 0.0)

        return newRaster

    def plot(self,vmin=-20,vmax=0,margin=250):

        fig = pl.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        x_min, y_min, x_max, y_max = self.bbox
        pc=pl.pcolormesh(self.xBinEdges, self.yBinEdges, self.z, vmin=vmin, vmax=vmax)
        ax.axis('image')
        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        cb=pl.colorbar(pc)
        cb.set_label('z')
        # pl.show()
        return fig, ax

    def interpolate(self,x,y):
        F = RegularGridInterpolator((self.yBinCentres,self.xBinCentres),self.z,bounds_error=False,fill_value=0.)
        pts = np.stack((y, x), axis=1)
        Z = F(pts)
        return Z

    def interpolateSpline(self,x,y):
        F = RectBivariateSpline(self.xBinCentres,self.yBinCentres,self.z.T,s=0)
        # pts = np.stack((y, x), axis=1)
        Z = F(x,y)
        return Z

    # def interpolateNatural(self,x,y):
    #     F = RegularGridInterpolator((self.yBinCentres,self.xBinCentres),self.z,bounds_error=False,fill_value=0.)
    #     pts = np.stack((y, x), axis=1)
    #     Z = F(pts)
    #     return Z

def alpha_shape(points, threshold):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    circum_radii = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c) + 1E-10)
        circum_r = a*b*c/(4.0*area + 1E-10) #to prevent divide by zero
        circum_radii.append(circum_r)
        # Here's the radius filter.
        #print circum_r
        if circum_r < threshold:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    return cascaded_union(triangles), edge_points
            
