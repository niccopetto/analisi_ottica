# FILE: gs_engine.py
import numpy as np
from LightPipes import *

import cv2

from scipy import ndimage
from pathlib import Path

import importlib
from scipy.constants import pi

import AuxiliaryTools

mm = 1e-3
nm = 1e-9

# Compatibility for numpy < 2.0.0
if hasattr(np, 'trapezoid'):
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz

# --- Utility ---
def color_txt(txt, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{txt}\033[0m"

class LengthError(Exception):
    pass

## Definition of AuxiliaryClass
class AuxiliaryClass():
    def __init__(self):
        print(">>\tAuxiliaryClass successfully created")

    def setImagesDir(self, dir):
        self.inputDir = Path(dir)
        if self.inputDir.is_dir():
            print(">>\tOK! Directory exists")
        else:
            print(">>\tERROR! Directory doesn\'t exist")

    def setImagesFilenames(self, *args):
        self.imagesFN = [x for x in args]
        #print(self.imagesFN)
        self.imagesPath = [self.inputDir.joinpath(f) for f in self.imagesFN]
        #print(self.imagesPath)
        for f in self.imagesPath:
            if f.is_file():
                print(">>\tOK, file {:s} exists".format(f.name))
            else:
                print(">>\tERROR! File {:s} doesn\'t exist".format(f.name))
        
class ImageClass():
    def __init__(self):
        print(">>\tImageClass successfully created")

    def loadImage(self, path,calibration:dict):
        self.imageOriginalData = cv2.imread(str(path))
        self.imageData = self.imageOriginalData.copy()
        self.imageCalibration = dict()
        self.imageCalibration["y"] = calibration["y"]
        self.imageCalibration["x"] = calibration["x"]
        self.imageName = path.parts[-1]
        print(">>\tImage {:s} successfully loaded. Shape: {:s}. Calibration: {:s}".format(
            str(path.name), str(self.imageOriginalData.shape), str(self.imageCalibration)))
    
    def selectChannel(self, channel):
        if channel == "B"  :
            self.imageData = self.imageOriginalData[:,:,0]
        elif channel == "G" :
            self.imageData = self.imageOriginalData[:,:,1]
        elif channel == "R" :
            self.imageData = self.imageOriginalData[:,:,2]
        print(f">>\tChannel '{channel}' selected for {self.imageName} image. New shape: {self.imageData.shape}")

    def showOriginal(self, gamma=1):
        WinName = f"Original image: {self.imageName}"
        cv2.namedWindow(WinName, cv2.WINDOW_NORMAL  )
        cv2.imshow(WinName, self.imageData)
        cv2.waitKey(0)
        cv2.destroyWindow(WinName)

    def resampleOriginal(self, factor=0.5, order=3):
        print(">>\tOriginal shape: {:s}".format(str(self.imageData.shape)))
        self.imageData = ndimage.zoom(self.imageData, zoom = factor, output=None, order=order)
        if isinstance(factor,float):
            for k,v in self.imageCalibration.items():
                self.imageCalibration[k] = v/factor
        elif isinstance(factor, (tuple,list,np.ndarray)):
            for j,(k,v) in enumerate(self.imageCalibration.items()):
                self.imageCalibration[k] = v/factor[j]
        print(">>\tResampled shape: {:s}. New calibration: {:s}".format(str(self.imageData.shape),str(self.imageCalibration)))
            
    def calculateMinMax(self):
        self.min, self.max = np.min(self.imageData), np.max(self.imageData)
        print(">>\tmin: {:d}, max: {:d}".format(self.min, self.max))
    
    def selectROI(self):
        WinName = f"Select ROI: {self.imageName}"
        cv2.namedWindow(WinName, cv2.WINDOW_NORMAL)
        self.edgesROI = cv2.selectROI(WinName, self.imageData, True, False)
        #print(self.edges)
        self.ROI = self.imageData[self.edgesROI[1]:self.edgesROI[1]+self.edgesROI[3], 
                                  self.edgesROI[0]:self.edgesROI[0]+self.edgesROI[2]]
        self.roiCalibration = self.imageCalibration.copy()
        cv2.waitKey(0)
        cv2.destroyWindow(WinName)
        print(">>\tROI shape: {:s}".format(str(self.ROI.shape)))

    def showROI(self, gamma=1):
        WinName = f"ROI: {self.imageName}"
        cv2.namedWindow(WinName, cv2.WINDOW_NORMAL)
        cv2.imshow(WinName, self.ROI)
        cv2.waitKey(0)
        cv2.destroyWindow(WinName)

    def selectBG(self):
        WinName = f"Select BG: {self.imageName}"
        cv2.namedWindow(WinName, cv2.WINDOW_NORMAL)
        self.edgesBG = cv2.selectROI(WinName, self.imageData, True, False)
        cv2.waitKey(0)
        cv2.destroyWindow(WinName)
        self.BG = self.imageData[self.edgesBG[1]:self.edgesBG[1]+self.edgesBG[3],
                                 self.edgesBG[0]:self.edgesBG[0]+self.edgesBG[2]]
        print(">>\tBG shape: {:s}".format(str(self.BG.shape)))

    def retrieveAverageBG(self):
        self.averageBG = np.mean(self.BG)
        print(">>\tAverage BG: {:f}".format(self.averageBG))

    def subtractBGfromROI(self):
        self.ROI = self.ROI-self.averageBG

    def resampleROI(self, factor, order=3):
        self.ROI = ndimage.zoom(self.ROI, zoom = factor, output=None, order=order)
        if isinstance(factor,float):
            for k,v in self.roiCalibration.items():
                self.roiCalibration[k] = v/factor
        elif isinstance(factor, (tuple,list,np.ndarray)):
            for j,(k,v) in enumerate(self.roiCalibration.items()):
                self.roiCalibration[k] = v/factor[j]
        print(f"ROI new shape: {self.ROI.shape}. New ROI calibration: {self.roiCalibration}")

    def getCenterOfMassOfROI(self):
        self.ROImoments = cv2.moments(self.ROI)
        self.cmx = int( self.ROImoments["m10"] / self.ROImoments["m00"] )
        self.cmy = int( self.ROImoments["m01"] / self.ROImoments["m00"] )
        print("{:s} CoM (y,x): ({:d}, {:d})".format(self.imageName,self.cmy, self.cmx))

    def buildImageWithSize(self, xsize=1024, ysize=1024):
        self.xsize, self.ysize = xsize, ysize
        self.centeredROI = np.zeros((self.ysize, self.xsize))
        self.centeredROI[:self.ROI.shape[0],:self.ROI.shape[1]] = self.ROI
        self.centeredROI = np.roll(self.centeredROI, 
                                   (self.ysize//2 - self.cmy, self.xsize//2 - self.cmx), axis=(0,1))

    def showCenteredROI(self):
        WinName = f"Centered ROI: {self.imageName}"
        cv2.namedWindow(WinName, cv2.WINDOW_NORMAL)
        cv2.imshow(WinName, self.centeredROI)
        cv2.waitKey(0)
        cv2.destroyWindow(WinName)
    
    
    @classmethod
    def rescale2minmax(cls, arrayin, newmin, newmax):
        oldmin, oldmax = np.min(arrayin), np.max(arrayin)
        a = (newmax - newmin) / (oldmax - oldmin)
        b = (newmin * oldmax - newmax * oldmin) / (oldmax - oldmin)
        arrayout = a * arrayin + b
        return arrayout
    
    @classmethod
    def enhanceContrast(cls, image, gamma):
        oldmin, oldmax = np.min(image), np.max(image)
        # First, rescale to [0,1]
        zu = cls.rescale2minmax(image, 0.0, 1.0)
        gammacorrected = np.power(zu, gamma)
        final = cls.rescale2minmax(gammacorrected, oldmin, oldmax)
        return final
        
        
# --- La Classe Principale ---

class GerSaxPhaseRetriever(object):
    def __init__(self, *inputImages:AuxiliaryTools.ImageClass, 
                 distances:np.ndarray[float]|list[float]|tuple[float],
                 wavelength:float,
                 ordering:np.ndarray[int]|list[int]|tuple[int]):
        if len(ordering) != len(inputImages):
            raise LengthError("The number of images and the ordering list must be coherent")
        if isinstance(ordering, (list,tuple)):
            ordering = np.array(ordering,dtype=int)
        if len(distances) != len(inputImages)-1:
            raise LengthError(f"The number of distances must be {len(inputImages)-1}")
        _,counts = np.unique_counts(ordering)
        if np.any(counts>1):
            raise ValueError("The ordering list must contain unique values")
        #for img in inputImages:
        #   img.centeredROI = np.where(img.centeredROI<0,0,img.centeredROI)
        self.inputImages = tuple(inputImages[i] for i in ordering)
        self.wavelength = wavelength
        self.distances = distances
        self.total_length = np.sum(distances)
        self.imagesNames = [img.imageName for img in self.inputImages]
        self.size = self.inputImages[0].xsize*self.inputImages[0].roiCalibration['x']
        self.FFname = self.imagesNames[-1]
        
        # Initialize Error arrays and FarField
        self.MSError = None
        self.CEError = None
        self.FarField = None
    
    def __error_(self,img:AuxiliaryTools.ImageClass,F,kind='MSE'):
        abs_E = np.abs(F.field)
        if kind == 'MSE':
            err = trapezoid(trapezoid((abs_E-np.sqrt(img.centeredROI))**2,dx=img.roiCalibration['y']),
                               dx=img.roiCalibration['x'])
        elif kind == 'Dkl':
            err = trapezoid(trapezoid(img.centeredROI*np.log((1e-60+img.centeredROI)/Intensity(F)),dx=img.roiCalibration['y']),
                                             dx=img.roiCalibration['x'])
        return err


    def __round_trip__(self,F,k,J,Niterations):
        """Perform a round trip for the k-th iteration."""
        self.MSError[k] = self.MSError[k]+self.__error_(self.inputImages[-1],F,'MSE')
        self.CEError[k] = self.CEError[k]+self.__error_(self.inputImages[-1],F,'Dkl')
        F=SubIntensity(self.inputImages[-1].centeredROI,F)
        F=Forvard(-self.total_length,F)
        for j,img in enumerate(self.inputImages[:-1]):
            print(color_txt(f"iteration: {k+1}/{Niterations}",int(255*(J-j-1)/J),255,0),end='\r',flush=True)
            self.MSError[k] = self.MSError[k]+self.__error_(img,F,'MSE')
            self.CEError[k] = self.CEError[k]+self.__error_(img,F,'Dkl')
            F=SubIntensity(img.centeredROI,F)
            F=Forvard(self.distances[j],F)
        return F

    def GS_algorithm(self,Niterations:int=50):
        self.MSError = np.zeros(Niterations)
        self.CEError = np.zeros(Niterations)
        N = self.inputImages[0].xsize
        F = Begin(self.size, self.wavelength,N)
        J = len(self.inputImages[:-1])
        for k in range(Niterations):
            print(color_txt(f"iteration: {k+1}/{Niterations}",255,255,0),end='\r',flush=True)
            F = self.__round_trip__(F,k,J,Niterations)
        self.FarField = F

    def RetrieveIntensity(self,name:str):
        if name not in self.imagesNames:
            raise ValueError(f"Image {name} not found in the input images")
        if name == self.FFname:
            return Intensity(self.FarField)
        else:
            index = self.imagesNames.index(name)
            z = np.sum(self.distances[index:])
            F = Forvard(-z,self.FarField)
            return Intensity(F)
    def RetrievePhase(self,name:str,unwrap:bool=False):
        if name not in self.imagesNames:
            raise ValueError(f"Image {name} not found in the input images")
        if name == self.FFname:
            return Phase(self.FarField,unwrap=unwrap)
        else:
            index = self.imagesNames.index(name)
            z = np.sum(self.distances[index:])
            F = Forvard(-z,self.FarField)
            return Phase(F,unwrap=unwrap)
