
import numpy as np
import cv2
from scipy import ndimage
from pathlib import Path

## Definition of AuxiliaryClass
class AuxiliaryClass():
    def __init__(self):
        self.inputDir = None
        self.imagesFN = []
        self.imagesPath = []
        print(">>\tAuxiliaryClass successfully created")

    def setImagesDir(self, dir):
        self.inputDir = Path(dir)
        if self.inputDir.is_dir():
            print(">>\tOK! Directory exists")
        else:
            print(">>\tERROR! Directory doesn\'t exist")

    def setImagesFilenames(self, *args):
        if self.inputDir is None:
            print(">>\tERROR! Input directory not set. Call setImagesDir() first.")
            return
        self.imagesFN = [x for x in args]
        #print(self.imagesFN)
        self.imagesPath = [self.inputDir.joinpath(f) for f in self.imagesFN]
        #print(self.imagesPath)
        for f in self.imagesPath:
            if f.is_file():
                print(">>\tOK, file {:s} exists".format(f.name))
            else:
                print(">>\tERROR! File {:s} doesn\'t exist".format(f.name))
        
        
## Definition of ImageClass
class ImageClass():
    def __init__(self):
        self.imageOriginalData = None
        self.imageData = None
        self.imageCalibration = {}
        self.imageName = ""
        self.min = 0
        self.max = 0
        self.edgesROI = None
        self.ROI = None
        self.roiCalibration = {}
        self.edgesBG = None
        self.BG = None
        self.averageBG = 0.0
        self.ROImoments = None
        self.cmx = 0
        self.cmy = 0
        self.xsize = 0
        self.ysize = 0
        self.centeredROI = None
        print(">>\tImageClass successfully created")

    def loadImage(self, path,calibration:dict):
        self.imageOriginalData = cv2.imread(str(path))
        if self.imageOriginalData is None:
            print(f">>\tERROR! Could not load image from {path}. Check file format or path.")
            return
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
        if self.ROImoments["m00"] != 0:
            self.cmx = int( self.ROImoments["m10"] / self.ROImoments["m00"] )
            self.cmy = int( self.ROImoments["m01"] / self.ROImoments["m00"] )
        else:
            self.cmx = self.ROI.shape[1] // 2
            self.cmy = self.ROI.shape[0] // 2
            print(f">>\tWarning: Mass is zero for {self.imageName}, defaulting CoM to center.")
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
        
