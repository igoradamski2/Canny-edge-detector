import cv2
import numpy as np

class EdgeDetection:

    def __init__(self, path):
        self.img  = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)

    def smooth(self, size, sd):
        self.simg = cv2.GaussianBlur(self.img, size, sd)

    def getGradient(self):
        img = self.simg
        
        xlen, ylen = img.shape

        self.xgrad    = np.zeros(img.shape)
        self.ygrad    = np.zeros(img.shape)
        self.gradnorm = np.zeros(img.shape)
        
        img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
        for x in range(xlen):
            for y in range(ylen):
                self.xgrad[x, y]    = 0.5 * (int(img[x+2, y+1]) - int(img[x, y+1]))
                self.ygrad[x, y]    = 0.5 * (int(img[x+1, y+2]) - int(img[x+1, y]))
                self.gradnorm[x, y] = np.sqrt(self.xgrad[x,y] ** 2 + self.ygrad[x,y] ** 2)
        
        self.gradnorm = self.gradnorm/self.gradnorm.max() * 255

    def getGradDirection(self):

        gradDir = np.zeros(self.img.shape)
        #xgrad   = self.xgrad/(self.gradnorm + 1e-9)
        #ygrad   = self.ygrad/(self.gradnorm + 1e-9)
        xgrad    = self.xgrad
        ygrad    = self.ygrad

        for x in range(self.img.shape[0]):
            for y in range(self.img.shape[1]):
                ang = EdgeDetection.getAngle(xgrad[x,y], ygrad[x,y])
                gradDir[x, y] = ang if ang >= 0 else ang + 360

        return gradDir

    def maximalSupression(self):
        gradDir     = EdgeDetection.getGradDirection(self)
        self.maxSup = np.zeros(gradDir.shape)
        gradNorms   = self.gradnorm
        gradNorms   = cv2.copyMakeBorder(gradNorms,1,1,1,1,cv2.BORDER_REPLICATE)

        for x in range(gradDir.shape[0]):
            for y in range(gradDir.shape[1]):

                quadrant = EdgeDetection.mapToQuadrant(gradDir[x,y])
                if quadrant == 1 or quadrant == 5:
                    pos_dir_val = gradNorms[x+1, y+2]
                    neg_dir_val = gradNorms[x+1, y]
                    curr_val    = gradNorms[x+1, y+1]
                    if curr_val == max(pos_dir_val, neg_dir_val, curr_val):
                        self.maxSup[x,y] = curr_val
                    else:
                        self.maxSup[x,y] = 0

                elif quadrant == 2 or quadrant == 6:
                    pos_dir_val = gradNorms[x, y+2]
                    neg_dir_val = gradNorms[x+2, y]
                    curr_val    = gradNorms[x+1, y+1]
                    if curr_val == max(pos_dir_val, neg_dir_val, curr_val):
                        self.maxSup[x,y] = curr_val
                    else:
                        self.maxSup[x,y] = 0

                elif quadrant == 3 or quadrant == 7:
                    pos_dir_val = gradNorms[x, y+1]
                    neg_dir_val = gradNorms[x+2, y+1]
                    curr_val    = gradNorms[x+1, y+1]
                    if curr_val == max(pos_dir_val, neg_dir_val, curr_val):
                        self.maxSup[x,y] = curr_val
                    else:
                        self.maxSup[x,y] = 0

                elif quadrant == 4 or quadrant == 8:
                    pos_dir_val = gradNorms[x, y]
                    neg_dir_val = gradNorms[x+2, y+2]
                    curr_val    = gradNorms[x+1, y+1]
                    if curr_val == max(pos_dir_val, neg_dir_val, curr_val):
                        self.maxSup[x,y] = curr_val
                    else:
                        self.maxSup[x,y] = 0


    def classifyStrongPixels(self, minVal, maxVal):

        strong_pixels = np.where(self.maxSup >= maxVal)
        weak_pixels   = np.where((self.maxSup < maxVal) & (self.maxSup >= minVal))

        self.maxSup[np.where(self.maxSup < minVal)] = 0

        return strong_pixels, weak_pixels

    def applyThreshold(self, minVal, maxVal):

        strongs, weaks = EdgeDetection.classifyStrongPixels(self, minVal, maxVal)

        self.edges = np.zeros(self.maxSup.shape)
        self.edges[strongs] = 255

        strongs = np.column_stack(strongs)
        weaks   = np.column_stack(weaks)
        while strongs.size > 0:
            new_strongs = []
            new_weaks = []
            for weak_point in weaks:
                calc = np.linalg.norm(strongs - weak_point, axis = 1)
                if any(calc >= np.sqrt(2)):
                    new_strongs.append(weak_point)
                    self.edges[weak_point[0], weak_point[1]] = 255
                else:
                    new_weaks.append(weak_point)
            
            strongs = np.array(new_strongs)
            weaks   = np.array(new_weaks)


    


    # -------------------------

    @staticmethod
    def getAngle(x, y):
        return -(180/np.pi)*np.arctan2(x, y)

    @staticmethod
    def mapToQuadrant(angle):

        if (angle < 22.5 and angle >= 0) or (angle >= 337.5 and angle <= 360):
            return 1
        
        elif (angle >= 22.5 and angle < 67.5):
            return 2

        elif (angle >= 67.5 and angle < 112.5):
            return 3

        elif (angle >= 112.5 and angle < 157.5):
            return 4

        elif (angle >= 157.5 and angle < 202.5):
            return 5

        elif (angle >= 202.5 and angle < 247.5):
            return 6

        elif (angle >= 247.5 and angle < 292.5):
            return 7
        
        elif (angle >= 292.5 and angle < 337.5):
            return 8

        else:
            print("Error!!")
            print(angle)

def CannyDetector(path, scale, thresholds):

    print("Loading image")
    img = EdgeDetection(path)
    print("Smooting image")
    img.smooth((5,5), scale)
    print("Getting image gradient")
    img.getGradient()
    print("Maximal supression")
    img.maximalSupression()
    print("Applying applyThreshold")
    img.applyThreshold(thresholds[0],thresholds[1])

    return img

