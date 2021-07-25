#wykrycie na podstawie filmu
#na podstawie area mozna wykluczyc dodatkowe kartki

import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import closing, opening, square
from skimage.measure import label
import numpy as np


#w zadaniu będzie rozszerzenie png i liczby o 0...N-1
def read_images(num_of_images, path):
    images = []
    for im in range(1,num_of_images):
        filename = os.path.join(path, 'img_' + str(im)+'.jpg')
        img = skimage.io.imread(filename)
        images.append(img)
        
    return images


###
def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew    

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
###

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


def getWarp(biggest, img):
    widthImg, heightImg, _ = img.shape
    
    if biggest.size != 0:
        biggest=reorder(biggest)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        
        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
        
    return imgWarpColored

def getContours(img, imgContour):
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContour, contours, -1, (0,255,0), 20)
#    for cnt in contours:
#        area = cv2.contourArea(cnt)
#        
#        if area > 1000000:
#            cv2.drawContours(imgContour, cnt, -1, (0,255,0), 20)
#            peri = cv2.arcLength(cnt, True)
#            approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
            
        #sprawdzac pole powstalego prostokata?

    biggest, maxArea = biggestContour(contours)
    
    for big in biggest:
        x=big[0][0]
        y=big[0][1]
    
        imgContour = cv2.circle(imgContour, (x,y), radius=20, color=(255, 0, 0), thickness=-1)
       
    return biggest

#load images
images = read_images(3, '.\examples')

#process images:
#licznik pomocniczy
licznik=0
for image in images:

    imgContour = image.copy()
    
    imgHSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    ###
    h_min = 0
    h_max = 100
    s_min = 0
    s_max = 255
    v_min = 0
    v_max = 255

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    result = cv2.bitwise_and(image,image, mask = mask)   
    ###
    
    imgBlur = cv2.GaussianBlur(result, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    _, imgThresh = cv2.threshold(imgGray, 127, 255, 0)
    
    #ktory z tego ostatecznie do get contours?
    #imgClosing = closing(imgThresh, square(5))
    imgDial = cv2.dilate(imgThresh, square(15), iterations=2)
    #imgEro = cv2.erode(imgDial, square(5), iterations=1) 
    
    biggest = getContours(imgDial, imgContour)
    
    
    imgWarp = getWarp(biggest, image)
    
 
    imgStack = stackImages(1, ([image, imgHSV, result, imgBlur, imgGray, imgThresh, imgDial, imgContour, imgWarp]))
    
    plt.figure()
    plt.title("Image number "+str(licznik))
    #cv2.imshow("Image number "+str(licznik), imgWarp)
    plt.imshow(imgStack[0])
    licznik+=1
    print(imgWarp.shape[0])
    print(imgWarp.shape[1])








