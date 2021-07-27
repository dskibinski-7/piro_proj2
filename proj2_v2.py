#wykrycie na podstawie filmu
#na podstawie area mozna wykluczyc dodatkowe kartki

import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import closing, opening, square
from skimage.measure import label
import numpy as np




####################from stackover
BLOCK_SIZE = 50
THRESHOLD = 25


def preprocess(image):
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return 255 - image


def postprocess(image):
    image = cv2.medianBlur(image, 5)
    # image = cv2.medianBlur(image, 5)
#    kernel = np.ones((3,3), np.uint8)
#    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def get_block_index(image_shape, yx, block_size): 
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)


def adaptive_median_threshold(img_in):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out


def block_image_process(image, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])

    return out_image
#######################################################












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

    

def getWarp(biggest, maxArea, img):
    widthImg, heightImg, _ = img.shape
    
    imgWarpColored = img.copy()
    
    if len(biggest)!=4 or maxArea < 2055550:
        imgWarpColored=imgWarpColored[80:imgWarpColored.shape[0] - 300, 80:imgWarpColored.shape[1] - 80]      
    else:
        biggest=reorder(biggest)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        
        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(heightImg,widthImg))
        
    return imgWarpColored

def DeleteGrid(img):
    
    #jezeli przekazany obraz z funkcji stackoverowerowej, to ponizssze operacje niepotrzebne (wejscie to obraz binarny)
#    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.bitwise_not(img)
#    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                                cv2.THRESH_BINARY, 31, -4) 
    

    horizontal = np.copy(imgThresh)
    vertical = np.copy(imgThresh)
    
    cols = horizontal.shape[1]
    horizontal_size = cols // 30  #????
    #horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))    
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
    rows = vertical.shape[0]
    vertical_size = cols // 30  #?????    
    #verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)) 
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    #kernel = np.ones((40,3),np.uint8)
    #dilation = cv2.dilate(img,kernel,iterations=1)
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    imgPlain = imgThresh.copy()
    
    for i in range(imgThresh.shape[0]):
        for j in range(imgThresh.shape[1]):
            if vertical[i][j]==255 or horizontal[i][j]==255:
                imgPlain[i][j]=0
                
    #find lines again
#    horizontal2 = np.copy(imgPlain)
#    cols2 = horizontal2.shape[1]
#    #horizontal_size2 = cols2 // 35  #?????    
#    horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))    
#    horizontal2 = cv2.erode(horizontal2, horizontalStructure2)
#    horizontal2 = cv2.dilate(horizontal2, horizontalStructure2)
    
    #'oczyszczenie' z drobinek
    #imgClosing = closing(imgThresh, square(5))
    #imgDial = cv2.dilate(imgPlain, np.ones(10,1), iterations=2)
    #imgEro = cv2.erode(imgDial, square(5), iterations=1) 

    
    
    return horizontal, vertical,  cv2.bitwise_not(imgPlain)


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
#    print(maxArea)
#    print(biggest)
#    print('-'*20)
    
    for big in biggest:
        x=big[0][0]
        y=big[0][1]
    
        imgContour = cv2.circle(imgContour, (x,y), radius=20, color=(255, 0, 0), thickness=-1)
       
    return biggest, maxArea

#load images
images = read_images(2, '.\examples')

#process images:
#licznik pomocniczy
licznik=0
for image in images:

    imgContour = image.copy()
    
    #preliminary image processing
    imgHSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
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
    
    imgBlur = cv2.GaussianBlur(result, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    _, imgThresh = cv2.threshold(imgGray, 127, 255, 0)
    
    
    #ktory z tego ostatecznie do get contours?
    #imgClosing = closing(imgThresh, square(5))
    imgDial = cv2.dilate(imgThresh, square(10), iterations=2)
    #imgEro = cv2.erode(imgDial, square(5), iterations=1) 
    
    biggest, maxArea = getContours(imgDial, imgContour) 
    imgWarp = getWarp(biggest, maxArea, image)
    
    #imgGray, imgThresh, horizontal, vertical, imgPlain = DeleteGrid(imgWarp)
    
    
    #############
    image_in1 = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
#    plt.figure()
#    plt.imshow(image_in)
    image_in2 = preprocess(image_in1)
#    plt.figure()
#    plt.imshow(image_in)
    image_out1 = block_image_process(image_in2, BLOCK_SIZE)
#    plt.figure()
#    plt.imshow(image_out)
    imgGridless = postprocess(image_out1)
#    plt.figure()
#    plt.imshow(image_out)
    ############
    
    #moze dac jeszce raz delete grid na imgout2
    
    #horizontal, vertical, imgPlain = DeleteGrid(image_out2)
 
    #imgStack = stackImages(1, ([image, imgHSV, result, imgBlur, imgGray, imgThresh, imgDial, imgContour, imgWarp]))
    imgStack = stackImages(1, ([imgWarp, imgGridless]))
    
    plt.figure()
    plt.title("Image number "+str(licznik))
    plt.imshow(imgStack)
#    plt.figure()
#    plt.imshow(255-image_out2)
    licznik+=1









