import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import closing, opening, square
import numpy as np
from sklearn.externals import joblib
import sys




def preprocess(image):
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return 255 - image


def postprocess(image):
    image = cv2.medianBlur(image, 5)
    return image


def get_block_index(image_shape, yx, block_size): 
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)


def adaptive_median_threshold(img_in):
    THRESHOLD = 25
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



def FindGlobalYmin(cnt_with_ymin):
    ymin_global = float('inf')
    for i in range(len(cnt_with_ymin)):
        y = cnt_with_ymin[i][0]
        if y < ymin_global:
            ymin_global = y
            ix_with_ymin = i
            
    return ymin_global, ix_with_ymin

def DetectWords(imgGridless, imgWarp):
    
    
    #imgWarp = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
    imgGrayScale = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
    imgGrayScale[imgGrayScale>0]=0

    thresh = cv2.threshold(imgGridless, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # use morphology erode to blur horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    
    # use morphology open to remove thin lines from dotted lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    #slownik z danym konturem oraz najwyzej polozonym punktem z danego konturu
    cnt_with_ymin = {}
    #klucz dla slownika
    i=0
    for cnt in contours[:-1]:
        
        #area = cv2.contourArea(cnt)
        if cv2.contourArea(cnt) > 2000 and len(cnt)>8:
            
            #cv2.drawContours(imgWarp, cnt, -1, (255,0,0),5)
            
            #w kazdym konturze znalezc najwyzej polozyny y (czyli najmniejszy)
            ymin = float('inf')
            for cnt_ix in range(len(cnt)):
                y = cnt[cnt_ix][0][1]
                if y < ymin:
                    ymin = y
                    
            cnt_with_ymin[i] = [ymin, cnt]
            i+=1
            
    #while cnts_dict > 0?
    licznik = 0
    contours_with_numbers = []
    while len(cnt_with_ymin) > 0:
        licznik+=1
        #znalezienie najwyzej polozonego konturu
        ymin_global, ix_with_ymin = FindGlobalYmin(cnt_with_ymin)
            
        #znalezienie konturow 'w jednej linii' oraz nastepne usuniecie ich z konturow, ktore maja byc przeszukiwane
        #w celu znalezienia kolejnego ymin (tj. kolejnego rzędu)
        cnts_in_one_row = []
        #wrzucenie najwyzszego konturu
        cnts_in_one_row.append(cnt_with_ymin[ix_with_ymin][1])
        #usuniecie ze slownika
        del cnt_with_ymin[ix_with_ymin]
        
        dict_keys = list(cnt_with_ymin.keys())
        
        for i in dict_keys:
            diff_between_contours = abs(ymin_global-cnt_with_ymin[i][0])
            
            #prawdopodobnie jedna linia
            if diff_between_contours <=30:
                #wrzucenie konturu
                cnts_in_one_row.append(cnt_with_ymin[i][1])
                #usunac z cnt_with_ymin, aby w dalszych iteracjach nie sprawdzac tego samego
                del cnt_with_ymin[i]
                
        
        #find xmin and xmax in cnt
        xmax_in_row = 0
        for cnt in cnts_in_one_row:
            #cnt = cnt[1]
            xmin = float('inf')
            xmax = -1
            for cnt_ix in range(len(cnt)):
                x = cnt[cnt_ix][0][0]
                if x < xmin:
                    xmin = x
                    y_for_xmin = cnt[cnt_ix][0][1]
                if x > xmax:
                    xmax = x
                    y_for_xmax = cnt[cnt_ix][0][1]
              
                if xmax > xmax_in_row:
                    xmax_in_row = xmax
                    contour_with_number = cnt
            
            
            
            #draw line
            #if licznik%2 ==0:
            cv2.line(imgGrayScale, (xmin, y_for_xmin), (xmax, y_for_xmax), licznik, 5)

       
        contours_with_numbers.append(contour_with_number)
        
                
    
    return imgGrayScale, contours_with_numbers






#w zadaniu będzie rozszerzenie png i liczby o 0...N-1
def read_images(num_of_images, path):
    images = []
    for im in range(1,num_of_images+1):
        filename = os.path.join(path, 'img_' + str(im)+'.jpg')
        #filename = os.path.join(path, str(im)+'.png')
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
        pts1 = []
    else:
        biggest=reorder(biggest)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        
        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(heightImg,widthImg))

        
    return imgWarpColored, pts1


def resetWarp(image, org_pts, img_number, path):
    
    if len(org_pts) != 0:
        widthImg, heightImg = image.shape
        
        #add 20 pixels each side
        #gdy bedzie grayscale mozna podmienic na dodawanie kolumn i wierszy
        image = cv2.copyMakeBorder( image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
        #do all ops from getWarp inversely
        img = cv2.resize(image, (widthImg, heightImg), interpolation = cv2.INTER_NEAREST)

    
        
        #aktualne wierzcholki ()
        pts1 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
        #wierzcholki po przeksztalceniu (mozna je wziac z funkcji get Warp)
        pts2 = org_pts
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        #inv_matrix = np.linalg.inv(matrix)
        #
        img = cv2.warpPerspective(img, matrix, (heightImg, widthImg), cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

        #img = cv2.resize(image,(heightImg,widthImg))
    else:
        #zostala wykonana jedynie operacja usuniecia pikseli
        img = cv2.copyMakeBorder( image, 80, 80, 80, 300, cv2.BORDER_CONSTANT, value=0)
    
    #zamiast wyswietlenia zapis do pliku
    file_name = os.path.join(path, str(img_number) + '-wyrazy.png')
    cv2.imwrite(file_name, img)
    


def DeleteGrid(img):    

    imgThresh = cv2.bitwise_not(img)

    

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
    
    
    imgPlain = imgThresh.copy()
    
    for i in range(imgThresh.shape[0]):
        for j in range(imgThresh.shape[1]):
            if vertical[i][j]==255 or horizontal[i][j]==255:
                imgPlain[i][j]=0
                 
    
    return horizontal, vertical,  cv2.bitwise_not(imgPlain)


def getContours(img, imgContour):
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContour, contours, -1, (0,255,0), 20)


    biggest, maxArea = biggestContour(contours)

    
    for big in biggest:
        x=big[0][0]
        y=big[0][1]
    
        imgContour = cv2.circle(imgContour, (x,y), radius=20, color=(255, 0, 0), thickness=-1)
       
    return biggest, maxArea



def DetectNumbers(img, contours_with_numbers, img_number, path, knn):
    
    file_name = os.path.join(path, str(img_number)+'-indeksy.txt')

    #clear file
    open(file_name, 'w').close()
    file = open(file_name, 'a')
    
    #do wywalenia
    numer_wiersza = 0
    
    
    for cnt in contours_with_numbers:
        #do wywalenia
        numer_wiersza +=1
    
        x,y,w,h = cv2.boundingRect(cnt)
        img_frame_with_numbers = img[y:y+h, x:x+w]
        index_numbers = []
        

        
        image_height = h
        
        #from detectwords function
        thresh = cv2.threshold(img_frame_with_numbers, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 1)
        
 
        #usuwanie linii, ale niezastosowane, poniewaz ucina tez znaki
#        horizontal = np.copy(erosion)
#        
#        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
#        horizontal = cv2.erode(horizontal, horizontalStructure)
#        horizontal = cv2.dilate(horizontal, horizontalStructure)
#        
#        imgWithoutLines = erosion.copy()
#        imgWithoutLines = cv2.subtract(erosion, horizontal)
        
        #zdylowac pojedyncze cyferki
        contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #rgb_frame = cv2.cvtColor(erosion,cv2.COLOR_GRAY2RGB)
        
        #posortowanie konturow (cyfr) od lewej do prawej
        cnt_x_dict = {}
        for cnt in contours:
            x, _, _, _  = cv2.boundingRect(cnt)
            cnt_x_dict[x] = cnt
        #sort dict by x
        #lista zawierajaca krotki klucz-wartosc (x-kontur)
        sorted_list_with_cnt_x = sorted(cnt_x_dict.items())
        contours = []
        for x_contour in sorted_list_with_cnt_x:
            contours.append(x_contour[1])
                    
        #do wywalenia
        numer_cyfry = 0
        
        #wykrycie cyfr; przed tym sprobowac wyodrebnic
        for cnt in contours:
            #area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt) 
            if len(cnt)>30 and h>0.5*image_height and len(cnt)<300:
                #cv2.drawContours(rgb_frame, cnt, -1, (0,245,200),2)
                img_frame_with_number = img_frame_with_numbers[y:y+h, x:x+w]
                number = cv2.bitwise_not(img_frame_with_number)
                
                #do wywalenia 
                numer_cyfry +=1
                

                #set equal ratio before resizing
                num_width = number.shape[1]
                num_height = number.shape[0]
                #make them even
                if num_height %2 == 1:
                    #add one row
                    number = np.r_[number, np.zeros((1,num_width))]
                    num_height+=1
                if num_width %2 == 1:
                    #add one column
                    number = np.c_[number, np.zeros((num_height,1))]
                    num_width+=1
                    
                if num_height > num_width:
                    diff = num_height - num_width
                    #width of the part to add
                    add_to_one_side = int(diff/2)
                    column_to_add = np.zeros((num_height,add_to_one_side))
                    
                    number = np.c_[column_to_add, number, column_to_add]
                #to w ogole moze byc nieprawidlowe? i trzeba inaczej to rozpykac //podzielic na pol?
                #znalezienei skupiska pikseli na x i wzgledem tego podzielic?
                elif num_width < num_height:
                    pass
                    #erozja, bo byc moze zlaczone cyfry
                    #powtorne przetworzenie
                    
                
                number = cv2.resize(number, (28,28))
                #do wywalenia
#                plt.figure()
#                plt.title('numer wiersza '+str(numer_wiersza)+' numer cyfry: '+str(numer_cyfry) )
#                plt.imshow(number)
                
                
                #change shape of image
                flatted_number = number.reshape(-1)
                flatted_number = flatted_number.astype(int)
                flatted_number = list(flatted_number)
                #print(flatted_number)
                index_numbers.append(flatted_number)
                

                
        #index_numbers = np.array(index_numbers)
        try:
            predicted_number = knn.predict(index_numbers)
        except:
            continue
        #print(predicted_number) 
        predicted_number = str(predicted_number)
        predicted_number = predicted_number.replace(' ', '')
        predicted_number = predicted_number.replace('[', '')
        predicted_number = predicted_number.replace(']', '')
        predicted_number = predicted_number + '\n'
        file.write(predicted_number)
        
        
    file.close()
        

def LoadInput():

    #load images
    #images = read_images(10, '.\examples')
    ##read only one for test:
#    images=[]
#    img = skimage.io.imread('.\examples\img_29.jpg')
#    images.append(img)
    
            
    
    try:
#        input_path = str(sys.argv[1])
#        amount_of_images = int(sys.argv[2])
#        output_path = str(sys.argv[3])
        input_path = '.\examples'
        amount_of_images = 29
        output_path = '.\scores'
        images = read_images(amount_of_images, input_path) 
        

    except:
        #print()
        sys.exit('Invalid input')
        
    return images, output_path



def ProcessImages():
    
    images, output_path = LoadInput()
    
    #load model
    #knn = pickle.load(open('knnpickle_mnist', 'rb'))
    knn = joblib.load('mnist_model.pkl')
    
    #process images:
    #licznik pomocniczy
    licznik=1
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
        

        imgDial = cv2.dilate(imgThresh, square(10), iterations=2)

        
        biggest, maxArea = getContours(imgDial, imgContour) 
        imgWarp, org_pts = getWarp(biggest, maxArea, image)
        
        
        
        BLOCK_SIZE = 50
        image_in1 = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
        image_in2 = preprocess(image_in1)
        image_out1 = block_image_process(image_in2, BLOCK_SIZE)
        imgGridless = postprocess(image_out1)
        
        
        imgWords, contours_with_numbers = DetectWords(imgGridless, imgWarp.copy()) 
        DetectNumbers(imgGridless, contours_with_numbers, licznik, output_path, knn)
        
#        imgStack = stackImages(1, ([image, imgWarp, imgWords]))  
#        plt.figure()
#        plt.title("Image number "+str(licznik))
#        plt.imshow(imgStack)
      
        resetWarp(imgWords, org_pts, licznik, output_path)
        
        licznik+=1










ProcessImages()







