import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import closing, opening, square
from skimage.measure import label

#w zadaniu będzie rozszerzenie png i liczby o 0...N-1
def read_images(num_of_images, path):
    images = []
    for im in range(1,num_of_images):
        filename = os.path.join(path, 'img_' + str(im)+'.jpg')
        img = skimage.io.imread(filename)
        images.append(img)
        
    return images


#load images
images = read_images(4, '.\examples')

#process images:
#licznik pomocniczy
licznik=0
for image in images:
    licznik+=1

    original_image = image.copy()
    
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    
    
    thresh = closing(thresh, square(10))
    plt.figure()
    plt.title("Threshold after closing")
    plt.imshow(thresh)
#    thresh = opening(thresh, square(2))
#    plt.figure()
#    plt.title("Threshold after opening")
#    plt.imshow(thresh)
    
    #gaussian filter + otsu threshohlind?
#    blur = cv2.GaussianBlur(thresh, (9,9),0)
#    plt.figure()
#    plt.title("Gaussian Blur")
#    plt.imshow(blur)
#    
#    _,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    blur = cv2.GaussianBlur(thresh, (9,9),0)
#    plt.figure()
#    plt.title("Otsu thresh")
#    plt.imshow(thresh)
    
    
    #jakies tricki z lab3 piro:
#    lab_fig = label(original_image)
#    plt.figure()
#    plt.title('lab_fig')
#    plt.imshow(lab_fig)
    
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    
    cv2.drawContours(original_image, contours, -1, (0,255,0), 10)
    
    plt.figure()
    plt.title("contours only")
    plt.imshow(original_image)
    
#    if licznik ==3:
#        plt.figure()
#        plt.imshow(image)
#        plt.show()
#        print('pauza')
#    
#
    for cnt in contours :
      
        
        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
      
        #i = 0
        
        for appr in approx:
            x=appr[0][0]
            y=appr[0][1]
    
            original_image = cv2.circle(original_image, (x,y), radius=20, color=(255, 0, 0), thickness=-1) 
            #cv2.putText(image, str(i), (x, y), font, 0.3, (255, 0, 0))
            #i+=1
         
#        if len(approx)==4:
#            break
    
    
    plt.figure()
    plt.imshow(original_image)
    plt.show










