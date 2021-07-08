import cv2
import os
import matplotlib.pyplot as plt
import skimage


#w zadaniu będzie rozszerzenie png i liczby o 0...N-1
def read_images(num_of_images, path):
    images = []
    for im in range(1,num_of_images):
        filename = os.path.join(path, 'img_' + str(im)+'.jpg')
        img = cv2.imread(filename)
        images.append(img)
        
    return images


#load images
images = read_images(4, '.\examples')

#process images:
for image in images:
    pass


img_lab = skimage.color.rgb2lab(images[2])

skimage.io.imshow(img_lab[:,:,2])