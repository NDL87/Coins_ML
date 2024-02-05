import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import matplotlib.image as mpimg
import time
from PIL import Image
import numpy as np

# Load picture, convert to grayscale and detect edges
#image_rgb = data.coffee()[0:220, 160:420]
image_rgb = mpimg.imread('g.jpg')
#print(image_rgb)
time_current = time.time()
#image_rgb = np.copy(image_rgb)

image_gray = color.rgb2gray(image_rgb)

width, height, channels = image_rgb.shape
print(width, height)

if width>height:
    max_dimension = width
else:
    max_dimension = height

canny_x = []
canny_y = []

edges = canny(image_gray, sigma = 3.3, low_threshold=10, high_threshold=80)
#print(edges)
print(time.time() - time_current)

for i in range(width-1):
    for j in range(height-1):
        if edges[i,j] == True:
            canny_x.append(i)
            canny_y.append(j)

print(min(canny_x),min(canny_y))
print(max(canny_x),max(canny_y))

print(time.time() - time_current)
image_res = np.copy(image_rgb) #[min(canny_x):max(canny_x),min(canny_y):max(canny_y)]

print(image_res.shape)

#height = max(canny_y) - min(canny_y)
#width = max(canny_x) - min(canny_x)

for i in range(int(width/2)+1):
    for j in range(int(height/2)+1):
        if edges[i-1,j-1] == True or edges[i-1,j] == True or edges[i,j-1] == True:
                image_res[i, j] = image_res[i, j]
                break
        elif edges[i,j] == True:
            image_res[i,j] = image_res[i, j]
            break
        else:
            image_res[i, j] = (255, 255, 255)

for i in range(width-1,int(width/2),-1):
    for j in range(height-1,int(height/2),-1):
        if edges[i-1,j-1] == True or edges[i-1,j] == True or edges[i,j-1] == True:
                image_res[i, j] = image_res[i, j]
                break
        elif edges[i,j] == True:
            image_res[i,j] = image_res[i, j]
            break
        else:
            image_res[i, j] = (255, 255, 255)

for i in range(int(width/2)+1):
    for j in range(height-1,int(height/2),-1):
        if edges[i-1,j-1] == True or edges[i-1,j] == True or edges[i,j-1] == True:
                image_res[i, j] = image_res[i, j]
                break
        elif edges[i,j] == True:
            image_res[i,j] = image_res[i, j]
            break
        else:
            image_res[i, j] = (255, 255, 255)

for i in range(width-1,int(width/2),-1):
    for j in range(int(height/2)+1):
        if edges[i-1,j-1] == True or edges[i-1,j] == True or edges[i,j-1] == True:
                image_res[i, j] = image_res[i, j]
                break
        elif edges[i,j] == True:
            image_res[i,j] = image_res[i, j]
            break
        else:
            image_res[i, j] = (255, 255, 255)

print(time.time() - time_current)

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 4),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Canny Edge Detection')
ax2.imshow(edges)

ax3.set_title('Removed Background')
ax3.imshow(image_res)

print(time.time() - time_current)

plt.show()

