input_filename = 'ee'
input_fileext = 'jpg'
output_filename = input_filename+'_cropped'
output_fileext = 'jpg'

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte,img_as_int,img_as_uint,img_as_float, invert
from skimage.exposure import equalize_hist, adjust_log, histogram, rescale_intensity
from skimage import filters

import matplotlib.image as mpimg
import time

# Load picture and detect edges
time_current = time.time()

#image_rgb = data.coffee()[0:220, 160:420]
image_rgb = mpimg.imread(input_filename+'.'+input_fileext)



image_res = np.copy(image_rgb)
image_orig = np.copy(image_rgb)
image_r = np.copy(image_rgb)

image_gray = color.rgb2gray(image_rgb)
image_gray_orig = np.copy(image_gray)
image_gray = img_as_ubyte(image_gray)
image_gray = adjust_log(image_gray)
image_gray = equalize_hist(image_gray)
#image_gray = filters.gaussian(image_gray,multichannel=True, sigma = 1)

edges = canny(image_gray, sigma = 4)
edges_plot_1 = np.copy(edges)
circle_plot = np.copy(image_rgb)

width, height, channels = image_rgb.shape
if width<height:
    min_dimension = width
else:
    min_dimension = height
print(width, height)

'''
line = []

x = 100
for y in range(height):
    line.append(image_gray[x,y])

hist, hist_centers = histogram(image_gray)

print(line)
print(sorted(line))

fig1, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))
ax = axes.ravel()
ax[0].imshow(image_gray_orig, cmap="gray")
ax[1].imshow(image_gray, cmap="gray")
ax[2].plot(hist_centers, hist, lw=2)

plt.show()

'''
print(time.time() - time_current, 'Canny is done')

# Detect two radii
hough_radii = np.arange(int(min_dimension/3), int(min_dimension/2), 2)
image_gray = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(image_gray, hough_radii, num_peaks = 5, total_num_peaks = 5)

for i in range(5):
    print(accums[i],radii[i])

cx = cx[list(radii).index(max(radii))]
cy = cy[list(radii).index(max(radii))]
radii = radii[list(radii).index(max(radii))]
circy, circx = circle_perimeter(int(cx), int(cy), int(radii))

print(cx,cy,radii)
print(time.time() - time_current, 'Circle detection is done')

#for x in range(int(height)):
#    for y in range(int(width)):
#        if ((x-cx)**2 + (y-cy)**2) < (radii*0.9)**2 or ((x-cx)**2 + (y-cy)**2) > (radii*1.1)**2:
#           edges[y,x] = 0


for i in range(len(circx)):
    circle_plot[circx[i], circy[i]] = (255, 0, 0)

for i in range(min(circx),max(circx)):
    x_min = circy[list(circx).index(i)]
    if list(circx).count(i) > 1:
        x_max = circy[list(circx).index(i) + 1]
        if x_min < x_max:
            pass
        else:
            x_min = x_max
            x_max = circy[list(circx).index(i)]
    else:
        x_max = x_min

    for j in range(min(circy),max(circy)):
        if not x_min <= j <= x_max:
            try:
                image_res[i, j] = (255, 255, 255)
            except:
                pass

image_res = image_res[min(circx):max(circx),min(circy):max(circy)]

print(time.time() - time_current, 'Image cropping is done')

#print(image_res.shape)

fig2, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))

ax1.set_title('Original picture')
ax1.imshow(image_orig)

ax2.set_title('Canny Edge Detection')
ax2.imshow(edges_plot_1)

ax3.set_title('Circle detector')
ax3.imshow(circle_plot)

cx, cy, colors = image_res.shape

image_res_2 = image_orig[min(circx):max(circx),min(circy):max(circy)]
image_res = filters.gaussian(image_res,multichannel=True, sigma = 1)*255

for x in range(int(cx)):
    for y in range(int(cy)):
        if ((x-cx/2)**2 + (y-cy/2)**2) > (cx*0.98/2)**2:
           # print(image_res[x,y])
            image_res_2[x,y] = image_res[x,y]

ax4.set_title('Circle detector')
ax4.imshow(image_res_2)
print(time.time() - time_current, 'Total time')
plt.show()

#print(image_res_2)

output_image = Image.fromarray(image_res_2)
output_image.save(output_filename+'.'+output_fileext)




