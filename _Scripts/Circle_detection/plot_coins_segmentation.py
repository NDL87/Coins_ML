input_filename = 'g'
input_fileext = 'jpg'
output_filename = input_filename+'_cropped'
output_fileext = 'jpg'

import time
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from PIL import Image

from skimage.feature import canny
from skimage.filters import sobel,gaussian,prewitt
from skimage import data,color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.exposure import histogram
import matplotlib.image as mpimg
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_hist, adjust_log, histogram, rescale_intensity
from scipy import ndimage

time_current = time.time()

image_rgb = mpimg.imread(input_filename+'.'+input_fileext)
coins_ = color.rgb2gray(image_rgb)
coins = adjust_log(coins_)
#coins = equalize_hist(coins)
coins = img_as_ubyte(coins)
#coins =
#coins = gaussian(coins,multichannel=True, sigma = 1)
#coins = data.coins()

range_outside = []
range_inside = []
width,height,channels = image_rgb.shape
if width<height:
    min_dimension = width
else:
    min_dimension = height
print(width,height)

for i in range(width):
    for j in range(height):
        if ((i - width/2) ** 2 + (j - height/2) ** 2) >= (min_dimension*0.48) ** 2:
            range_outside.append(coins[i,j])
        if ((i - width / 2) ** 2 + (j - height / 2) ** 2) <= (min_dimension*0.1) ** 2:
            range_inside.append(coins[i,j])

print(mean(range_inside),'mean(range_inside)')
print(mean(range_outside),'mean(range_outside)')

mean_inside = mean(range_inside)
mean_outside = mean(range_outside)

if mean_inside > mean_outside:
    background = 'dark'
    print('Dark Background')
else:
    background = 'light'
    print('Light Background')

markers = np.zeros_like(coins)
if background == 'dark':
    markers[coins < mean_outside] = 1
    markers[coins > mean_inside] = 2
else:
    markers[coins > mean_outside] = 1
    markers[coins < mean_inside] = 2


# Region-based segmentation
# =========================
#
# We therefore try a region-based method using the watershed transform.
# First, we find an elevation map using the Sobel gradient of the image.

from skimage import morphology

elevation_map = prewitt(coins)
edges = canny(elevation_map, sigma = 2)

fig1, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))
ax = axes.ravel()
ax[0].imshow(markers, cmap="gray")
ax[1].imshow(elevation_map, cmap="gray")
ax[2].imshow(edges)
plt.show()


hough_radii = np.arange(int(min_dimension/3), int(min_dimension/2), 2)
image_gray = hough_circle(edges, hough_radii)
accums, cxx, cyy, radii = hough_circle_peaks(image_gray, hough_radii, num_peaks = 5, total_num_peaks = 5)
cy = cxx[list(radii).index(max(radii))]
cx = cyy[list(radii).index(max(radii))]
radii = radii[list(radii).index(max(radii))]

#elevation_map = morphology.remove_small_objects(ndimage.binary_fill_holes(edges), min_size = 23)


'''
print(cy,cx,radii)

for x in range(width):
    for y in range(height):
        if ((x-cx)**2 + (y-cy)**2) > (radii*1.1)**2:
            elevation_map[x,y] = 0
#elevation_map = edges
'''



fig2, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
ax1.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
ax1.set_title('elevation map')

######################################################################
# Next we find markers of the background and the coins based on the extreme
# parts of the histogram of gray values.
'''
markers = np.zeros_like(coins)

if background == 'dark':
    markers[coins < mean_outside] = 1
    markers[coins > mean_inside] = 2
else:
    markers[coins > mean_outside] = 1
    markers[coins < mean_inside] = 2

if background == 'dark':
    markers[elevation_map < 1] = 1
    markers[elevation_map > 0] = 2
else:
    markers[elevation_map > 1] = 1
    markers[elevation_map < 0] = 2
'''
ax2.imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax2.set_title('markers')

######################################################################
# Finally, we use the watershed transform to fill regions of the elevation
# map starting from the markers determined above:

segmentation = morphology.watershed(elevation_map, markers)

ax3.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
ax3.set_title('segmentation')

######################################################################
# This last method works even better, and the coins can be segmented and
# labeled individually.
from scipy import ndimage
from skimage.color import label2rgb

segmentation = ndimage.binary_fill_holes(segmentation - 1)
labeled_coins, num_labels = ndimage.label(segmentation)

'''
for i in range(255):
    print('')
    for j in range(255):
        print(labeled_coins[i][j], end='')

for i in range(255):
    print('')
    for j in range(255):
        print(coins[i][j], end='')
'''
coins = coins*labeled_coins

#for i in range(255):
#    print('')
#    for j in range(255):
#        print(coins[i][j],' ', end='')

ax4.imshow(coins,  interpolation='nearest')
ax4.set_title('Final')
#axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
#axes[1].imshow(image_label_overlay, interpolation='nearest')

plt.tight_layout()

print(time.time() - time_current, 'Total time')

plt.show()

#image_label_overlay = label2rgb(labeled_coins, image=coins, colors=['white','red'])

#segmentation = color.gray2rgb(segmentation)
#coins = color.gray2rgb(coins)
#segmentation = img_as_ubyte(segmentation)
#coins = img_as_ubyte(coins)
#segmentation = segmentation + image_label_overlay

#output_image = Image.fromarray(coins)
#coins = (coins).astype(np.uint8)
output_image = Image.fromarray(np.uint8(coins))
output_image.save(output_filename+'.'+output_fileext)