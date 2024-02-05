input_filename = 'f'
input_fileext = 'jpg'
output_filename = input_filename+'_cropped'
output_fileext = 'jpg'

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image

time_current = time.time()

img = mpimg.imread(input_filename+'.'+input_fileext)
mask = img.astype(bool)
img = img.astype(float)
# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(img, mask=mask)

print(time.time() - time_current)

# Take a decreasing function of the gradient: we take it weakly
# dependant from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())

print(time.time() - time_current)

# Force the solver to be arpack, since amg is numerically
# unstable on this example
labels = spectral_clustering(graph, n_clusters=1)
label_im = -np.ones(mask.shape)
label_im[mask] = labels

print(time.time() - time_current)

plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.imshow(img, cmap=plt.cm.spectral, interpolation='nearest')
plt.axis('off')
plt.subplot(122)
plt.imshow(label_im, cmap=plt.cm.spectral, interpolation='nearest')
plt.axis('off')

print(time.time() - time_current, 'Total time')

plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.01, right=0.99)
plt.show()