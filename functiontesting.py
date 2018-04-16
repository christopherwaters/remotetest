from cardiachelpers import confocalhelper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

image_file = 'C:/Users/cdw2be/Documents/PI Correction Base/PI-Corr Laser-100.tif'

img_gradient = confocalhelper.getImageGradient(image_file)

smoothed_gradient = confocalhelper._smoothGradient(list(range(img_gradient.shape[1])), list(range(img_gradient.shape[0])), img_gradient)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_inds, y_inds, img_gradient)

ratio_img = confocalhelper.multiplyImageGradient(image_file, img_gradient)