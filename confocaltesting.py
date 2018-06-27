from cardiachelpers import confocalhelper
import numpy as np
import matplotlib.pyplot as mplt
from mpl_toolkits.mplot3d import Axes3D

test_tif = 'C:/Users/cdw2be/Documents/pythoncardiacmodel/Test Data/fake_confocal_stack.tif'

test_list = confocalhelper.openModelImage(test_tif)

endo_traces = [None]*len(test_list)
epi_traces = [None]*len(test_list)
slice_gap = 10

for im_num, im_slice in enumerate(test_list):
	print(im_num)
	skeleton_image = confocalhelper.contourMaskImage(im_slice)
	print('Edges Traced')
	endo_path, epi_path, labelled_arr = confocalhelper.splitImageObjects(skeleton_image)
	print('Split Endo / Epi')
	endo_traces[im_num] = np.array(confocalhelper.smoothPathTrace(endo_path)).swapaxes(0, 1)
	print('Endo Smoothed')
	epi_traces[im_num] = np.array(confocalhelper.smoothPathTrace(epi_path)).swapaxes(0, 1)
	print('Epi Smoothed')

slice_gaps = [i*slice_gap for i in range(len(endo_traces))]
	
fig = mplt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(endo_traces)):
	ax.plot(endo_traces[i][:, 0], endo_traces[i][:, 1], [slice_gaps[i]]*endo_traces[i].shape[0])
	ax.plot(epi_traces[i][:, 0], epi_traces[i][:, 1], [slice_gaps[i]]*epi_traces[i].shape[0])