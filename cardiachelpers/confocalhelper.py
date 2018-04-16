import scipy.ndimage
import numpy as np
from PIL import Image
from PIL import ImageOps
from cardiachelpers import importhelper
import skimage.filters
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path

def splitChannels(images_in, pull_channel=-1):
	"""Splits an image (or multiple images) to its different channels, then returns a nested list of the channels.
	"""
	# Determine if images in are a list or a single image
	if isinstance(images_in, list):
		split_images = [None]*len(images_in)
		# Iterate through images
		for i in range(len(images_in)):
			im = images_in[i]
			# Convert the image to an array and get the number of bands
			image_arr = np.array(im)
			num_channels = len(im.getbands())
			# Slice array based on channels selected (channel = -1 indicates all channels)
			if pull_channel >= 0 and pull_channel < num_channels:
				# Rebuild an image from the sliced array to isolate the channel
				split_images[i] = Image.fromarray(image_arr[:, :, pull_channel])
			else:
				# Create a new image for each channel and store as a list
				split_by_band = [None]*num_channels
				for chan_ind in range(num_channels):
					split_by_band[chan_ind] = Image.fromarray(image_arr[:, :, chan_ind])
				split_images[i] = split_by_band
		return(split_images)
	else:
		# Convert the image to an array and get the number of bands
		image_arr = np.array(images_in)
		num_channels = len(images_in.getbands())
		# Slice array based on channels selected (channel = -1 indicates all channels)
		if pull_channel >= 0 and pull_channel < num_channels:
			# Rebuild image from the sliced array to isolate the channel
			split_by_band = Image.fromarray(image_arr[:, :, pull_channel])
		else:
			# Create a new image for each channel and store as a list
			split_by_band = [None]*num_channels
			for chan_ind in range(num_channels):
				split_by_band[chan_ind] = Image.fromarray(image_arr[:, :, chan_ind])
		return(split_by_band)
		
def splitImageFrames(image_in):
	"""Splits an image (or multiple images as a list) to its different frames, and returns a list containing the images.
	"""
	# Determine if images in are a list or a single image
	if isinstance(image_in, list):
		full_images = []
		# Iterate through images, creating a sublist of frames for each image
		for image in image_in:
			split_image = [None]*image.n_frames
			# Iterate through frames and copy each frame independently, converting to RGB
			for i in range(image.n_frames):
				image.seek(i)
				split_image[i] = image.copy() if image.mode == 'RGB' else image.copy().convert(mode='RGB')
			full_images.append(split_image)
		return(full_images)
	else:
		split_image = [None]*image_in.n_frames
		# Iterate through frames and copy each frame independently, converting to RGB
		for i in range(image_in.n_frames):
			image_in.seek(i)
			split_image[i] = image_in.copy() if image_in.mode == 'RGB' else image_in.copy().convert(mode='RGB')
		return(split_image)

def stitchImages(images_in, image_x_inds, image_y_inds, overlap=0.1, save_pos=False):
	"""Piece images together based on x and y indices to form a single large image.
	"""
	# Get the maximum number of images in each direction
	x_range = max(image_x_inds)
	y_range = max(image_y_inds)
	# Pull the first image to be stitched
	first_sublist = images_in[0]
	while isinstance(first_sublist, list):
		first_sublist = first_sublist[0]
	# Get the size of the tiling from the input image
	if isinstance(first_sublist, Image.Image):
		tile_size = first_sublist.size
	else:
		return(False)
	# Create an image based on the width of the input image that should hold the full stitched image
	stitched_image = Image.new('RGB', (int(x_range*(1-overlap)*tile_size[0]), int(y_range*(1-overlap)*tile_size[1])))
	# Iterating through each image, copy the information to the stitched image based on position
	for image_num in range(len(images_in)):
		# Calculate X and Y positions for the image
		x_pos = int((x_range-image_x_inds[image_num])*(1-overlap)*tile_size[0])
		y_pos = int(image_y_inds[image_num]*(1-overlap)*tile_size[1])
		# Select the appropriate image to use
		cur_image = images_in[image_num]
		while isinstance(cur_image, list):
			cur_image = cur_image[0]
		if not isinstance(cur_image, Image.Image):
			return(False)
		# Paste the image into the large stitched image
		stitched_image.paste(cur_image, (x_pos, y_pos))
	# Determine if the image is being saved to disc or returned
	if save_pos:
		try:
			# Save the image to the indicated file
			stitched_image.save(save_pos)
			return(True)
		except Exception as e:
			raise(e)
	else:
		# Don't save the image, but return it as an Image object
		return(stitched_image)

def getImagePositions(image_files):
	"""Small function to pull image position data from Volocity-exported TIF files.
	"""
	# Set which data is desired
	data_categories = ['XLocationMicrons', 'YLocationMicrons', 'XCalibrationMicrons', 'YCalibrationMicrons']
	
	# Create array to store all categorical data for each image
	image_positions = np.empty((len(image_files), len(data_categories)))
	for file_num, tif_file in enumerate(image_files):
		with open(tif_file, encoding='utf8', errors='ignore') as temp_file:
			file_lines = temp_file.readlines()
			for line in file_lines:
				# Line split by '=' represents a property (if length = 2)
				line_split = line.split('=')
				if len(line_split) == 2:
					# If the category is in the list of desired categories, store it by appropriate column
					if line_split[0] in data_categories:
						image_positions[file_num, data_categories.index(line_split[0])] = float(line_split[1])
		
	# Create a dict object to represent which data is in each column
	column_dict = {data_categories[i] : i for i in range(len(data_categories))}
	return([image_positions, column_dict])

def getImageGrid(image_files, image_locs, locs_dict):
	"""Generate the x and y indices for image placement in stitching.
	
	X and Y indices are determined based on image data indicating absolute position and the "buckets" indicated
	image locations, pulled earlier from the files.
	"""
	# Determine how to find x and y positions
	x_col = locs_dict['XLocationMicrons']
	y_col = locs_dict['YLocationMicrons']
	# Pull absolute x and y locations from columns
	locs_x = image_locs[:, x_col]
	locs_y = image_locs[:, y_col]
	# Determine the possible slots for x and y data
	x_slots = np.unique(np.round(locs_x))
	y_slots = np.unique(np.round(locs_y))
	# Determine the relative positions for each image
	img_x_inds = [np.where(np.round(locs_x[i]) == x_slots)[0] for i in range(locs_x.shape[0])]
	img_y_inds = [np.where(np.round(locs_y[i]) == y_slots)[0] for i in range(locs_y.shape[0])]
	
	return(np.column_stack((img_x_inds, img_y_inds)))
	
def compressImages(images_in, image_scale=0.5):
	"""Resize the raw images in the model, to allow easier manipulation and display.
	
	Resets the compressed_images field on-call, to allow only one set of compressed images per instance.
	
	args:
		image_scale (float): Determines the ratio of new image size to old image size
	"""
	# Determine if input images are a list
	if isinstance(images_in, list):
		# Determine image size based on the ratio to the input image
		compressed_images = [None]*len(images_in)
		new_size = [int(image_scale*dimension) for dimension in images_in[0].size]
		# Iterate through images and resize using Lanczos reconstruction
		for image_num in range(len(images_in)):
			compressed_images[image_num] = images_in[image_num].resize(new_size, Image.LANCZOS)
		return(compressed_images)
	else:
		# Establish new size from ratio and resize image using Lanczos reconstruction
		new_size = [int(image_scale*dimension) for dimension in images_in.size]
		compressed_image = images_in.resize(new_size, Image.LANCZOS)
		return(compressed_image)
		
def readImageGrid(file_name):
	"""Read image grid information from a file instead of pulling it from image file data.
	"""
	im_grid = np.empty([])
	with open(file_name) as grid_file:
		# Iterate through each line of the input file
		for file_line in grid_file.readlines():
			# Split the line string using ',' as a delimiter, stripping whitespace and converting to int
			cur_inds = np.array([int(ind.strip()) for ind in file_line.split(',')])
			# Either create the new array or append locations to the growing array row-by-row
			if im_grid.ndim:
				im_grid = np.vstack((im_grid, cur_inds))
			else:
				im_grid = cur_inds
	return(im_grid)
	
def writeImageGrid(image_grid, file_name):
	"""Write image grid information to a file, to allow easier access.
	"""
	# Generate a blank file to use for grid storage
	open(file_name, 'w').close()
	with open(file_name, 'w') as grid_file:
		# Iterate through rows in the locations
		for row in range(image_grid.shape[0]):
			# Pull respective x and y indices from the input grid array
			x_ind = image_grid[row, 0]
			y_ind = image_grid[row, 1]
			# Write the locations to the file with a ',' separator and move to a new line
			grid_file.write(str(x_ind) + ',' + str(y_ind) + '\n')
	return(True)
	
def splitForeground(image_file):
	image_in = Image.open(image_file)
	image_split = image_in.split()
	mask_list = [Image.new('1', image_split[0].size)]*len(image_split)
	contours = [None]*len(image_split)
	for i, image_chan in enumerate(image_split):
		mask_list[i], contours[i] = _getThresholdMask(image_chan)
		
	save_dir, im_filename = os.path.split(image_file)
	file_names = [None]*len(mask_list)
	
	for i, mask in enumerate(mask_list):
		cur_file = im_filename.split('.')[0] + 'MaskChannel' + str(i) + '.png'
		filename = os.path.join(save_dir, cur_file)
		file_names[i] = filename
		plt.imsave(filename, mask, cmap=cm.gray)
	return([file_names, contours])
	
def _getThresholdMask(image_in):
	image_arr = np.array(image_in)
	
	if np.any(image_arr):
		thresh = skimage.filters.threshold_minimum(image_arr)
		mask = image_arr > thresh
		fr_filter_mask = skimage.filters.frangi(mask)
		fr_filter_thresh = skimage.filters.threshold_otsu(fr_filter_mask)*0.25
		contours = skimage.measure.find_contours(fr_filter_mask, fr_filter_thresh)
		fr_thresh_mask = fr_filter_mask > fr_filter_thresh
		return([fr_thresh_mask, contours])
	else:
		return([image_arr, [None]])
	
# Below here are attempts to do flat-field correction	
def getImageGradient(image_file):
	image_in = Image.open(image_file)
	image_arr = np.array(image_in)
	channel_arr = image_arr[:, :, 0]
	img_mean = int(np.mean(channel_arr))
	channel_ratio = img_mean / channel_arr
	return(channel_ratio)
	
def multiplyImageGradient(image_file, gradient_arr):
	image_in = Image.open(image_file)
	image_arr = np.array(image_in)
	mult_arr = np.zeros(image_arr.shape)
	num_chans = image_arr.shape[2]
	for chan in range(num_chans):
		chan_arr = image_arr[:, :, chan]
		mult_arr[:, :, chan] = np.multiply(chan_arr, gradient_arr)
	mult_arr = np.round(mult_arr).astype('uint8')
	ratio_image = Image.fromarray(mult_arr)
	return(ratio_image)
	
def _smoothGradient(x_inds, y_inds, gradient_arr):
	gradient_arr.shape
	