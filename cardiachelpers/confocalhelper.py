import scipy.ndimage
import numpy as np
from PIL import Image
from cardiachelpers import importhelper

def splitChannels(images_in, pull_channel=-1):
	"""Splits an image (or multiple images) to its different channels, then returns a nested list of the channels.
	"""
	if isinstance(images_in, list):
		split_images = [None]*len(images_in)
		for i in range(len(images_in)):
			im = images_in[i]
			image_arr = np.array(im)
			num_channels = len(im.getbands())
			if pull_channel >= 0 and pull_channel < num_channels:
				split_images[i] = Image.fromarray(image_arr[:, :, pull_channel])
			else:
				split_by_band = [None]*num_channels
				for chan_ind in range(num_channels):
					split_by_band[chan_ind] = Image.fromarray(image_arr[:, :, chan_ind])
				split_images[i] = split_by_band
		return(split_images)
	else:
		image_arr = np.array(images_in)
		num_channels = len(images_in.getbands())
		if pull_channel >= 0 and pull_channel < num_channels:
			split_by_band = Image.fromarray(image_arr[:, :, pull_channel])
		else:
			split_by_band = [None]*num_channels
			for chan_ind in range(num_channels):
				split_by_band[chan_ind] = Image.fromarray(image_arr[:, :, chan_ind])
		return(split_by_band)
		
def splitImageFrames(image_in):
	"""Splits an image (or multiple images as a list) to its different frames, and returns a list containing the images.
	"""
	if isinstance(image_in, list):
		full_images = []
		for image in image_in:
			split_image = [None]*image.n_frames
			for i in range(image.n_frames):
				split_image[i] = image.copy() if image.mode == 'RGB' else image.copy().convert(mode='RGB')
			full_images.append(split_image)
		return(full_images)
	else:
		split_image = [None]*image_in.n_frames
		for i in range(image_in.n_frames):
			image_in.seek(i)
			split_image[i] = image_in.copy() if image_in.mode == 'RGB' else image_in.copy().convert(mode='RGB')
		return(split_image)

def stitchImages(images_in, image_x_inds, image_y_inds, overlap=0.1, save_pos=False):
	"""Piece images together based on x and y indices to form a single large image.
	"""
	x_range = max(image_x_inds)
	y_range = max(image_y_inds)
	first_sublist = images_in[0]
	while isinstance(first_sublist, list):
		first_sublist = first_sublist[0]
	if isinstance(first_sublist, Image.Image):
		tile_size = first_sublist.size
	else:
		return(False)
	stitched_image = Image.new('RGB', (int(x_range*(1-overlap)*tile_size[0]), int(y_range*(1-overlap)*tile_size[1])))
	for image_num in range(len(images_in)):
		x_pos = int((x_range-image_x_inds[image_num])*(1-overlap)*tile_size[0])
		y_pos = int(image_y_inds[image_num]*(1-overlap)*tile_size[1])
		cur_image = images_in[image_num]
		while isinstance(cur_image, list):
			cur_image = cur_image[0]
		if not isinstance(cur_image, Image.Image):
			return(False)
		stitched_image.paste(cur_image, (x_pos, y_pos))
	if save_pos:
		stitched_image.save(save_pos)
		return(True)
	else:
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
	x_col = locs_dict['XLocationMicrons']
	y_col = locs_dict['YLocationMicrons']
	
	locs_x = image_locs[:, x_col]
	locs_y = image_locs[:, y_col]
	
	x_slots = np.unique(np.round(locs_x))
	y_slots = np.unique(np.round(locs_y))
	
	img_x_inds = [np.where(np.round(locs_x[i]) == x_slots)[0] for i in range(locs_x.shape[0])]
	img_y_inds = [np.where(np.round(locs_y[i]) == y_slots)[0] for i in range(locs_y.shape[0])]
	
	return(np.column_stack((img_x_inds, img_y_inds)))
	
def compressImages(images_in, image_scale=0.5):
	"""Resize the raw images in the model, to allow easier manipulation and display.
	
	Resets the compressed_images field on-call, to allow only one set of compressed images per instance.
	
	args:
		image_scale (float): Determines the ratio of new image size to old image size
	"""
	if isinstance(images_in, list):
		compressed_images = [None]*len(images_in)
		new_size = [int(image_scale*dimension) for dimension in images_in[0].size]
		for image_num in range(len(images_in)):
			compressed_images[image_num] = images_in[image_num].resize(new_size, Image.LANCZOS)
		return(compressed_images)
	else:
		new_size = [int(image_scale*dimension) for dimension in images_in.size]
		compressed_image = images_in.resize(new_size, Image.LANCZOS)
		return(compressed_image)