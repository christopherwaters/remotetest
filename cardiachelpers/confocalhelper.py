import scipy.ndimage
import numpy as np
from PIL import Image

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
				split_image[i] = image.copy()
			full_images.append(split_image)
		return(full_images)
	else:
		split_image = [None]*image_in.n_frames
		for i in range(image_in.n_frames):
			image_in.seek(i)
			split_image[i] = image_in.copy()
		return(split_image)

def chamferDist(image_in):
	dist_out = scipy.ndimage.morphology.distance_transform_cdt(image_in)
	return(dist_out)