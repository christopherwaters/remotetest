import scipy.ndimage
import numpy as np
from PIL import Image

def splitChannels(images_in):
	"""Splits an image (or multiple images) to its different channels, then returns a nested list of the channels.
	"""
	if isinstance(images_in, list):
		split_images = None
		for image in images_in:
			num_bands = len(image.getbands())
			image_bands = [None]*num_bands
			for band in range(num_bands):
				image_bands[band] = image.getdata(band)
			if len(split_images):
				split_images.append(image_bands)
			else:
				split_images = [image_bands]
		return(split_images)
	else:
		num_bands = len(images_in.getbands())
		image_bands = [None]*num_bands
		for band in range(num_bands):
			image_bands[band] = images_in.getdata(band)
		return(image_bands)
		
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