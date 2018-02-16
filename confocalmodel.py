# -*- coding: utf-8 -*-
"""
Contains information necessary for import and alignment of confocal microscopy images.

Created on Mon Jan 27 12:22:15 2018

@author: cdw2be
"""

# Imports
import math
import tkinter as tk
from tkinter import filedialog
import scipy as sp
import numpy as np
import matplotlib.pyplot as mplt
import glob
from PIL import Image
import os
from cardiachelpers import importhelper

class ConfocalModel():
	"""Model class to hold confocal microscopy images and format them to generate a mesh to align with MRI data.
	"""

	def __init__(self, confocal_dir):
		"""Initialize the model made to import confocal microscopy data.
		
		args:
			confocal_dir (string): The path to the directory containing tif image files
		"""
		
		# Get all TIFF files in the directory
		self.tif_files = glob.glob(os.path.join(confocal_dir, '*.tif'))
		self.raw_images = [None]*len(self.tif_files)
		
		# Open each image file using PIL
		for file_num in range(len(self.tif_files)):
			image_file = self.tif_files[file_num]
			self.raw_images[file_num] = Image.open(self.tif_files[file_num])
		
		# Import each file as an image itself
		self.compressImages()
		self.raw_images[0].show()
		self.compressed_images[0].show()
	
	def compressImages(self, image_scale=0.5):
		"""Resize the raw images in the model, to allow easier manipulation and display.
		
		Resets the compressed_images field on-call, to allow only one set of compressed images per instance.
		
		args:
			image_scale (float): Determines the ratio of new image size to old image size
		"""
		self.compressed_images = [None]*len(self.raw_images)
		new_size = [int(image_scale*dimension) for dimension in self.raw_images[0].size]
		if any(self.raw_images):
			for image_num in range(len(self.raw_images)):
				self.compressed_images[image_num] = self.raw_images[image_num].resize(new_size, Image.LANCZOS)
			return(True)
		else:
			return(False)
	
	def splitImageChannels(self, images=None):
		"""Split images by channels indicated in args.
		"""
		# Set images to internal images if nothing passed
		if not images:
			images = self.raw_images
		
		image_channels = [None]*len(images)
		
		for image_num in range(len(images)):
			channels = images[image_num].getbands()
			cur_channel = [None]*len(channels)
			for channel_ind in range(len(channels)):
				cur_channel[channel_ind] = images[image_num].getdata(channel_ind)
			image_channels[image_num] = cur_channel
			
		image_channels_arr = np.array(image_channels)
		return(image_channels_arr)
	
	def stitchImages(self, images=None, overlap=0.1):
		"""Stitch images together in grid, based on grid size definitions and image numbering.
		"""
		if not images:
			images = self.raw_images
		# Get the image positions and pixel-micron conversion data
		image_positions, column_dict = importhelper.getImagePositions(self.tif_files)
		# Create a grid to match the number of positions possible, then fill images into the grid
		x_slots = np.unique(image_positions[:, 0])
		y_slots = np.unique(image_positions[:, 1])
		image_grid = np.empty((x_slots.size, y_slots.size))
		for image_num in range(image_positions.shape[0]):
			x_slot = np.where(x_slots == image_positions[image_num, column_dict['XLocationMicrons']])[0]
			y_slot = np.where(y_slots == image_positions[image_num, column_dict['YLocationMicrons']])[0]
			image_grid[x_slot, y_slot] = image_num
		# Get the size in pixel count for the images
		image_x, image_y = images[0].size
		image_z = images[0].n_frames
		image_size = [image_x, image_y, image_z]
		return(image_grid)
	
	def validateIntensity(self):
		"""Adjust image intensity based on edge intensity of adacent images.
		"""
		pass