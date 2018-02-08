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
import exifread
import os

class ConfocalModel():
	"""Model class to hold confocal microscopy images and format them to generate a mesh to align with MRI data.
	"""

	def __init__(self):
		"""Initialize the model made to import confocal microscopy data.
		"""
		root = tk.Tk()
		frame = tk.Frame(root)
		frame.pack()
		
		# Select folder to import for confocal data
		confocal_folder = filedialog.askdirectory(title='Select confocal microscopy folder.')
		root.destroy()
		
		# Get all TIFF files in the 
		self.tif_files = glob.glob(os.path.join(confocal_folder, '*.tif'))
		self.raw_images = [None]*len(self.tif_files)
		
		# Open each image file using PIL
		for file_num in range(len(self.tif_files)):
			image_file = self.tif_files[file_num]
			self.raw_images[file_num] = Image.open(self.tif_files[file_num])
		
		# Import each file as an image itself
		self.channels = []
			
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
	
	def stitchImages(self, images=None):
		"""Stitch images together in grid, based on grid size definitions and image numbering.
		"""
		if not images:
			images = self.raw_images
		# Get the image positions and pixel-micron conversion data
		image_positions, column_dict = self._getImagePositions(self.tif_files)
		# Create a grid to match the number of positions possible, then fill images into the grid
		x_slots = np.unique(image_positions[:, 0])
		y_slots = np.unique(image_positions[:, 1])
		image_grid = np.empty((x_slots.size, y_slots.size))
		for image_num in range(image_positions.shape[0]):
			x_slot = np.where(x_slots == image_positions[image_num, column_dict['XLocationMicrons']])[0]
			y_slot = np.where(y_slots == image_positions[image_num, column_dict['YLocationMicrons']])[0]
			image_grid[x_slot, y_slot] = image_num
		return(image_grid)
	
	def validateIntensity(self):
		"""Adjust image intensity based on edge intensity of adacent images.
		"""
		pass
		
	def _getImagePositions(self, image_files=None):
		"""Small function to pull image position data from Volocity-exported TIF files.
		"""
		if not image_files:
			image_files = self.tif_files
		data_categories = ['XLocationMicrons', 'YLocationMicrons', 'XCalibrationMicrons', 'YCalibrationMicrons']
		image_positions = np.empty((len(image_files), len(data_categories)))
		for file_num, tif_file in enumerate(image_files):
			with open(tif_file, encoding='utf8', errors='ignore') as temp_file:
				file_lines = temp_file.readlines()
				for line in file_lines:
					line_split = line.split('=')
					if len(line_split) == 2:
						if line_split[0] in data_categories:
							image_positions[file_num, data_categories.index(line_split[0])] = float(line_split[1])
							
		column_dict = {data_categories[i] : i for i in range(len(data_categories))}
		return([image_positions, column_dict])