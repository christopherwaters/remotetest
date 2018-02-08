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
	
	def stitchImages(self):
		"""Stitch images together in grid, based on grid size definitions and image numbering.
		"""
		with open(self.tif_files[0]) as temp_file:
			file_lines = temp_file.readlines()
			for line in file_lines:
				line_split = line.split('=')
				print(len(line_split))
		return(tags)
	
	def validateIntensity(self):
		"""Adjust image intensity based on edge intensity of adacent images.
		"""
		pass